"""Streaming ML_grad ptychographic reconstruction for the Holoscan pipeline.

This module owns the *streaming wrapper* around ptycho's reconstruction engine:
live-data lifecycle (per-scan object reset, incremental point growth, tail
clearing, mmap snapshots) and the single-GPU buffer/geometry setup. The actual
per-batch reconstruction math is **not reimplemented here** — it is ptycho
master's own, beamline-proven code, driven directly:

  - ``ptycho_trans.recon_ml_grad_trans_gpu_single`` — the ML_grad algorithm.
  - ``ptycho_trans.recon_gpu_launch``               — batch driver.
  - ``ptycho_trans.calc_prb_obj_gpu`` / ``forward_prop_gpu`` / ``back_prop_gpu``.

Those methods are bound onto this instance in :meth:`gpu_setup` (via
``getattr(ptycho_trans, name).__get__(self)``) so they run against this
object's buffers without us subclassing ``ptycho_trans`` at import time — which
would force ptycho/cupy/matplotlib imports just to construct the config object
(the pure-Python unit tests build a recon without a GPU). The object/probe
accumulate + gather steps remain local faithful single-rank ports of
``accumulate_obj_gpu``/``gather_obj_gpu`` etc. (no MPI), with the streaming
divide-by-zero guard the beamline batch code doesn't need.

ptycho is otherwise imported as a kernel library:

  - ``ptycho.cupy_util``        — pinned-memory helpers, block/grid sizing.
  - ``ptycho.prop_class_asm``   — angular-spectrum propagation.
  - ``ptycho.cupy_collection``  — compiled CuPy/RawModule kernels (incl. ML_grad).

The engine implements the subset Holoptycho actually uses:

  - **ML_grad algorithm** (Poisson), the HXN beamline default.
  - **Single probe mode, single object mode** (``prb_mode_num = obj_mode_num = 1``).
  - **No multislice, no nearfield, no Bragg, no partial coherence**.
  - **No position correction, no data refinement**.
  - **No MPI** — single process, single GPU.

The driven algorithm methods are character-identical to ptycho's
``HXN_production`` (see NSLS2/ptycho#87); only the streaming wrapper lives here.
"""

import logging

import numpy as np
from ptychoml import compute_sample_pixel_size

logger = logging.getLogger(__name__)


class StreamingPtychoRecon:
    """Streaming DM ptychographic reconstruction (single-mode, single-GPU).

    Lifecycle:
        1. ``__init__(config)``                — stash config; no GPU work.
        2. ``gpu_setup(num_points_max)``       — allocate GPU buffers + plans.
        3. ``reset_for_scan(...)``             — resize object arrays per scan.
        4. Per iteration, driven externally by PtychoRecon.compute():
           - ``clear_region(prev, new)``
           - ``initial_probe(num_diffs)``     (once per scan)
           - ``propagate_probe()``            (once, optional)
           - ``iter_once(it)``                (many times)
           - ``snapshot()``                   (for the live viewer)
        5. ``save_final(save_dir)``            — persist final recon.
    """

    def __init__(self, config):
        self.config = config

        # --- Core config ---
        self.nx_prb = int(config.nx)
        self.ny_prb = int(config.ny)
        self.n_iterations = int(config.n_iterations)
        self.gpu_batch_size = int(config.gpu_batch_size)
        self.init_prb_flag = bool(config.init_prb_flag)
        self.alpha = float(getattr(config, "alpha", 1e-3))
        self.beta = float(getattr(config, "beta", 0.9))
        self.sigma2 = float(getattr(config, "sigma2", 5e-5))

        # --- ML_grad algorithm config ---
        # recon_ml_grad_trans_gpu_single only implements the Poisson gradient.
        # refine_data_flag / nearfield_ptycho gate code paths in the inherited
        # method that holoptycho does not use; both stay False (their branches
        # are dead, so their kernels/buffers are never touched). alpha_ml is a
        # scalar the ML_grad step computes and stores each batch.
        self.ml_mode = str(getattr(config, "ml_mode", "Poisson"))
        self.refine_data_flag = False
        self.nearfield_ptycho = False
        self.alpha_ml = 0.0
        # Hold the probe fixed at the FFT-init estimate for the first
        # ``start_update_probe`` iterations while the object locks onto the
        # data. HXN_development:ptycho_trans_ml.py line 3546 ships this with
        # default 2; without it the probe update at iter 0 correlates against
        # a near-uniform seed object and the DM loop can diverge.
        self.start_update_probe = int(getattr(config, "start_update_probe", 2))
        self.start_update_object = int(getattr(config, "start_update_object", 0))
        self.mask_obj_flag = bool(getattr(config, "mask_obj_flag", False))
        self.amp_max = float(getattr(config, "amp_max", 1.0))
        self.amp_min = float(getattr(config, "amp_min", 0.5))
        self.pha_max = float(getattr(config, "pha_max", 0.01))
        self.pha_min = float(getattr(config, "pha_min", -1.0))
        self.precision = str(getattr(config, "precision", "single"))

        # --- GPU device ---
        self.gpu = int(config.gpus[0]) if config.gpus else 0

        # --- Streaming is single-mode only ---
        self.prb_mode_num = int(getattr(config, "prb_mode_num", 1))
        self.obj_mode_num = int(getattr(config, "obj_mode_num", 1))
        if self.prb_mode_num != 1 or self.obj_mode_num != 1:
            raise NotImplementedError(
                "StreamingPtychoRecon only supports single-mode "
                f"(prb_mode_num={self.prb_mode_num}, "
                f"obj_mode_num={self.obj_mode_num})"
            )

        # --- Geometry (computed in gpu_setup) ---
        self.x_pixel_m = 0.0
        self.y_pixel_m = 0.0
        self.obj_pad = int(getattr(config, "obj_pad", 4))
        self.x_range_um = 0.0
        self.y_range_um = 0.0
        self.nx_obj = 0
        self.ny_obj = 0
        self.scan_num = ""

        # Scan direction for THIS (iterative) engine. Follows the shared
        # x/y_direction unless the iterative-only override is set. No longer
        # copied onto PointProcessorOp (which keeps the shared convention for
        # the ViT positions_um); the iterative divergence reaches point_info
        # via point_proc.x/y_sign_rel — see ptycho_holo.compose().
        _xd = float(getattr(config, "x_direction", -1.0))
        _yd = float(getattr(config, "y_direction", -1.0))
        self.x_direction = float(getattr(config, "x_direction_iterative", _xd) or _xd)
        self.y_direction = float(getattr(config, "y_direction_iterative", _yd) or _yd)

        # Probe propagation distance (config stores it under 'distance' or
        # 'prb_prop_dist_um' depending on branch)
        self.prb_prop_dist_um = float(
            getattr(config, "prb_prop_dist_um", getattr(config, "distance", 0.0))
        )

        # --- Point counts ---
        self.num_points = 0
        self.num_points_l = 0
        self.num_points_recon = 0
        self.last = 0  # leftover batch remainder, updated per flush
        self.current_it = 0

        # --- Precision types (set in gpu_setup) ---
        self.float_precision = None
        self.complex_precision = None

        # --- GPU buffers (allocated in gpu_setup) ---
        self.prb_d = None
        self.obj_d = None
        self.diff_d = None
        self.point_info_d = None
        self.product_d = None
        self.prb_obj_d = None
        self.fft_tmp_d = None
        self.amp_tmp_d = None
        # ML_grad gradient scratch (batch-sized), HXN ref: gpu_init tmp1_d/tmp2_d
        self.tmp1_d = None
        self.tmp2_d = None
        self.obj_upd_d = None
        self.prb_upd_d = None
        self.prb_norm_d = None
        self.obj_norm_d = None
        self.prb_sqr_d = None
        self.prb_conj_d = None
        self.obj_sqr_d = None
        self.obj_conj_d = None

        # --- Pinned host shadows ---
        self.prb_mode = None
        self.obj_mode = None
        self.obj_update_l = None
        self.prb_update_l = None
        self.prb_norm_l = None
        self.obj_norm_l = None

        # --- Zero-state flat views used by reset_for_scan ---
        self.obj_mode_0 = None
        self.obj_d_0 = None
        self.obj_upd_d_0 = None
        self.prb_norm_d_0 = None
        self.obj_update_l_0 = None
        self.prb_norm_l_0 = None

        # --- Shared-memory-equivalent output arrays (plain numpy; Holoptycho
        # reads them in-process via SaveLiveResult). ---
        self.mmap_prb = None
        self.mmap_obj = None
        self._mmap_obj_nx_base = 0
        self._mmap_obj_ny_base = 0

        # --- cuFFT plans ---
        self.cufft_plan = None
        self.cufft_plan_last = None

        # --- Kernel handles (set in gpu_setup) ---
        self.kernel_multiply_with_support_mode = None
        self.kernel_restrict_range = None
        self.kernel_get_amp = None
        self.kernel_ml_calc_grad = None
        self.kernel_multiply_with_mode = None
        self.kernel_accumulate_obj_mode = None
        self.kernel_accumulate_prb_mode = None

    def _required_object_shape(self, x_range_um: float, y_range_um: float) -> tuple[int, int]:
        """Compute the object array shape required for a scan region.

        This uses the same geometry-driven sizing for both preallocation in
        ``gpu_setup()`` and per-scan validation in ``reset_for_scan()`` so the
        two paths cannot drift apart.
        """
        if self.x_pixel_m <= 0 or self.y_pixel_m <= 0:
            raise RuntimeError(
                "Object shape cannot be computed before pixel sizes are initialized."
            )

        nx_obj = int(
            self.nx_prb + np.ceil(abs(x_range_um) * 1e-6 / self.x_pixel_m) + self.obj_pad
        )
        ny_obj = int(
            self.ny_prb + np.ceil(abs(y_range_um) * 1e-6 / self.y_pixel_m) + self.obj_pad
        )
        nx_obj += nx_obj % 2
        ny_obj += ny_obj % 2
        return nx_obj, ny_obj

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def gpu_setup(self, num_points_max):
        """Allocate all GPU buffers, pinned host buffers, cuFFT plans, and
        kernel handles. Called once per process.

        HXN ref: ``ptycho_trans.gpu_init`` lines 1945-2110 of
        ``HXN_development``'s ``ptycho_trans_ml.py``. This is the single-mode
        DM subset only (no multislice, no ML_grad, no holoscan_flag
        gather buffers).
        """
        import cupy as cp
        import cupyx.scipy.fftpack as cufft
        from numba import cuda as numba_cuda

        from ptycho.cupy_util import (
            get_page_locked_array,
            convert_dtype_to_c_type_str,
        )
        from ptycho.cupy_collection import cupy_mod, get_mod2, get_amp
        # Deferred here (not at module import): pulling ptycho_trans imports the
        # whole ptycho/cupy stack, which must not happen just to construct the
        # config object in the pure-Python unit tests.
        from ptycho.ptycho_trans_ml import ptycho_trans

        # --- Select GPU ---
        cp.cuda.Device(self.gpu).use()
        numba_cuda.select_device(self.gpu)
        cp.cuda.set_pinned_memory_allocator()

        # --- Precision ---
        if self.precision == "double":
            self.float_precision = np.float64
            self.complex_precision = np.complex128
        else:
            self.float_precision = np.float32
            self.complex_precision = np.complex64

        # --- Pixel sizes (lambda_nm * z_m / (nx_prb * ccd_pixel_um * 1e-6)) ---
        # HXN ref: ``ptycho_trans.setup`` ~line 4210-4220.
        lambda_nm = getattr(self.config, "lambda_nm", None)
        if lambda_nm is None or lambda_nm == 0:
            # lambda_nm = 1.2398 / xray_energy_kev
            xray_energy_kev = float(self.config.xray_energy_kev)
            lambda_nm = 1.2398 / xray_energy_kev if xray_energy_kev else 0.0
        self.lambda_nm = float(lambda_nm)
        z_m = float(getattr(self.config, "z_m", 0.0))
        ccd_pixel_um = float(getattr(self.config, "ccd_pixel_um", 55.0))
        if z_m and ccd_pixel_um and self.nx_prb and self.ny_prb:
            wavelength_m = self.lambda_nm * 1e-9
            ccd_pixel_m = ccd_pixel_um * 1e-6
            self.x_pixel_m = compute_sample_pixel_size(
                wavelength_m, z_m, ccd_pixel_m, self.nx_prb
            )
            self.y_pixel_m = compute_sample_pixel_size(
                wavelength_m, z_m, ccd_pixel_m, self.ny_prb
            )
        else:
            # Not enough info — leave as zero; caller can set before use.
            self.x_pixel_m = 0.0
            self.y_pixel_m = 0.0

        self.num_points_l = int(num_points_max)
        self.num_points = int(num_points_max)
        self.num_points_recon = int(num_points_max)
        self.last = self.num_points_recon % self.gpu_batch_size

        mp = self.prb_mode_num
        mo = self.obj_mode_num
        nx = self.nx_prb
        ny = self.ny_prb
        b = self.gpu_batch_size

        # --- Data buffers ---
        self.diff_d = cp.empty(
            (self.num_points_l, nx, ny), dtype=self.float_precision, order="C"
        )
        # MUST be zeros, not empty: iter_once's it==0 init kernel processes
        # ALL num_points_l slots, including ones whose windows haven't been
        # written yet. Garbage int32s from recycled pool memory become object
        # array indices -> nondeterministic cudaErrorIllegalAddress. A zero
        # window reads obj[0:nx, 0:ny], which is always in bounds.
        self.point_info_d = cp.zeros(
            (self.num_points_l, 4), dtype=cp.int32, order="C"
        )

        # --- Iteration working buffers (single-mode ML_grad) ---
        self.prb_obj_d = cp.empty(
            (b * mp * mo, nx, ny), dtype=self.complex_precision, order="C"
        )
        self.amp_tmp_d = cp.empty(
            (b * mp * mo, nx, ny), dtype=self.float_precision, order="C"
        )
        self.product_d = cp.empty(
            (self.num_points_l, mp, mo, nx, ny),
            dtype=self.complex_precision,
            order="C",
        )
        self.fft_tmp_d = cp.empty(
            (b * mp * mo, nx, ny), dtype=self.complex_precision, order="C"
        )
        # ML_grad gradient scratch (diff_int / model_int), batch-sized float.
        # HXN ref: gpu_init tmp1_d/tmp2_d, used by recon_ml_grad_trans_gpu_single.
        self.tmp1_d = cp.empty((b, nx, ny), dtype=self.float_precision, order="C")
        self.tmp2_d = cp.empty((b, nx, ny), dtype=self.float_precision, order="C")

        # --- Probe + object GPU buffers ---
        # Preallocate for the configured scan geometry so reset_for_scan() can
        # reuse the same buffers without hitting heuristic size mismatches.
        x_range_um = float(abs(getattr(self.config, "x_range", 0.0)))
        y_range_um = float(abs(getattr(self.config, "y_range", 0.0)))
        nx_obj_req, ny_obj_req = self._required_object_shape(x_range_um, y_range_um)
        nx_obj_max = max(nx_obj_req, nx * 8, 1024)
        ny_obj_max = max(ny_obj_req, ny * 8, 1024)
        self.nx_obj = nx_obj_max
        self.ny_obj = ny_obj_max

        self.prb_d = cp.empty(
            (mp, nx, ny), dtype=self.complex_precision, order="C"
        )
        self.obj_d = cp.empty(
            (mo, nx_obj_max, ny_obj_max), dtype=self.complex_precision, order="C"
        )
        self.obj_upd_d = cp.empty(
            (mo, nx_obj_max, ny_obj_max), dtype=self.complex_precision, order="C"
        )
        self.prb_norm_d = cp.empty(
            (mo, nx_obj_max, ny_obj_max), dtype=self.float_precision, order="C"
        )
        self.prb_upd_d = cp.empty(
            (mp, nx, ny), dtype=self.complex_precision, order="C"
        )
        self.obj_norm_d = cp.empty(
            (mp, nx, ny), dtype=self.float_precision, order="C"
        )

        # --- Pinned host shadows ---
        self.prb_mode = get_page_locked_array(
            (mp, nx, ny), dtype=self.complex_precision
        )
        self.obj_mode = get_page_locked_array(
            (mo, nx_obj_max, ny_obj_max), dtype=self.complex_precision
        )
        self.prb_update_l = get_page_locked_array(
            (mp, nx, ny), dtype=self.complex_precision
        )
        self.obj_update_l = get_page_locked_array(
            (mo, nx_obj_max, ny_obj_max), dtype=self.complex_precision
        )
        self.prb_norm_l = get_page_locked_array(
            (mo, nx_obj_max, ny_obj_max), dtype=self.float_precision
        )
        self.obj_norm_l = get_page_locked_array(
            (mp, nx, ny), dtype=self.float_precision
        )

        # --- cuFFT plan ---
        self.cufft_plan = cufft.get_fft_plan(
            self.fft_tmp_d[0 : self.gpu_batch_size * mp * mo], axes=(1, 2)
        )
        if self.last > 0:
            self.cufft_plan_last = cufft.get_fft_plan(
                self.fft_tmp_d[0 : self.last * mp * mo], axes=(1, 2)
            )

        # --- Kernels ---
        dtype_str = convert_dtype_to_c_type_str(self.float_precision)
        self.kernel_multiply_with_support_mode = cupy_mod.get_function(
            f"multiply_with_support_mode<{dtype_str}>"
        )
        self.kernel_restrict_range = cupy_mod.get_function(
            f"restrict_range<{dtype_str}>"
        )
        # ML_grad kernels (added to ptycho master in NSLS2/ptycho#87).
        self.kernel_multiply_with_mode = cupy_mod.get_function(
            f"multiply_with_mode<{dtype_str}>"
        )
        self.kernel_ml_calc_grad = cupy_mod.get_function(
            f"ml_calc_grad<{dtype_str}>"
        )
        self.kernel_get_amp = get_amp(mp * mo)

        func_mod = get_mod2(mp, mo)
        self.kernel_accumulate_prb_mode = func_mod.get_function(
            f"accumulate_prb_mode<{dtype_str}, {mp}, {mo}>"
        )
        self.kernel_accumulate_obj_mode = func_mod.get_function(
            f"accumulate_obj_mode<{dtype_str}, {mp}, {mo}>"
        )

        # Bind ptycho master's ML_grad reconstruction methods onto this instance
        # so they execute against our buffers — verbatim beamline code, no
        # reimplementation (NSLS2/ptycho#87). refine_data_gpu is intentionally
        # not bound: it is only reached when refine_data_flag is True (kept False).
        for _name in (
            "recon_gpu_launch",
            "recon_ml_grad_trans_gpu_single",
            "calc_prb_obj_gpu",
            "forward_prop_gpu",
            "back_prop_gpu",
        ):
            setattr(self, _name, getattr(ptycho_trans, _name).__get__(self))

        # --- mmap buffers for live snapshots (plain numpy; Holoptycho reads
        # them in-process via SaveLiveResult) ---
        # init_mmap crops the object by obj_pad + probe size in each dim.
        mmap_obj_nx = nx_obj_max - self.obj_pad - nx
        mmap_obj_ny = ny_obj_max - self.obj_pad - ny
        self.mmap_prb = np.empty(
            (self.n_iterations, mp, nx, ny), dtype=self.complex_precision
        )
        self.mmap_obj = np.empty(
            (self.n_iterations, mo, max(mmap_obj_nx, 1), max(mmap_obj_ny, 1)),
            dtype=self.complex_precision,
        )
        self._mmap_obj_nx_base = nx_obj_max
        self._mmap_obj_ny_base = ny_obj_max

        # --- Initial probe/object values (random-ish, will be reset per scan) ---
        self.prb_mode[...] = 0
        self.obj_mode[...] = 0.99 * np.exp(-0.1j)

        # --- Save flat "_0" views for reset_for_scan to reshape later ---
        self.obj_mode_0 = self.obj_mode.reshape(self.obj_mode.size, order="C")
        self.obj_d_0 = self.obj_d.reshape(self.obj_d.size, order="C")
        self.obj_upd_d_0 = self.obj_upd_d.reshape(self.obj_upd_d.size, order="C")
        self.prb_norm_d_0 = self.prb_norm_d.reshape(
            self.prb_norm_d.size, order="C"
        )
        self.obj_update_l_0 = self.obj_update_l.reshape(
            self.obj_update_l.size, order="C"
        )
        self.prb_norm_l_0 = self.prb_norm_l.reshape(
            self.prb_norm_l.size, order="C"
        )

    # ------------------------------------------------------------------
    # Per-scan lifecycle
    # ------------------------------------------------------------------

    def reset_for_scan(self, scan_num, x_range_um, y_range_um, num_points_max):
        """Reset state for a new scan region without reallocating GPU buffers.

        HXN refs:
          - ``flush_live_recon`` lines 4025-4036
          - ``new_obj`` lines 3892-3924
          - ``reset_obj`` lines 3926-3930
          - ``init_auxiliary_arrays`` (single-rank subset) lines 4077-4155
        """
        from ptycho.cupy_util import copy_from_pinned

        self.scan_num = str(scan_num)
        self.x_range_um = float(abs(x_range_um))
        self.y_range_um = float(abs(y_range_um))
        self.num_points = int(num_points_max)
        # Note: do NOT mutate self.num_points_l here. It is the dimension of the
        # GPU buffers (product_d, prb_obj_d, etc.) that were allocated once in
        # gpu_setup() and intentionally not reallocated per scan. Shrinking it
        # here desyncs the product_d reshape in the recon batch from the actual
        # buffer size and triggers a ValueError on the first iteration.
        self.num_points_recon = 0
        self.last = self.num_points_recon % self.gpu_batch_size
        self.current_it = 0

        # --- Compute new obj dimensions (HXN ref: flush_live_recon) ---
        nx_obj, ny_obj = self._required_object_shape(self.x_range_um, self.y_range_um)

        if nx_obj > self._mmap_obj_nx_base or ny_obj > self._mmap_obj_ny_base:
            raise RuntimeError(
                f"New scan dimensions ({nx_obj}x{ny_obj}) exceed pre-allocated "
                f"maximum ({self._mmap_obj_nx_base}x{self._mmap_obj_ny_base}). "
                "Increase the object preallocation for this scan geometry."
            )
        self.nx_obj = nx_obj
        self.ny_obj = ny_obj
        logger.info("StreamingPtychoRecon: Obj dim %s %s", nx_obj, ny_obj)

        # --- Reshape the flat "_0" views into the new (mo, nx_obj, ny_obj) shape ---
        mo = self.obj_mode_num
        n = mo * nx_obj * ny_obj
        self.obj_mode = self.obj_mode_0[:n].reshape(
            (mo, nx_obj, ny_obj), order="C"
        )
        self.obj_d = self.obj_d_0[:n].reshape((mo, nx_obj, ny_obj), order="C")
        self.obj_upd_d = self.obj_upd_d_0[:n].reshape(
            (mo, nx_obj, ny_obj), order="C"
        )
        self.prb_norm_d = self.prb_norm_d_0[:n].reshape(
            (mo, nx_obj, ny_obj), order="C"
        )
        self.obj_update_l = self.obj_update_l_0[:n].reshape(
            (mo, nx_obj, ny_obj), order="C"
        )
        self.prb_norm_l = self.prb_norm_l_0[:n].reshape(
            (mo, nx_obj, ny_obj), order="C"
        )

        # --- Zero the accumulation buffers ---
        self.obj_upd_d[...] = 0
        self.prb_norm_d[...] = 0
        self.obj_update_l[...] = 0
        self.prb_norm_l[...] = 0

        # --- Re-initialise the object with a uniform random phase seed ---
        # HXN ref: reset_obj + init_obj_function
        for i in range(mo):
            self.obj_mode[i, :, :] = 0.99 * np.exp(-0.1j)
        copy_from_pinned(self.obj_mode, self.obj_d, self.obj_mode.nbytes)

        # --- Re-shape the mmap object buffer for the new dimensions ---
        mmap_obj_nx = max(nx_obj - self.obj_pad - self.nx_prb, 1)
        mmap_obj_ny = max(ny_obj - self.obj_pad - self.ny_prb, 1)
        self.mmap_obj = np.empty(
            (self.n_iterations, mo, mmap_obj_nx, mmap_obj_ny),
            dtype=self.complex_precision,
        )

    # ------------------------------------------------------------------
    # Probe initialisation
    # ------------------------------------------------------------------

    def initial_probe(self, num_diffs):
        """Initialise the probe.

        With ``init_prb_flag=False`` and a readable ``prb_path``, warm-start
        from that ``.npy`` file (a probe from a prior reconstruction, e.g. the
        beamline GUI's mADMM result — shape ``(modes, nx, ny)`` or ``(nx, ny)``,
        complex). Otherwise compute it from the first ``num_diffs`` diffraction
        patterns.

        HXN ref: ``init_live_prb`` lines 3960-3979. Streaming-mode subset:
        single-mode only, no ``shift_sum`` for secondary modes.
        """
        import cupy as cp

        from ptycho.cupy_util import copy_to_pinned

        prb_path = str(getattr(self.config, "prb_path", "") or "")
        if not self.init_prb_flag and prb_path:
            prb_file = np.load(prb_path)
            prb_file = np.asarray(prb_file)
            if prb_file.ndim == 3:
                prb_file = prb_file[0]  # first mode (streaming is single-mode)
            if prb_file.shape != (self.nx_prb, self.ny_prb):
                raise ValueError(
                    f"prb_path {prb_path!r} has shape {prb_file.shape}; "
                    f"expected ({self.nx_prb}, {self.ny_prb}) to match nx/ny"
                )
            self.prb_d[0, :, :] = cp.asarray(
                prb_file.astype(self.complex_precision)
            )
            copy_to_pinned(self.prb_d, self.prb_mode, self.prb_d.nbytes)
            logger.info("initial_probe: warm-started from %s", prb_path)
            return
        if not self.init_prb_flag:
            logger.warning(
                "initial_probe: init_prb_flag is False but no prb_path is "
                "set; falling back to probe-from-diff."
            )

        prb = cp.fft.fftshift(
            cp.fft.ifftn(cp.mean(self.diff_d[:num_diffs, :, :], axis=0))
        ) * cp.sqrt(self.nx_prb * self.ny_prb)
        self.prb_d[0, :, :] = prb
        copy_to_pinned(self.prb_d, self.prb_mode, self.prb_d.nbytes)

    def propagate_probe(self):
        """Propagate the current probe by ``self.prb_prop_dist_um`` metres.

        HXN ref: ``propagate_prb`` lines 822-833. Single-mode.
        """
        from ptycho.cupy_util import copy_from_pinned
        from ptycho.prop_class_asm import propagate

        self.prb_mode[0] = propagate(
            self.prb_mode[0],
            direction=1,
            x_pixel_size_m=self.x_pixel_m,
            y_pixel_size_m=self.y_pixel_m,
            wavelength_m=self.lambda_nm * 1e-9,
            z_m=self.prb_prop_dist_um * 1e-6,
            xp=np,
        )
        copy_from_pinned(self.prb_mode, self.prb_d, self.prb_d.nbytes)

    # ------------------------------------------------------------------
    # Object-region management
    # ------------------------------------------------------------------

    def clear_region(self, prev, new):
        """Zero the object region covered by ``point_info_d[prev:new]``.

        HXN ref: ``clear_obj_tail`` lines 3932-3941.
        """
        import cupy as cp

        from ptycho.cupy_util import copy_from_pinned

        x_low = int(cp.min(self.point_info_d[prev:new, 0]))
        x_high = int(cp.max(self.point_info_d[prev:new, 1]))
        y_low = int(cp.min(self.point_info_d[prev:new, 2]))
        y_high = int(cp.max(self.point_info_d[prev:new, 3]))

        mo = self.obj_mode_num
        for i in range(mo):
            self.obj_mode[i, x_low:x_high, y_low:y_high] = 0.99 * np.exp(-0.1j)
        copy_from_pinned(self.obj_mode, self.obj_d, self.obj_mode.nbytes)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def iter_once(self, it):
        """Run one ML_grad iteration on the points currently loaded.

        Streaming equivalent of HXN ``one_iter``: it drives ptycho's exit-wave
        update then the object/probe gradient step. Calls:
          - ``_update_psi(it)``                     — ML_grad via recon_gpu_launch
          - ``_accumulate_obj() + _gather_obj()``   — gather_obj_gpu (grad_upd)
          - ``_accumulate_prb() + _gather_prb()``   — gather_prb_gpu (grad_upd)
          - ``_update_mmap(it)``                    — snapshot to numpy

        Skips GUI/preview/error-calculation paths that Holoptycho doesn't use.
        """
        self.current_it = it

        # exit-wave update
        self._update_psi(it)

        # probe + object gradient updates. Gating mirrors HXN_development's
        # update_prb_obj: with the default ``start_update_probe = 2`` the probe
        # is held fixed at the FFT init for the first 2 iterations while the
        # object updates against the measured intensities. Without this gate,
        # probe and object update together from a near-uniform seed and the
        # gradient loop diverges.
        if it >= self.start_update_object:
            self._accumulate_obj()
            self._gather_obj()
        if it >= self.start_update_probe:
            self._accumulate_prb()
            self._gather_prb()

        # snapshot into mmap arrays (for SaveLiveResult)
        self._update_mmap(it)

    def _update_psi(self, it):
        """Update the exit wave via ptycho master's ML_grad algorithm.

        Drives ``recon_gpu_launch(recon_ml_grad_trans_gpu_single)`` — both bound
        from ``ptycho_trans`` in :meth:`gpu_setup` — so the per-batch math is
        ptycho's verbatim beamline code, not a reimplementation. Equivalent to
        HXN ``update_psi_gpu`` dispatching ``alg == 'ML_grad'``.

        Unlike DM, ML_grad does not carry ``product_d`` as running dual state:
        ``recon_ml_grad_trans_gpu_single`` recomputes ``prb_obj`` from the
        current probe/object each call and overwrites its ``product_d`` slice, so
        no it==0 / newly-streamed-point product initialisation is needed here.
        """
        # recon_gpu_launch reads self.last for the remainder batch.
        self.last = self.num_points_recon % self.gpu_batch_size
        self.recon_gpu_launch(self.recon_ml_grad_trans_gpu_single)

    def init_product_range(self, start, end):
        """Initialize product_d = prb * obj(window) for points [start, end).

        Called when newly streamed points activate. Same kernel call as the
        it==0 initialization in ``_update_psi``, but scoped to the new range —
        without it, points arriving after iteration 0 carry a stale dual
        state (computed from a zero window before their positions existed)
        for the rest of the run.
        """
        from ptycho.cupy_util import get_3d_block_grid_config

        count = int(end) - int(start)
        if count <= 0:
            return
        nx, ny = self.nx_prb, self.ny_prb
        block, grid = get_3d_block_grid_config((count, nx, ny))
        args = (
            self.prb_d,
            self.obj_d,
            self.product_d,
            self.point_info_d,
            nx,
            ny,
            self.prb_mode_num,
            self.nx_obj,
            self.ny_obj,
            self.obj_mode_num,
            int(start),
            count,
        )
        self.kernel_multiply_with_support_mode(grid, block, args)

    def _accumulate_obj(self):
        """HXN ref: ``accumulate_obj_gpu`` lines 2783-2799."""
        import cupy as cp

        from ptycho.cupy_util import get_3d_block_grid_config

        n_batch = self.num_points_recon // self.gpu_batch_size
        self.obj_upd_d[...] = 0.0
        self.prb_norm_d[...] = self.float_precision(self.alpha)
        self.prb_sqr_d = cp.abs(self.prb_d) ** 2
        self.prb_conj_d = cp.conj(self.prb_d)
        block, grid = get_3d_block_grid_config(
            (self.gpu_batch_size, self.nx_prb, self.ny_prb)
        )
        for i in range(n_batch):
            self._accumulate_obj_single(
                i * self.gpu_batch_size, self.gpu_batch_size, block, grid
            )
        if self.last > 0:
            block, grid = get_3d_block_grid_config(
                (self.last, self.nx_prb, self.ny_prb)
            )
            self._accumulate_obj_single(
                n_batch * self.gpu_batch_size, self.last, block, grid
            )

    def _accumulate_obj_single(self, start_point, batch_size, block, grid):
        """HXN ref: ``accumulate_obj_gpu_single`` lines 1877-1890."""
        import cupy as cp

        mp = self.prb_mode_num
        mo = self.obj_mode_num
        # Pass the FULL product array: the accumulate kernels index product_d
        # with the GLOBAL point index (start + z). Passing a slice already
        # offset to start_point double-offsets the reads — wrong points'
        # data for every batch after the first (corrupted object updates),
        # and out-of-bounds (cudaErrorIllegalAddress) once
        # start_point >= num_points_l / 2.
        psi = self.product_d.reshape(
            self.num_points_l * mp * mo, self.nx_prb, self.ny_prb
        )
        args = (
            self.prb_norm_d,
            self.obj_upd_d,
            self.prb_sqr_d,
            self.prb_conj_d,
            psi,
            self.point_info_d,
            cp.uint32(self.nx_prb),
            cp.uint32(self.ny_prb),
            cp.uint32(self.nx_obj),
            cp.uint32(self.ny_obj),
            cp.uint32(start_point),
            cp.uint32(batch_size),
        )
        self.kernel_accumulate_obj_mode(grid, block, args)

    def _gather_obj(self):
        """Single-rank flavour of ``gather_obj_gpu`` (grad_upd path).

        Skips the MPI Allreduce and NCCL branches; we're single-process.
        Also skips the ``mask_obj_flag`` kernel by default — re-enable
        below if your config sets it.
        """
        import cupy as cp

        from ptycho.cupy_util import copy_to_pinned, copy_from_pinned

        copy_to_pinned(self.obj_upd_d, self.obj_update_l, self.obj_upd_d.nbytes)
        copy_to_pinned(self.prb_norm_d, self.prb_norm_l, self.prb_norm_d.nbytes)

        # ML_grad is a gradient method: HXN's gather runs the grad_upd branch
        # (update_psi_gpu sets grad_upd=True for alg=='ML_grad'), i.e.
        #   obj_mode += obj_update_l / prb_norm_l
        # NOT the DM-style replace. Guarded against 0/0 where no probe has
        # covered a pixel yet (prb_norm_l ~ 0 if alpha is tiny); those pixels
        # are left unchanged so the streaming dashboard never sees NaNs.
        quotient = np.zeros_like(self.obj_update_l)
        np.divide(
            self.obj_update_l,
            self.prb_norm_l,
            out=quotient,
            where=self.prb_norm_l > 0,
        )
        self.obj_mode += quotient

        copy_from_pinned(self.obj_mode, self.obj_d, self.obj_mode.nbytes)

        if self.mask_obj_flag:
            mo = self.obj_mode_num
            o_nx = np.int32(self.nx_obj)
            o_ny = np.int32(self.ny_obj)
            amp_max = self.float_precision(self.amp_max)
            amp_min = self.float_precision(self.amp_min)
            pha_max = self.float_precision(self.pha_max)
            pha_min = self.float_precision(self.pha_min)
            N_block = 256
            N_grid = int((mo * o_nx * o_ny - 1) // N_block + 1)
            args = (
                self.obj_d,
                cp.uint32(mo),
                cp.uint32(o_nx),
                cp.uint32(o_ny),
                amp_max,
                amp_min,
                pha_max,
                pha_min,
            )
            self.kernel_restrict_range((N_grid, 1, 1), (N_block, 1, 1), args)
            copy_to_pinned(self.obj_d, self.obj_mode, self.obj_mode.nbytes)

    def _accumulate_prb(self):
        """HXN ref: ``accumulate_prb_gpu`` lines 2857-2870."""
        import cupy as cp

        from ptycho.cupy_util import get_3d_block_grid_config

        self.prb_upd_d[...] = 0.0
        self.obj_norm_d[...] = self.float_precision(self.alpha)
        self.obj_sqr_d = cp.abs(self.obj_d) ** 2
        self.obj_conj_d = self.obj_d.conj()
        n_batch = self.num_points_recon // self.gpu_batch_size
        block, grid = get_3d_block_grid_config(
            (self.gpu_batch_size, self.nx_prb, self.ny_prb)
        )
        for i in range(n_batch):
            self._accumulate_prb_single(
                i * self.gpu_batch_size, self.gpu_batch_size, block, grid
            )
        if self.last > 0:
            block, grid = get_3d_block_grid_config(
                (self.last, self.nx_prb, self.ny_prb)
            )
            self._accumulate_prb_single(
                n_batch * self.gpu_batch_size, self.last, block, grid
            )

    def _accumulate_prb_single(self, start_point, batch_size, block, grid):
        """HXN ref: ``accumulate_prb_gpu_single`` lines 1892-1915."""
        import cupy as cp

        mp = self.prb_mode_num
        mo = self.obj_mode_num
        # Full array, NOT a slice — accumulate_prb_mode indexes product_d
        # with the global point index (start + z); see _accumulate_obj_single.
        psi = self.product_d.reshape(
            self.num_points_l * mp * mo, self.nx_prb, self.ny_prb
        )
        args = (
            self.obj_norm_d,
            self.prb_upd_d,
            self.obj_sqr_d,
            self.obj_conj_d,
            psi,
            self.point_info_d,
            cp.uint32(self.nx_prb),
            cp.uint32(self.ny_prb),
            cp.uint32(self.nx_obj),
            cp.uint32(self.ny_obj),
            cp.uint32(start_point),
            cp.uint32(batch_size),
        )
        self.kernel_accumulate_prb_mode(grid, block, args)

    def _gather_prb(self):
        """Single-rank flavour of ``gather_prb_gpu`` (grad_upd path)."""
        from ptycho.cupy_util import copy_to_pinned, copy_from_pinned

        copy_to_pinned(self.obj_norm_d, self.obj_norm_l, self.obj_norm_d.nbytes)
        copy_to_pinned(self.prb_upd_d, self.prb_update_l, self.prb_upd_d.nbytes)

        # ML_grad grad_upd path: prb_mode += prb_update_l / obj_norm_l (not the
        # DM-style replace). obj_norm_l >= alpha > 0 over the probe window, so no
        # divide-by-zero guard is needed here.
        self.prb_mode += self.prb_update_l / self.obj_norm_l

        copy_from_pinned(self.prb_mode, self.prb_d, self.prb_mode.nbytes)

    def _update_mmap(self, it):
        """Snapshot the current probe/object into the mmap numpy buffers.

        HXN ref: the mmap-writing block around line 875 of ``ptycho_trans_ml.py``.
        Single-mode: prb_mode is on the host (pinned), obj_mode is on the host
        too. For multi-mode or holoscan_flag=True the HXN code does ``.get()``
        for GPU-resident arrays; here we're in single-mode host layout so a
        direct copy works.
        """
        idx = it % self.n_iterations
        mmap_prb_idx_x = (self.nx_prb + self.obj_pad) // 2
        mmap_prb_idx_y = (self.ny_prb + self.obj_pad) // 2

        self.mmap_prb[idx, ...] = self.prb_mode[...]
        # Crop the object to the "inner" region that excludes the obj_pad +
        # probe footprint on each side.
        self.mmap_obj[idx, ...] = self.obj_mode[
            :,
            mmap_prb_idx_x:-mmap_prb_idx_x,
            mmap_prb_idx_y:-mmap_prb_idx_y,
        ]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def snapshot(self):
        """Return a snapshot of the current probe/object for SaveLiveResult.

        Returns a tuple ``(probe, object, iteration, scan_num)``.
        """
        if self.mmap_prb is None or self.mmap_obj is None:
            raise RuntimeError(
                "StreamingPtychoRecon.snapshot called before gpu_setup()"
            )
        it_mmap = self._last_iter % self.n_iterations
        return (
            self.mmap_prb[it_mmap],
            self.mmap_obj[it_mmap],
            self._last_iter,
            self.scan_num,
        )

    def save_final(self, save_dir=None):
        """Persist the final reconstruction to disk.

        Writes ``probe.npy`` and ``object.npy`` into ``save_dir`` and
        returns the directory path so callers can stash additional
        auxiliary files alongside. If ``save_dir`` is None, compute a
        default from ``self.scan_num``.
        """
        import os

        if save_dir is None:
            save_dir = f"/data/users/Holoscan/recon_{self.scan_num or 'last'}"
        os.makedirs(save_dir, exist_ok=True)
        np.save(
            os.path.join(save_dir, "probe.npy"),
            np.asarray(self.prb_mode),
        )
        np.save(
            os.path.join(save_dir, "object.npy"),
            np.asarray(self.obj_mode),
        )
        return save_dir

    # ------------------------------------------------------------------
    # Internal state tracking (used by snapshot)
    # ------------------------------------------------------------------

    _last_iter = 0

    def _track_iter(self, it):
        """Record the most recent iteration index (for snapshot())."""
        self._last_iter = int(it)
