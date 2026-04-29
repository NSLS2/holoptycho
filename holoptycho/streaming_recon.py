"""Streaming DM ptychographic reconstruction for the Holoscan pipeline.

This module owns the iterative reconstruction state (GPU buffers, cuFFT plans,
probe/object arrays) that was previously delegated to
``ptycho.ptycho_trans_ml.ptycho_trans`` via ``recon_thread``. Keeping it here
mirrors the architecture of ``vit_inference.PtychoViTInferenceOp`` which owns
its GPU-1 TensorRT context without leaning on the ptycho class hierarchy.

The ptycho package is imported purely as a *kernel library*:

  - ``ptycho.cupy_util``         — pinned-memory helpers, block/grid sizing.
  - ``ptycho.prop_class_asm``    — angular-spectrum propagation.
  - ``ptycho.cupy_collection``   — compiled CuPy/RawModule kernels for DM.
  - ``ptycho.numba_collection``  — numba.cuda helpers.

None of those modules touches ``ptycho_trans`` state. They are already how
``ptycho_trans`` itself dispatches GPU work, so reusing them preserves
numerical parity with ptycho's batch reconstruction.

The reconstruction engine only implements the subset Holoptycho actually uses:

  - **DM algorithm** (no mADMM, APG, PM, PIE, ML, RAAR, ER).
  - **Single probe mode, single object mode** (``prb_mode_num = obj_mode_num = 1``).
  - **No multislice, no nearfield, no Bragg, no partial coherence**.
  - **No position correction, no data refinement**.
  - **No MPI** — single process, single GPU.

All logic is adapted from ``ptycho.ptycho_trans_ml.ptycho_trans`` on the
``HXN_development`` branch, which has been running in production at the
NSLS-II HXN beamline.
"""

import logging

import numpy as np

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

        # Scan direction (read by upstream PointProcessorOp)
        self.x_direction = float(getattr(config, "x_direction", -1.0))
        self.y_direction = float(getattr(config, "y_direction", -1.0))

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
        self.obj_d = None
        self.diff_d = None
        self.point_info_d = None
        self.product_d = None
        self.prb_obj_d = None
        self.fft_tmp_d = None
        self.fft_total_d = None
        self.amp_tmp_d = None
        self.dev_d = None
        self.power_d = None
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

        # --- cuFFT plans ---
        self.cufft_plan = None
        self.cufft_plan_last = None

        # --- Kernel handles (set in gpu_setup) ---
        self.kernel_multiply_with_support_mode = None
        self.kernel_multiply_and_sum = None
        self.kernel_restrict_range = None
        self.kernel_dm_update_mode_amp1 = None
        self.kernel_dm_update_mode_amp2 = None
        self.kernel_accumulate_obj_mode = None
        self.kernel_accumulate_prb_mode = None

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
        from ptycho.cupy_collection import (
            cupy_mod,
            dm_update_amp1_v2,
            dm_update_amp2_v2,
            get_mod2,
            dm_update_mode_amp1_v2,
            dm_update_mode_amp2_v2,
        )

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
            self.x_pixel_m = self.lambda_nm * 1e-9 * z_m / (
                self.nx_prb * ccd_pixel_um * 1e-6
            )
            self.y_pixel_m = self.lambda_nm * 1e-9 * z_m / (
                self.ny_prb * ccd_pixel_um * 1e-6
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
        self.point_info_d = cp.empty(
            (self.num_points_l, 4), dtype=cp.int32, order="C"
        )

        # --- Iteration working buffers (single-mode DM) ---
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
        self.dev_d = cp.empty(
            (b * mp * mo, nx, ny), dtype=self.float_precision, order="C"
        )
        self.power_d = cp.empty(
            (b * mp * mo,), dtype=self.float_precision, order="C"
        )
        self.fft_total_d = cp.empty(
            (b, nx, ny), dtype=self.float_precision, order="C"
        )

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
        self.kernel_multiply_and_sum = cupy_mod.get_function(
            f"multiply_and_sum<{dtype_str}>"
        )
        self.kernel_restrict_range = cupy_mod.get_function(
            f"restrict_range<{dtype_str}>"
        )
        self.kernel_dm_update_mode_amp1 = dm_update_mode_amp1_v2(mp * mo)
        self.kernel_dm_update_mode_amp2 = dm_update_mode_amp2_v2(mp * mo)

        func_mod = get_mod2(mp, mo)
        self.kernel_accumulate_prb_mode = func_mod.get_function(
            f"accumulate_prb_mode<{dtype_str}, {mp}, {mo}>"
        )
        self.kernel_accumulate_obj_mode = func_mod.get_function(
            f"accumulate_obj_mode<{dtype_str}, {mp}, {mo}>"
        )

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
        self.num_points_l = int(num_points_max)
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
        """Initialise the probe from the first ``num_diffs`` diffraction patterns.

        HXN ref: ``init_live_prb`` lines 3960-3979. Streaming-mode subset:
        single-mode only, no ``shift_sum`` for secondary modes.
        """
        import cupy as cp

        from ptycho.cupy_util import copy_to_pinned

        if not self.init_prb_flag:
            # Load-from-file path not wired up in streaming mode. Holoptycho's
            # config sets init_prb_flag=False but in practice the engine
            # initialises from diffraction either way. Be permissive.
            logger.warning(
                "initial_probe: init_prb_flag is False but streaming mode "
                "only supports probe-from-diff; computing anyway."
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
        """Run one DM iteration on the points currently loaded.

        Orchestration equivalent of ``one_iter`` (HXN line 3826). Calls:
          - ``_update_psi(it)``                     — ``update_psi_gpu`` line 4400
          - ``_accumulate_obj() + _gather_obj()``   — lines 2783-2855
          - ``_accumulate_prb() + _gather_prb()``   — lines 2857-2910
          - ``_update_mmap(it)``                    — snapshot to numpy

        Skips GUI/preview/error-calculation paths that Holoptycho doesn't use.
        """
        self.current_it = it

        # exit-wave update
        self._update_psi(it)

        # probe + object updates
        self._accumulate_obj()
        self._gather_obj()
        self._accumulate_prb()
        self._gather_prb()

        # snapshot into mmap arrays (for SaveLiveResult)
        self._update_mmap(it)

    def _update_psi(self, it):
        """Call ``recon_dm_trans_cupy_single`` over all points in batches.

        HXN refs:
          - ``update_psi_gpu`` lines 4400-4435 (DM path only).
          - ``recon_gpu_launch`` lines 1858-1865.
          - ``init_product_gpu`` lines 2105-2112.
          - ``recon_dm_trans_cupy_single`` lines 2716-2779.
        """
        import cupy as cp

        from ptycho.cupy_util import get_3d_block_grid_config

        mp = self.prb_mode_num
        mo = self.obj_mode_num
        nx = self.nx_prb
        ny = self.ny_prb

        # At iteration 0, build product_d = prb * obj(support). HXN ref:
        # init_product_gpu, lines 2105-2112.
        if it == 0:
            block, grid = get_3d_block_grid_config(
                (self.num_points_l, nx, ny)
            )
            args = (
                self.prb_d,
                self.obj_d,
                self.product_d,
                self.point_info_d,
                nx,
                ny,
                mp,
                self.nx_obj,
                self.ny_obj,
                mo,
                0,
                self.num_points_l,
            )
            self.kernel_multiply_with_support_mode(grid, block, args)

        # Batched DM update. HXN ref: recon_gpu_launch.
        self.last = self.num_points_recon % self.gpu_batch_size
        n_batch = self.num_points_recon // self.gpu_batch_size
        for i in range(n_batch):
            self._recon_dm_batch(i * self.gpu_batch_size, self.gpu_batch_size)
        if self.last > 0:
            self._recon_dm_batch(n_batch * self.gpu_batch_size, self.last)

    def _recon_dm_batch(self, start_point, batch_size):
        """One DM batch step: the kernel sequence from ``recon_dm_trans_cupy_single``.

        HXN ref: lines 2716-2779 of ``ptycho_trans_ml.py``.
        """
        import cupy as cp
        import cupyx.scipy.fftpack as cufft

        from ptycho.cupy_util import get_3d_block_grid_config

        mp = self.prb_mode_num
        mo = self.obj_mode_num
        nx = self.nx_prb
        ny = self.ny_prb
        mode_batch_size = batch_size * mp * mo

        block, grid = get_3d_block_grid_config((batch_size, nx, ny))
        threadsperblock, blockspergrid = get_3d_block_grid_config(
            (mode_batch_size, nx, ny)
        )

        # Aliases
        prb_obj_d = self.prb_obj_d[0:mode_batch_size]
        fft_tmp_d = self.fft_tmp_d[0:mode_batch_size]
        amp_tmp_d = self.amp_tmp_d[0:mode_batch_size]
        dev_d = self.dev_d[0:mode_batch_size]
        power_d = self.power_d[0:mode_batch_size]
        diff = self.diff_d[start_point : start_point + batch_size]
        tmp_fft_mode_total = self.fft_total_d[0:batch_size]
        plan = (
            self.cufft_plan
            if batch_size == self.gpu_batch_size
            else self.cufft_plan_last
        )
        product_d = self.product_d.reshape(
            self.num_points_l * mp * mo, nx, ny
        )[start_point * mp * mo : start_point * mp * mo + mode_batch_size]

        # 1. prb_obj_d = prb * obj_support_slice
        args = (
            self.prb_d,
            self.obj_d,
            prb_obj_d,
            self.point_info_d,
            cp.uint32(nx),
            cp.uint32(ny),
            mp,
            cp.uint32(self.nx_obj),
            cp.uint32(self.ny_obj),
            mo,
            cp.uint32(start_point),
            cp.uint32(batch_size),
        )
        self.kernel_multiply_with_support_mode(grid, block, args)

        # 2. fft_tmp_d = 2*prb_obj_d - product_d
        args = (
            prb_obj_d,
            product_d,
            fft_tmp_d,
            self.float_precision(2.0),
            self.float_precision(-1.0),
            False,
            nx,
            ny,
            mode_batch_size,
        )
        self.kernel_multiply_and_sum(blockspergrid, threadsperblock, args)

        # 3. Forward FFT
        fft_tmp_d[...] = cufft.fftn(
            fft_tmp_d, axes=(1, 2), overwrite_x=True, plan=plan
        )

        # 4. DM amplitude constraint step 1
        self.kernel_dm_update_mode_amp1(
            fft_tmp_d,
            diff,
            False,
            nx,
            ny,
            amp_tmp_d,
            dev_d,
            fft_tmp_d,
            tmp_fft_mode_total,
        )
        power_d[...] = cp.sum(dev_d ** 2, axis=(1, 2))

        # 5. DM amplitude constraint step 2
        self.kernel_dm_update_mode_amp2(
            amp_tmp_d,
            dev_d,
            fft_tmp_d,
            tmp_fft_mode_total,
            diff,
            power_d,
            self.float_precision(self.sigma2),
            False,
            nx,
            ny,
            fft_tmp_d,
        )

        # 6. Inverse FFT
        fft_tmp_d[...] = cufft.ifftn(
            fft_tmp_d, axes=(1, 2), overwrite_x=True, plan=plan
        )

        # 7. product_d += beta * sqrt(nx*ny) * fft_tmp_d - beta * prb_obj_d
        #    (in-place, accumulate=True)
        args = (
            fft_tmp_d,
            prb_obj_d,
            product_d,
            self.float_precision(self.beta * np.sqrt(nx * ny)),
            self.float_precision(-self.beta),
            True,
            nx,
            ny,
            mode_batch_size,
        )
        self.kernel_multiply_and_sum(blockspergrid, threadsperblock, args)

        # product_d needs to be back in (num_points_l, mp, mo, nx, ny) shape
        # for the next iteration — the reshape view above doesn't require a
        # copy, but HXN's code explicitly re-reshapes afterwards.
        self.product_d = self.product_d.reshape(
            self.num_points_l, mp, mo, nx, ny
        )

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
        psi = self.product_d.reshape(
            self.num_points_l * mp * mo, self.nx_prb, self.ny_prb
        )[start_point * mp * mo : (start_point + batch_size) * mp * mo]
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
        """Single-rank flavour of ``gather_obj_gpu`` (lines 2802-2855).

        Skips the MPI Allreduce and NCCL branches; we're single-process.
        Also skips the ``mask_obj_flag`` kernel by default — re-enable
        below if your config sets it.
        """
        import cupy as cp

        from ptycho.cupy_util import copy_to_pinned, copy_from_pinned

        copy_to_pinned(self.obj_upd_d, self.obj_update_l, self.obj_upd_d.nbytes)
        copy_to_pinned(self.prb_norm_d, self.prb_norm_l, self.prb_norm_d.nbytes)

        # obj_mode[...] = obj_update_l / prb_norm_l
        self.obj_mode[...] = self.obj_update_l / self.prb_norm_l

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
        psi = self.product_d.reshape(
            self.num_points_l * mp * mo, self.nx_prb, self.ny_prb
        )[start_point * mp * mo : (start_point + batch_size) * mp * mo]
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
        """Single-rank flavour of ``gather_prb_gpu`` (lines 2872-2910)."""
        from ptycho.cupy_util import copy_to_pinned, copy_from_pinned

        copy_to_pinned(self.obj_norm_d, self.obj_norm_l, self.obj_norm_d.nbytes)
        copy_to_pinned(self.prb_upd_d, self.prb_update_l, self.prb_upd_d.nbytes)

        self.prb_mode[...] = self.prb_update_l / self.obj_norm_l

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
