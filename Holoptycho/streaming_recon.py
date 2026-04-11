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
  - ``ptycho.numba_collection``  — numba.cuda kernels for obj/prb accumulation.

None of those modules touches ``ptycho_trans`` state. They are already how
``ptycho_trans`` itself dispatches GPU work, so reusing them preserves
numerical parity with ptycho's batch reconstruction.

The reconstruction engine only implements the subset Holoptycho actually uses:

  - **DM algorithm** (no mADMM, APG, PM, PIE, ML, RAAR, ER).
  - **Single probe mode, single object mode** (``prb_mode_num = obj_mode_num = 1``).
  - **No multislice, no nearfield, no Bragg, no partial coherence**.
  - **No position correction, no data refinement**.
  - **No MPI** — single process, single GPU.

All of these simplifications are valid per Holoptycho's
``eiger_test/ptycho_holo/ptycho_config.txt``.

Reference implementation:

The logic in this module is lifted from
``ptycho.ptycho_trans_ml.ptycho_trans`` on the ``HXN_development`` branch,
which has been running in production at the NSLS-II HXN beamline. Every
method below has a ``HXN ref:`` comment pointing at the source lines it was
derived from — consult those when filling in the TODO sections or when
debugging a numerical discrepancy.

Usage:
    engine = StreamingPtychoRecon(config=param)
    engine.gpu_setup(num_points_max=param.live_num_points_max or 8192)
    engine.reset_for_scan(scan_num="12345", x_range_um=20.0, y_range_um=20.0,
                          num_points_max=4096)
    engine.initial_probe(num_diffs=256)
    if engine.prb_prop_dist_um != 0:
        engine.propagate_probe()
    for it in range(engine.n_iterations):
        engine.iter_once(it)
    prb, obj, it_num, scan_num = engine.snapshot()
    engine.save_final(save_dir="/data/users/Holoscan")
"""

import logging

import numpy as np

# Ptycho is imported as a kernel library. These modules must be pip-importable
# from the ptycho package that holoscan-framework pins as a git dependency. See
# pixi.toml for the exact pin.
#
# NOTE: ptycho expects CuPy to be installed in the same environment. Do not
# import cupy at module-load time here; do it lazily inside methods so that
# this module can at least be imported on CPU-only dev machines (for static
# checks and linting).

logger = logging.getLogger(__name__)


class StreamingPtychoRecon:
    """Streaming DM ptychographic reconstruction.

    Designed for incremental data ingestion: diffraction patterns and scan
    positions arrive over time via the upstream Holoscan operators, and the
    reconstruction loop is driven externally by ``PtychoRecon.compute()``.

    Lifecycle:
        1. ``__init__(config)``                — stash the config; no GPU work.
        2. ``gpu_setup(num_points_max)``       — allocate all GPU buffers and
                                                 cuFFT plans for at most
                                                 ``num_points_max`` scan points.
                                                 Called once per process.
        3. ``reset_for_scan(...)``             — re-dimension object arrays for
                                                 a new scan region without
                                                 reallocating. Called once per
                                                 scan.
        4. During the scan, as more diffraction points arrive:
           - ``clear_region(prev, new)``       — zero the object region where
                                                 new points have landed.
           - ``initial_probe(num_diffs)``      — initialise probe from the
                                                 averaged diffraction (called
                                                 once per scan, when enough
                                                 points have accumulated).
           - ``propagate_probe()``             — optional angular-spectrum
                                                 propagation of the probe.
           - ``iter_once(it)``                 — run one DM iteration on the
                                                 points currently loaded.
           - ``snapshot()``                    — return current probe/object
                                                 snapshot for the live viewer.
        5. ``save_final(save_dir)``            — persist the final recon to
                                                 disk.
    """

    def __init__(self, config):
        """Initialise with a config object.

        ``config`` is whatever ``ptycho.utils.parse_config`` returned — a
        ``SimpleNamespace``-like object with the usual reconstruction
        attributes (``nx_prb``, ``ny_prb``, ``n_iterations``, ``gpus``,
        ``gpu_batch_size``, ``init_prb_flag``, ``alg_flag``, etc.). We copy
        the ones we actually use onto ``self``, and leave the rest alone.
        """
        self.config = config

        # Core config we actually reference. Leave most attributes on the
        # config object; copy here only the ones that are hot-path reads.
        self.nx_prb = int(config.nx)
        self.ny_prb = int(config.ny)
        self.n_iterations = int(config.n_iterations)
        self.gpu_batch_size = int(config.gpu_batch_size)
        self.init_prb_flag = bool(config.init_prb_flag)

        # GPU device. Holoptycho config has gpus=[0]; PtychoRecon reads
        # ``engine.gpu`` unconditionally in ``compute()``.
        self.gpu = int(config.gpus[0]) if config.gpus else 0

        # Probe/object mode counts. Streaming is single-mode only.
        self.prb_mode_num = int(getattr(config, "prb_mode_num", 1))
        self.obj_mode_num = int(getattr(config, "obj_mode_num", 1))
        if self.prb_mode_num != 1 or self.obj_mode_num != 1:
            raise NotImplementedError(
                "StreamingPtychoRecon only supports single-mode "
                f"(prb_mode_num={self.prb_mode_num}, "
                f"obj_mode_num={self.obj_mode_num})"
            )

        # Geometry and object-sizing attributes. These are reset per scan via
        # ``reset_for_scan()``, but we initialise them here so reads don't
        # AttributeError before the first scan flush.
        self.x_pixel_m = 0.0  # set in gpu_setup from detector/wavelength
        self.y_pixel_m = 0.0
        self.obj_pad = int(getattr(config, "obj_pad", 4))
        self.x_range_um = 0.0
        self.y_range_um = 0.0
        self.nx_obj = 0
        self.ny_obj = 0
        self.scan_num = ""

        # Scan-direction flags. Read by upstream PointProcessorOp during
        # config_ops; not used inside the reconstruction loop.
        self.x_direction = float(getattr(config, "x_direction", -1.0))
        self.y_direction = float(getattr(config, "y_direction", -1.0))

        # Probe propagation distance, optional. Config stores it in different
        # fields across branches; prefer explicit ``prb_prop_dist_um`` but
        # fall back to ``distance`` for backwards compat with older configs.
        self.prb_prop_dist_um = float(
            getattr(config, "prb_prop_dist_um", getattr(config, "distance", 0.0))
        )

        # Point-count state.
        self.num_points = 0          # total points in the current scan
        self.num_points_l = 0        # local rank (always == num_points here)
        self.num_points_recon = 0    # points currently loaded into the DM loop

        # GPU buffers. Allocated in gpu_setup(); None until then.
        self.prb_d = None
        self.obj_d = None
        self.diff_d = None
        self.point_info_d = None
        self.product_d = None
        self.prb_obj_d = None
        self.fft_tmp_d = None
        self.amp_tmp_d = None
        self.dev_d = None
        self.power_d = None
        self.obj_upd_d = None
        self.prb_upd_d = None
        self.prb_norm_d = None
        self.obj_norm_d = None

        # Pinned host shadows. Allocated in gpu_setup().
        self.prb_mode = None  # host, page-locked; probe at module-num granularity
        self.obj_mode = None  # host, page-locked; object at module-num granularity
        self.obj_update_l = None
        self.prb_norm_l = None

        # "Zero" views used by keep_obj0 / reset_for_scan. Allocated in gpu_setup.
        self.obj_mode_0 = None
        self.obj_d_0 = None
        self.obj_upd_d_0 = None
        self.prb_norm_d_0 = None
        self.obj_update_l_0 = None
        self.prb_norm_l_0 = None

        # Shared-memory output arrays (probe/object per iteration snapshot).
        # Allocated in gpu_setup() via init_mmap-equivalent logic.
        self.mmap_prb = None
        self.mmap_obj = None

        # cuFFT plans. Allocated in gpu_setup.
        self.cufft_plan = None
        self.cufft_plan_last = None

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def gpu_setup(self, num_points_max):
        """Allocate all GPU buffers and pinned host buffers.

        Called once per process. After this, the engine is ready to handle
        up to ``num_points_max`` scan points without further reallocation.

        HXN ref: ``ptycho_trans.gpu_init`` lines 1945-2110 of
        ``ptycho_trans_ml.py`` on ``HXN_development``. Strip the
        ``multislice_flag``, ``mode_flag`` else-branch, and
        ``holoscan_flag`` paths — we always use mode_flag=True with a
        single mode, never multislice, never holoscan_flag branches.
        """
        # TODO(gpu machine): fill in the following in order, lifting from
        # HXN_development ``gpu_init``:
        #
        # 1. ``import cupy as cp`` and select the device:
        #        cp.cuda.Device(self.gpu).use()
        #        from numba import cuda as numba_cuda
        #        numba_cuda.select_device(self.gpu)
        #
        # 2. Set ``self.complex_precision = cp.complex64`` and
        #    ``self.float_precision = cp.float32`` (matching HXN's
        #    ``set_precisions`` for ``'single'``).
        #
        # 3. Compute ``self.x_pixel_m`` and ``self.y_pixel_m`` from
        #    ``self.config.xray_energy_kev``, ``self.config.z_m``, and
        #    ``self.config.ccd_pixel_um``. HXN ref: ``ptycho_trans.setup``
        #    lines ~4200-4220 of HXN_development.
        #
        # 4. Allocate GPU buffers (sizes below assume single-mode DM):
        #        mp = mo = 1  # prb_mode_num * obj_mode_num
        #        nx, ny = self.nx_prb, self.ny_prb
        #        b = self.gpu_batch_size
        #        self.diff_d = cp.empty((num_points_max, nx, ny), dtype=self.float_precision)
        #        self.point_info_d = cp.empty((num_points_max, 4), dtype=cp.int32)
        #        self.prb_d = cp.empty((mp, nx, ny), dtype=self.complex_precision)
        #        self.product_d = cp.empty((num_points_max, mp, mo, nx, ny), dtype=self.complex_precision)
        #        self.prb_obj_d = cp.empty((b, mp, mo, nx, ny), dtype=self.complex_precision)
        #        self.fft_tmp_d = cp.empty((b * mp * mo, nx, ny), dtype=self.complex_precision)
        #        self.amp_tmp_d = cp.empty((b, nx, ny), dtype=self.float_precision)
        #        self.dev_d = cp.empty_like(self.amp_tmp_d)
        #        self.power_d = cp.empty(b, dtype=self.float_precision)
        #    (see HXN_development gpu_init for the exact shapes of obj_d,
        #    obj_upd_d, prb_upd_d, prb_norm_d, obj_norm_d — they depend on
        #    nx_obj/ny_obj which are set per-scan by reset_for_scan, so
        #    allocate them at the maximum size you expect and reshape later).
        #
        # 5. Pinned host buffers (analogue of ``ptycho_trans.obj_mode`` etc.):
        #        from ptycho.cupy_util import (
        #            page_lock_array, get_page_locked_array,
        #            copy_to_pinned, copy_from_pinned,
        #        )
        #        self.prb_mode = get_page_locked_array(
        #            (mp, nx, ny), dtype=self.complex_precision
        #        )
        #    Do the same for obj_mode (max nx_obj, ny_obj), obj_update_l,
        #    prb_norm_l.
        #
        # 6. cuFFT plans for the forward/backward step:
        #        import cupyx.scipy.fftpack as cufft
        #        self.cufft_plan = cufft.get_fft_plan(
        #            self.fft_tmp_d, axes=(1, 2),
        #        )
        #
        # 7. mmap buffers for live snapshots (HXN ref: init_mmap). These can
        #    just be plain numpy arrays of shape
        #    (n_iterations, mp, nx, ny) for prb and similar for obj.
        #
        # 8. Save the "zero-shape" flattened views (equivalent of keep_obj0):
        #        self.obj_mode_0 = self.obj_mode.reshape(self.obj_mode.size, order='C')
        #        self.obj_d_0 = self.obj_d.reshape(self.obj_d.size, order='C')
        #        ... etc for obj_upd_d, prb_norm_d, obj_update_l, prb_norm_l.
        raise NotImplementedError(
            "StreamingPtychoRecon.gpu_setup is a scaffold. "
            "Fill in on the GPU machine by lifting from HXN_development "
            "ptycho_trans.gpu_init (lines ~1945-2110 of ptycho_trans_ml.py)."
        )

    # ------------------------------------------------------------------
    # Per-scan lifecycle
    # ------------------------------------------------------------------

    def reset_for_scan(self, scan_num, x_range_um, y_range_um, num_points_max):
        """Reset state for a new scan region without reallocating GPU buffers.

        Called on every ``PtychoRecon.flush()``. Combines what HXN calls
        ``new_obj()`` + ``flush_live_recon()`` + ``keep_obj0``/reshape into a
        single entry point.

        HXN ref:
          - ``flush_live_recon`` (lines 4025-4036) for nx_obj/ny_obj math.
          - ``new_obj`` (lines 3892-3924) for the reshape-from-zero-views dance.
          - ``reset_obj`` (lines 3926-3930) for the random initial object.
        """
        self.scan_num = str(scan_num)
        self.x_range_um = float(abs(x_range_um))
        self.y_range_um = float(abs(y_range_um))
        self.num_points = int(num_points_max)
        self.num_points_l = int(num_points_max)
        self.num_points_recon = 0

        # TODO(gpu machine):
        #
        # 1. Compute new obj dimensions (from flush_live_recon, lines 4027-4030):
        #        self.nx_obj = int(self.nx_prb + ceil(self.x_range_um*1e-6/self.x_pixel_m) + self.obj_pad)
        #        self.ny_obj = int(self.ny_prb + ceil(self.y_range_um*1e-6/self.x_pixel_m) + self.obj_pad)
        #        self.nx_obj += self.nx_obj % 2  # force even
        #        self.ny_obj += self.ny_obj % 2
        #
        # 2. Reshape the flattened "_0" views back into the new (mo, nx_obj, ny_obj)
        #    shape (from new_obj, lines 3898-3903):
        #        mo = self.obj_mode_num
        #        n = mo * self.nx_obj * self.ny_obj
        #        self.obj_mode = self.obj_mode_0[:n].reshape((mo, self.nx_obj, self.ny_obj), order='C')
        #        self.obj_d = self.obj_d_0[:n].reshape((mo, self.nx_obj, self.ny_obj), order='C')
        #        ... etc for obj_upd_d, prb_norm_d, obj_update_l, prb_norm_l.
        #
        # 3. Zero the accumulation buffers:
        #        self.obj_upd_d[...] = 0
        #        self.prb_norm_d[...] = 0
        #        self.obj_update_l[...] = 0
        #        self.prb_norm_l[...] = 0
        #
        # 4. Re-initialise the object with the standard random phase seed
        #    (from reset_obj, lines 3926-3930):
        #        self.obj_mode[0, :, :] = 0.99 * np.exp(-0.1j)
        #        # (For single-mode, that's all we need.)
        #        from ptycho.cupy_util import copy_from_pinned
        #        copy_from_pinned(self.obj_mode, self.obj_d, self.obj_mode.nbytes)
        #
        # 5. Resize ``self.point_info`` to the new num_points. Holoptycho
        #    never re-broadcasts it across MPI ranks, so we can just
        #    allocate as ``np.empty((num_points, 4), dtype=np.intc)``. (HXN's
        #    flush_live_recon uses shape (num_points, 2, 2) which is a
        #    harmless typo — reshape is never observed before overwrite.)
        #    Then copy to GPU into ``self.point_info_d[:num_points]``. The
        #    actual coordinates are filled in by the upstream
        #    ``PointProcessorOp`` operator — here we just make sure the
        #    buffer exists and is the right size.
        raise NotImplementedError(
            "StreamingPtychoRecon.reset_for_scan is a scaffold. "
            "Fill in from HXN_development flush_live_recon + new_obj "
            "+ reset_obj on the GPU machine."
        )

    # ------------------------------------------------------------------
    # Probe initialisation
    # ------------------------------------------------------------------

    def initial_probe(self, num_diffs):
        """Initialise the probe from the first ``num_diffs`` diffraction patterns.

        For ``init_prb_flag == True``: compute
        ``fftshift(ifftn(mean(diff_d[:num_diffs])))`` scaled by
        ``sqrt(nx_prb * ny_prb)``.

        HXN ref: ``init_live_prb`` lines 3960-3979. Strip the
        ``self.init_prb_flag`` else-branch — Holoptycho sets
        init_prb_flag=False in its sample config but we only need the
        init-from-diff path (single probe mode).
        """
        # TODO(gpu machine):
        # import cupy as cp
        # from ptycho.cupy_util import copy_to_pinned
        #
        # if self.init_prb_flag:
        #     prb = cp.fft.fftshift(
        #         cp.fft.ifftn(cp.mean(self.diff_d[:num_diffs], axis=0))
        #     ) * cp.sqrt(self.nx_prb * self.ny_prb)
        #     self.prb_d[0] = prb
        #     copy_to_pinned(self.prb_d, self.prb_mode, self.prb_d.nbytes)
        # else:
        #     # Load-from-file path: Holoptycho doesn't use this today.
        #     raise NotImplementedError(
        #         "Loading probe from file not supported in streaming mode"
        #     )
        raise NotImplementedError(
            "StreamingPtychoRecon.initial_probe is a scaffold. "
            "Fill in from HXN_development init_live_prb on the GPU machine."
        )

    def propagate_probe(self):
        """Propagate the current probe by ``self.prb_prop_dist_um`` metres.

        HXN ref: ``propagate_prb`` at lines ~822-833. Uses
        ``ptycho.prop_class_asm.propagate`` which is self-contained and
        accepts an ``xp`` backend argument (numpy or cupy).
        """
        # TODO(gpu machine):
        # from ptycho.prop_class_asm import propagate
        # Propagate on host array (numpy). The HXN path reads self.prb_mode
        # (host) and writes back to self.prb_mode, then copies to prb_d.
        #
        # import numpy as _np
        # self.prb_mode[0] = propagate(
        #     self.prb_mode[0], direction=1,
        #     x_pixel_size_m=self.x_pixel_m,
        #     y_pixel_size_m=self.y_pixel_m,
        #     wavelength_m=self.config.lambda_nm * 1e-9,
        #     z_m=self.prb_prop_dist_um * 1e-6,
        #     xp=_np,
        # )
        # from ptycho.cupy_util import copy_from_pinned
        # copy_from_pinned(self.prb_mode, self.prb_d, self.prb_d.nbytes)
        raise NotImplementedError(
            "StreamingPtychoRecon.propagate_probe is a scaffold. "
            "Fill in from HXN_development propagate_prb on the GPU machine."
        )

    # ------------------------------------------------------------------
    # Object-region management
    # ------------------------------------------------------------------

    def clear_region(self, prev, new):
        """Zero the object region covered by point_info_d[prev:new].

        Called when new scan points have been added and the object region
        over them still holds stale values from earlier iterations.

        HXN ref: ``clear_obj_tail`` lines 3932-3941.
        """
        # TODO(gpu machine):
        # import cupy as cp
        # from ptycho.cupy_util import copy_from_pinned
        #
        # x_low  = int(cp.min(self.point_info_d[prev:new, 0]))
        # x_high = int(cp.max(self.point_info_d[prev:new, 1]))
        # y_low  = int(cp.min(self.point_info_d[prev:new, 2]))
        # y_high = int(cp.max(self.point_info_d[prev:new, 3]))
        #
        # self.obj_mode[0, x_low:x_high, y_low:y_high] = 0.99 * np.exp(-0.1j)
        # copy_from_pinned(self.obj_mode, self.obj_d, self.obj_mode.nbytes)
        raise NotImplementedError(
            "StreamingPtychoRecon.clear_region is a scaffold. "
            "Fill in from HXN_development clear_obj_tail on the GPU machine."
        )

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def iter_once(self, it):
        """Run one DM iteration on the points currently loaded.

        This is the hot path. Dispatches the kernel sequence:

            multiply_with_support (prb * obj -> prb_obj_d)
            multiply_and_sum (prb_obj_d -> fft_tmp_d, psi = 2*prb_obj_d - product_d)
            cuFFT forward    (fft_tmp_d)
            dm_update_amp1   (apply amplitude constraint, compute dev)
            dm_update_amp2   (fft_tmp_d = diff * fft_tmp_d / amp_tmp_d + lambda correction)
            cuFFT backward   (fft_tmp_d)
            multiply_and_sum (fft_tmp_d -> product_d, product_d += 2*fft_tmp_d - prb_obj_d)
            accumulate_obj   (obj_update_l accumulates weighted obj contributions)
            gather_obj       (obj_update_l -> obj_d via division by prb_norm_l)
            accumulate_prb   (prb_update_l similar)
            gather_prb       (prb_update_l -> prb_d)
            snapshot to mmap_prb[it % n_iterations], mmap_obj[it % n_iterations]

        HXN refs:
          - ``one_iter``                    lines 3826-3854  (orchestration)
          - ``update_psi_gpu``               lines 4387-4441  (the dispatch)
          - ``recon_dm_trans_cupy_single``   lines 2716-2782  (the kernel body)
          - ``accumulate_obj_gpu``           lines 2783-2802
          - ``gather_obj_gpu``               lines 2802-2856
          - ``accumulate_prb_gpu``           lines 2857-2872
          - ``gather_prb_gpu``               lines 2872-2909

        Kernels imported from the ptycho library:
          - ``ptycho.cupy_collection.cupy_mod.get_function('multiply_with_support_mode')``
          - ``ptycho.cupy_collection.cupy_mod.get_function('multiply_and_sum')``
          - ``ptycho.cupy_collection.dm_update_amp1_v2``
          - ``ptycho.cupy_collection.dm_update_amp2_v2``
          - ``ptycho.numba_collection.accumulate_prb_mode``
          - ``ptycho.numba_collection.accumulate_obj_mode``

        MPI Allreduce calls in the HXN version's gather functions should be
        dropped — we're single-rank.
        """
        # TODO(gpu machine): this is the meat of the engine. Fill in by
        # carefully transcribing ``recon_dm_trans_cupy_single`` and the
        # accumulate/gather methods, removing the mp_mo > 1 branches (we're
        # single-mode) and the MPI Allreduce calls (we're single-rank).
        raise NotImplementedError(
            "StreamingPtychoRecon.iter_once is a scaffold. "
            "Fill in on the GPU machine by transcribing the DM iteration "
            "body from HXN_development ptycho_trans_ml.py: one_iter -> "
            "update_psi_gpu -> recon_dm_trans_cupy_single, plus "
            "accumulate_obj_gpu + gather_obj_gpu + accumulate_prb_gpu + "
            "gather_prb_gpu."
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def snapshot(self):
        """Return a snapshot of the current probe and object for the live viewer.

        Returns a tuple ``(probe, object, iteration, scan_num)`` suitable
        for the downstream ``SaveLiveResult`` operator in ``ptycho_holo.py``.
        Reads from ``self.mmap_prb[it % n_iterations]`` and
        ``self.mmap_obj[it % n_iterations]``.
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
        returns the directory path (so the caller can stash additional
        auxiliary files alongside). If ``save_dir`` is None, compute a
        default from ``self.scan_num``.

        HXN ref: ``ptycho_trans.save_recon`` does much more (TIFF output,
        orthonormalisation, phase-ramp removal, mode-reshuffling). For the
        streaming use case we start with the simplest thing and add extras
        if Holoptycho actually needs them.
        """
        # TODO(gpu machine):
        # import os
        # if save_dir is None:
        #     save_dir = f"/data/users/Holoscan/recon_{self.scan_num}"
        # os.makedirs(save_dir, exist_ok=True)
        # # prb_mode / obj_mode are the pinned host shadows kept up-to-date
        # # by gather_prb_gpu / gather_obj_gpu after every iteration.
        # np.save(os.path.join(save_dir, "probe.npy"), np.asarray(self.prb_mode))
        # np.save(os.path.join(save_dir, "object.npy"), np.asarray(self.obj_mode))
        # return save_dir
        raise NotImplementedError(
            "StreamingPtychoRecon.save_final is a scaffold. "
            "Fill in with the minimal probe/object .npy writes on the GPU "
            "machine."
        )

    # ------------------------------------------------------------------
    # Internal state tracking (used by snapshot)
    # ------------------------------------------------------------------

    _last_iter = 0

    def _track_iter(self, it):
        """Record the most recent iteration index (for snapshot())."""
        self._last_iter = int(it)
