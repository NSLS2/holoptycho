"""
PtychoViT TensorRT inference operator for Holoscan pipeline.

Runs PtychoViT neural network inference in parallel with the iterative
PtychoRecon solver. Takes preprocessed diffraction amplitudes from
ImagePreprocessorOp, runs TRT inference via the ``ptychoml`` package,
and saves predicted amplitude/phase patches to disk.

No PyTorch imports — uses TensorRT + PyCUDA via ptychoml (safe for
NSLS-II container).

Usage:
    See ptycho_holo.py for wiring into PtychoApp.
"""

import logging
import os
import time

import numpy as np

from holoscan.core import Operator, OperatorSpec, ConditionType, IOSpec
from .mosaic_stitch import stitch_batch_into
from .tiled_writer import get_writer

# Module-level writer shared with ptycho_holo.py operators.
_writer = get_writer()


def read_engine_batch_size(engine_path: str) -> int:
    """Return the batch dim of the input tensor for a TensorRT .engine file.

    Used by the pipeline composer to size frame batches to match the model,
    so the streaming pipeline never feeds the engine more frames than it was
    compiled for.
    """
    import tensorrt as trt

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TRT engine: {engine_path}")
    input_name = next(
        engine.get_tensor_name(i)
        for i in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
    )
    return int(engine.get_tensor_shape(input_name)[0])


class PtychoViTInferenceOp(Operator):
    """Holoscan operator that runs PtychoViT TRT inference on diffraction batches.

    Delegates TRT engine loading, buffer allocation, and inference (including
    fftshift, spatial padding, and final-batch padding) to
    ``ptychoml.PtychoViTInference``. This operator is a thin Holoscan adapter
    around that session.

    Inputs:
        diff_amp:      [B, H, W] float32 — preprocessed diffraction amplitude
        image_indices: [B] int32 — frame indices (for correlating with scan positions)

    Outputs:
        vit_result: tuple(pred, indices) where pred is [B, 2, H, W] or [B, H, W]

    Parameters:
        engine_path:       Path to .engine file (must match batch size B)
        gpu:               CUDA device ordinal (default 1; leave 0 for PtychoRecon)
        output_save_dir:   Directory for saving predictions (default /data/users/Holoscan)
        data_is_shifted:   If True, input diff_amp has been fftshift'd and
                           should be undone before inference.
    """

    def __init__(
        self,
        fragment,
        *args,
        engine_path: str,
        gpu: int = 1,
        output_save_dir: str = "/data/users/Holoscan",
        data_is_shifted: bool = False,
        **kwargs,
    ):
        super().__init__(fragment, *args, **kwargs)
        self._logger = logging.getLogger("PtychoViTInferenceOp")
        self.engine_path = engine_path
        self.gpu = gpu
        self.output_save_dir = output_save_dir
        self._data_is_shifted = data_is_shifted

        # Lazy-initialized on first compute()
        self._session = None

        # Stats
        self.n_batches = 0
        self.total_infer_time = 0.0

    def _init_session(self):
        """Create and eagerly initialize the ptychoml inference session.

        ``PtychoViTInference.__init__`` only stores config; the TRT engine and
        CUDA context are loaded lazily on the first ``predict()`` call. We
        force that load here by calling the (private) ``_init_engine`` so the
        first batch isn't slowed down by the ~1–2 s engine deserialize.
        """
        from ptychoml import PtychoViTInference

        if self.gpu == 0:
            self._logger.warning(
                "VIT running on GPU 0 — same as PtychoRecon (CuPy). "
                "PyCUDA + CuPy on the same GPU from different threads can cause "
                "CUDA context crashes. Use gpu=1 on multi-GPU systems."
            )
        self._session = PtychoViTInference(
            engine_path=self.engine_path,
            gpu=self.gpu,
            data_is_shifted=self._data_is_shifted,
        )
        self._session._init_engine()
        self.engine_batch_size = int(self._session.expected_input_shape[0])
        self._logger.info(
            "ptychoml.PtychoViTInference ready: engine=%s gpu=%d engine_batch=%d",
            self.engine_path,
            self.gpu,
            self.engine_batch_size,
        )

    def setup(self, spec: OperatorSpec):
        spec.input("diff_amp").connector(
            IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32
        )
        spec.input("image_indices").connector(
            IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32
        )
        spec.output("vit_result").condition(ConditionType.NONE)

    def start(self):
        # Load the TRT engine before the scheduler starts dispatching data so
        # the first ViT batch doesn't pay the ~1–2 s engine load latency.
        if self._session is None:
            self._init_session()

    def compute(self, op_input, op_output, context):
        try:
            self._compute_inner(op_input, op_output, context)
        except Exception:
            self._logger.exception("VIT inference failed (pipeline continues)")

    def _compute_inner(self, op_input, op_output, context):
        if self._session is None:
            self._init_session()

        diff_amp = op_input.receive("diff_amp")
        indices = op_input.receive("image_indices")

        if diff_amp is None:
            return

        # --- Hot-swap engine reload via sentinel file ---
        reload_file = os.path.join(
            os.path.dirname(self.engine_path), "reload_engine.txt"
        )
        if os.path.exists(reload_file):
            try:
                with open(reload_file) as f:
                    new_path = f.read().strip()
                if (
                    new_path
                    and new_path != self.engine_path
                    and os.path.exists(new_path)
                ):
                    self._logger.info(
                        "Reloading engine: %s -> %s", self.engine_path, new_path
                    )
                    self._session.cleanup()
                    self.engine_path = new_path
                    self._session = None
                    os.remove(reload_file)
                    self._init_session()
                    self._logger.info("Engine reload complete: %s", new_path)
                else:
                    os.remove(reload_file)
            except Exception as e:
                self._logger.warning("Engine reload failed: %s", e)

        # --- Run inference via ptychoml (handles fftshift, padding, TRT) ---
        # If the streaming pipeline batch is larger than the engine's compiled
        # batch dim, split into engine-sized sub-batches and concatenate results.
        # ptychoml.PtychoViTInference handles a final partial sub-batch via
        # internal padding.
        ebs = self.engine_batch_size
        n = diff_amp.shape[0]
        t0 = time.perf_counter()
        if n <= ebs:
            pred, _ = self._session.predict(diff_amp)
        else:
            preds = []
            for start in range(0, n, ebs):
                sub_pred, _ = self._session.predict(diff_amp[start:start + ebs])
                preds.append(sub_pred)
            pred = np.concatenate(preds, axis=0)
        dt = time.perf_counter() - t0

        # --- Stats ---
        self.n_batches += 1
        self.total_infer_time += dt
        if self.n_batches % 10 == 0:
            avg_ms = (self.total_infer_time / self.n_batches) * 1000
            self._logger.info(
                "VIT batch %d: %.1f ms (avg %.1f ms), pred shape %s",
                self.n_batches,
                dt * 1000,
                avg_ms,
                pred.shape,
            )

        # --- Emit ---
        op_output.emit((pred.copy(), indices.copy()), "vit_result")

    def __del__(self):
        if self._session is not None:
            try:
                self._session.cleanup()
            except Exception:
                pass


class SaveViTResult(Operator):
    """Save VIT predictions and the running phase mosaic to tiled.

    Per batch:

    * Publishes the raw ``(pred, indices)`` to ``<run>/vit/batches/NNNNNN``
      (and the convenience ``vit/pred_latest`` mirror) so an offline analyst
      can re-stitch with any algorithm.
    * Accumulates the phase channel into a Fourier-shift stitched mosaic at
      ``<run>/vit/mosaic`` (counts-normalised average), pre-allocated from
      the commanded scan extent. The dashboard reads ``vit/mosaic`` directly
      via the same path used for the iterative live object — no client-side
      stitching.

    A new scan is detected by the smallest frame index in the new batch
    being less than the largest seen so far; on that signal both the batch
    counter and the mosaic state are reset so the next run starts fresh.
    """

    def __init__(
        self,
        fragment,
        *args,
        positions_provider=None,
        pixel_size_m: float | None = None,
        x_range_um: float | None = None,
        y_range_um: float | None = None,
        inner_crop: int | None = None,
        canvas_pad: int = 64,
        fourier_pad: int = 32,
        phase_channel_index: int = 1,
        overshoot_factor: float = 1.2,
        enable_batch_writes: bool = False,
        transpose_patches: bool = True,
        **kwargs,
    ):
        # Holoscan's Operator.__init__ calls setup(spec), so any attribute
        # setup() reads must be assigned BEFORE super().__init__() runs.
        self._enable_batch_writes = enable_batch_writes

        super().__init__(fragment, *args, **kwargs)
        self.batch_num = 0
        self.max_index_seen = -1
        # Transpose each predicted patch before stitching. Required for HXN
        # data: the object-domain array axes are swapped relative to the
        # probe/dp frame, so patches stitched without this transpose come
        # out rotated 90° relative to the scan grid (matches Fix #9 in the
        # offline run_inference.py).
        self._transpose_patches = transpose_patches
        # Optional callable returning the latest (n, 2) per-frame positions
        # array (microns) — typically lambda: point_proc.positions_um. When
        # supplied, the snapshot is published alongside each ViT batch so
        # downstream consumers can stitch using real positions.
        self._positions_provider = positions_provider

        self._pixel_size_m = pixel_size_m
        self._x_range_um = x_range_um
        self._y_range_um = y_range_um
        self._inner_crop = inner_crop
        self._canvas_pad = canvas_pad
        self._fourier_pad = fourier_pad
        self._phase_channel_index = phase_channel_index
        self._overshoot_factor = overshoot_factor

        # Lazy state — built on first batch once we know the model's patch
        # size (``pred.shape[-1]``). Reset on each new scan.
        self._mosaic: np.ndarray | None = None
        self._counts: np.ndarray | None = None
        # Parallel amplitude mosaic — stitched in lockstep with phase so the
        # dashboard can render either channel.
        self._mosaic_amp: np.ndarray | None = None
        self._counts_amp: np.ndarray | None = None
        self._canvas_origin_um: tuple[float, float] | None = None
        # Cropped half-patch dims (set when canvas is allocated). Used by
        # the grow path to compute the buffer that must surround the
        # bounding box of all positions.
        self._half_h: int = 0
        self._half_w: int = 0
        # Whether stitching is even possible (requires scan-grid params and a
        # positions provider). If not, we still write the per-batch arrays
        # so an offline analyst has the raw data.
        self._stitch_enabled = (
            positions_provider is not None
            and pixel_size_m is not None
            and pixel_size_m > 0
            and x_range_um is not None
            and y_range_um is not None
        )

        self._logger = logging.getLogger("holoptycho.SaveViTResult")
        if not self._stitch_enabled:
            self._logger.info(
                "ViT mosaic stitching disabled (missing pixel/range/positions); "
                "per-batch publishing still active"
            )

    def _reset_mosaic(self) -> None:
        self._mosaic = None
        self._counts = None
        self._mosaic_amp = None
        self._counts_amp = None
        self._canvas_origin_um = None
        self._half_h = 0
        self._half_w = 0

    def _ensure_canvas(self, patch_h: int, patch_w: int, positions_um: np.ndarray) -> bool:
        """Allocate canvas + counts on first batch. Returns True if ready."""
        if self._mosaic is not None:
            return True
        finite = np.isfinite(positions_um).all(axis=1)
        if not finite.any():
            # No finite positions yet — defer until PandA catches up.
            return False
        ps = self._pixel_size_m
        # Auto-derive inner_crop from the actual model output size when the
        # caller didn't pin it. Trims ~25% of edge from each side, leaving the
        # central half — historical default for the 256-patch model was 64,
        # which matches patch_size // 4 and scales correctly to smaller patches.
        # The artifact band is from the Fourier-shift placement and is
        # proportional to patch size, so a fixed fraction is the right metric.
        if self._inner_crop is None:
            self._inner_crop = min(patch_h, patch_w) // 4
            self._logger.info(
                "Auto-derived inner_crop=%d for %dx%d ViT patches",
                self._inner_crop, patch_h, patch_w,
            )
        cropped_h = patch_h - 2 * self._inner_crop
        cropped_w = patch_w - 2 * self._inner_crop
        if cropped_h <= 0 or cropped_w <= 0:
            self._logger.error(
                "inner_crop=%d too large for patch %dx%d; disabling stitching",
                self._inner_crop, patch_h, patch_w,
            )
            self._stitch_enabled = False
            return False

        # Pre-allocate the canvas large enough to hold the full scan including
        # encoder overshoot on settling rows. Tiled does not allow node
        # deletion via the writer client, so the canvas shape must be fixed
        # for the lifetime of the run — we cannot grow.
        #
        # Strategy: take max(observed, commanded * safety_factor) on each
        # axis and center on the observed midpoint. The safety factor
        # (3×) covers the HXN settling-row overshoot seen in practice
        # (commanded 2 µm → observed 6 µm).
        half_h = cropped_h // 2
        half_w = cropped_w // 2
        x_min_um = float(np.nanmin(positions_um[finite, 0]))
        x_max_um = float(np.nanmax(positions_um[finite, 0]))
        y_min_um = float(np.nanmin(positions_um[finite, 1]))
        y_max_um = float(np.nanmax(positions_um[finite, 1]))
        x_obs_um = x_max_um - x_min_um
        y_obs_um = y_max_um - y_min_um
        x_range_um = max(x_obs_um, self._x_range_um * self._overshoot_factor)
        y_range_um = max(y_obs_um, self._y_range_um * self._overshoot_factor)
        x_mid_um = 0.5 * (x_min_um + x_max_um)
        y_mid_um = 0.5 * (y_min_um + y_max_um)
        canvas_h = (
            int(np.ceil(y_range_um * 1e-6 / ps))
            + 2 * half_h + 2 + 2 * self._canvas_pad
        )
        canvas_w = (
            int(np.ceil(x_range_um * 1e-6 / ps))
            + 2 * half_w + 2 + 2 * self._canvas_pad
        )
        origin_x_um = x_mid_um - (canvas_w / 2.0) * ps * 1e6
        origin_y_um = y_mid_um - (canvas_h / 2.0) * ps * 1e6

        self._mosaic = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        self._counts = np.zeros_like(self._mosaic)
        self._mosaic_amp = np.zeros_like(self._mosaic)
        self._counts_amp = np.zeros_like(self._mosaic)
        self._canvas_origin_um = (origin_y_um, origin_x_um)
        self._half_h = half_h
        self._half_w = half_w
        self._logger.info(
            "ViT mosaic canvas allocated: %dx%d px (%.2f x %.2f um), origin=(%.3f, %.3f) um, "
            "cropped patch=%dx%d",
            canvas_h, canvas_w,
            canvas_h * ps * 1e6, canvas_w * ps * 1e6,
            origin_y_um, origin_x_um,
            cropped_h, cropped_w,
        )
        return True

    def _stitch_batch(self, pred: np.ndarray, indices: np.ndarray):
        """Stitch the batch into the in-memory mosaic and return a snapshot.

        Returns ``(mosaic_copy, counts_copy, batch_num, pixel_size_m,
        canvas_origin_um)`` for the downstream ``MosaicWriterOp`` to
        normalize and write to tiled, or ``None`` if no work was done.
        Copying the mosaic is ~1 ms for the typical 666x668 float32, well
        under the per-batch budget.
        """
        if not self._stitch_enabled:
            return
        positions_um = self._positions_provider()
        if positions_um is None:
            return
        if not self._ensure_canvas(pred.shape[-1], pred.shape[-1], positions_um):
            return

        def _extract_channel(ch_idx: int) -> np.ndarray:
            patches = pred[:, ch_idx].astype(np.float32, copy=False)
            if self._transpose_patches:
                patches = patches.transpose(0, 2, 1)
            if self._inner_crop > 0:
                c = self._inner_crop
                patches = patches[:, c:-c, c:-c]
            return patches

        phase = _extract_channel(self._phase_channel_index)
        amp = _extract_channel(1 - self._phase_channel_index)

        # Map per-frame um → canvas px. positions_um columns: 0=x, 1=y.
        sub = positions_um[indices]
        finite = np.isfinite(sub).all(axis=1)
        if not finite.any():
            return

        # Drop frames whose center would land outside the placement window
        # (canvas can't grow — tiled doesn't allow node deletion via the
        # writer client). The canvas is sized from commanded extent ×
        # overshoot_factor so this should be rare in practice.
        ps = self._pixel_size_m
        oy_um, ox_um = self._canvas_origin_um
        canvas_h, canvas_w = self._mosaic.shape
        margin_y = self._half_h + 1
        margin_x = self._half_w + 1
        x_um = sub[finite, 0]
        y_um = sub[finite, 1]
        py = (y_um - oy_um) * 1e-6 / ps
        px = (x_um - ox_um) * 1e-6 / ps
        in_bounds = (
            (py >= margin_y) & (py < canvas_h - margin_y)
            & (px >= margin_x) & (px < canvas_w - margin_x)
        )
        if not in_bounds.any():
            self._logger.warning(
                "ViT mosaic: all %d frames in batch fall outside canvas — "
                "increase overshoot_factor", int(finite.sum()),
            )
            return
        if not in_bounds.all():
            self._logger.warning(
                "ViT mosaic: %d/%d frames in batch fall outside canvas — clipping",
                int((~in_bounds).sum()), int(finite.sum()),
            )
        positions_px = np.empty((int(in_bounds.sum()), 2), dtype=np.float64)
        positions_px[:, 0] = py[in_bounds]
        positions_px[:, 1] = px[in_bounds]
        finite_idx = np.where(finite)[0][in_bounds]
        phase_batch = phase[finite_idx]
        amp_batch = amp[finite_idx]

        try:
            self._mosaic, self._counts = stitch_batch_into(
                self._mosaic,
                self._counts,
                phase_batch,
                positions_px,
                pad=self._fourier_pad,
            )
        except Exception:
            self._logger.exception("stitch_batch_into (phase) failed (skipping batch)")
            return

        try:
            self._mosaic_amp, self._counts_amp = stitch_batch_into(
                self._mosaic_amp,
                self._counts_amp,
                amp_batch,
                positions_px,
                pad=self._fourier_pad,
            )
        except Exception:
            self._logger.exception("stitch_batch_into (amp) failed")
            self._mosaic_amp = None
            self._counts_amp = None

        # Bounding box of the patches just placed, in **uncropped** canvas
        # pixel coords. MosaicWriterOp uses this to patch only the affected
        # subregion to Tiled, instead of pushing the full 36 MB canvas every
        # batch. Margin = half the cropped patch size + Fourier pad slop.
        canvas_h, canvas_w = self._mosaic.shape
        ph, pw = phase_batch.shape[-2:]
        margin_y = ph // 2 + self._fourier_pad + 1
        margin_x = pw // 2 + self._fourier_pad + 1
        py_min = int(np.floor(positions_px[:, 0].min())) - margin_y
        py_max = int(np.ceil(positions_px[:, 0].max())) + margin_y
        px_min = int(np.floor(positions_px[:, 1].min())) - margin_x
        px_max = int(np.ceil(positions_px[:, 1].max())) + margin_x

        # Crop the canvas_pad border before handing off — the padding pixels
        # are needed internally for FFT wrap-around safety but should not be
        # shipped to Tiled (they'd show as a uniform median-fill border
        # around the scan). Translate origin_um and bbox into cropped coords
        # so pixel↔position mapping stays correct on the dashboard side.
        cp = self._canvas_pad
        if cp > 0:
            cropped_h = canvas_h - 2 * cp
            cropped_w = canvas_w - 2 * cp
            mosaic_snap = self._mosaic[cp:cp + cropped_h, cp:cp + cropped_w].copy()
            counts_snap = self._counts[cp:cp + cropped_h, cp:cp + cropped_w].copy()
            mosaic_amp_snap = (
                self._mosaic_amp[cp:cp + cropped_h, cp:cp + cropped_w].copy()
                if self._mosaic_amp is not None else None
            )
            counts_amp_snap = (
                self._counts_amp[cp:cp + cropped_h, cp:cp + cropped_w].copy()
                if self._counts_amp is not None else None
            )
            oy_um, ox_um = self._canvas_origin_um
            origin_snap = (oy_um + cp * ps * 1e6, ox_um + cp * ps * 1e6)
            bbox = (
                max(0, py_min - cp),
                min(cropped_h, py_max - cp),
                max(0, px_min - cp),
                min(cropped_w, px_max - cp),
            )
        else:
            mosaic_snap = self._mosaic.copy()
            counts_snap = self._counts.copy()
            mosaic_amp_snap = self._mosaic_amp.copy() if self._mosaic_amp is not None else None
            counts_amp_snap = self._counts_amp.copy() if self._counts_amp is not None else None
            origin_snap = self._canvas_origin_um
            bbox = (
                max(0, py_min),
                min(canvas_h, py_max),
                max(0, px_min),
                min(canvas_w, px_max),
            )

        # Hand off to MosaicWriterOp via a copy. The downstream operator runs
        # on its own scheduler thread and does the (heavier) normalize +
        # tiled write so this compute thread can return to the next ViT
        # batch immediately. Copying ~7 MB total takes well under 2 ms.
        return (
            mosaic_snap,
            counts_snap,
            mosaic_amp_snap,
            counts_amp_snap,
            self.batch_num,
            self._pixel_size_m,
            origin_snap,
            bbox,
        )

    def setup(self, spec: OperatorSpec):
        spec.input("results").connector(
            IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32
        )
        spec.output("mosaic_snapshot").condition(ConditionType.NONE)
        spec.output("positions_snapshot").condition(ConditionType.NONE)
        # Per-batch pred + indices export is opt-in (config field
        # vit_batch_writes). Disabled by default because tiled HTTPS PUT
        # throughput (~1 MB/s for the 33 MB pred) gates the whole ViT
        # branch when enabled.
        if self._enable_batch_writes:
            spec.output("vit_batch").condition(ConditionType.NONE)

    def start(self):
        # Numba JIT-compiles ``stitch_batch_into`` (and its FFT helpers) on
        # the first call — ~1–2 s on the very first batch otherwise. Burn
        # that cost here with a 2-frame dummy so the first real batch hits a
        # warm cache.
        try:
            dummy_canvas = np.zeros((16, 16), dtype=np.float32)
            dummy_counts = np.zeros_like(dummy_canvas)
            dummy_patches = np.zeros((1, 4, 4), dtype=np.float32)
            dummy_positions = np.array([[8.0, 8.0]], dtype=np.float64)
            stitch_batch_into(
                dummy_canvas, dummy_counts, dummy_patches, dummy_positions, pad=2,
            )
            self._logger.info("SaveViTResult: numba stitch kernel pre-compiled")
        except Exception:
            self._logger.exception("Numba pre-warm failed (non-fatal)")

    def compute(self, op_input, op_output, context):
        try:
            results = op_input.receive("results")
            if results is None:
                return
            pred, indices = results

            # Detect new scan: if smallest index in this batch is less than
            # what we've seen, a new scan has started
            min_idx = int(indices.min())
            if min_idx < self.max_index_seen and self.batch_num > 0:
                self.batch_num = 0
                self.max_index_seen = -1
                self._reset_mosaic()

            self.max_index_seen = max(self.max_index_seen, int(indices.max()))
            t0 = time.perf_counter()

            # Compute progress info (chunk N / M, frames per chunk) for the
            # per-op timing logs. ``positions_um`` is allocated at the
            # configured grid size (x_num * y_num) but only positions for
            # frames that actually arrive get populated; the rest stay NaN.
            # Counting finite rows gives the true n_frames once PandA has
            # delivered (handles --skip-frames cleanly: the skipped frames'
            # positions are never published, so they stay NaN and don't
            # inflate the denominator).
            chunk_size = int(pred.shape[0])
            total_batches = 0
            positions = None
            if self._positions_provider is not None:
                positions = self._positions_provider()
                if positions is not None and chunk_size > 0:
                    n_finite = int(np.isfinite(positions[:, 0]).sum())
                    total_batches = (n_finite + chunk_size - 1) // chunk_size

            # Hand off the positions snapshot to PositionsWriterOp on its own
            # scheduler thread (capacity=1 + QueuePolicy.POP). Each tiled
            # HTTPS PUT of positions_um is ~700 ms; doing it inline blocked
            # SaveViTResult for ~85% of every chunk and prevented the pipeline
            # from keeping up with the 200 Hz publish rate.
            t_pos_start = time.perf_counter()
            if positions is not None:
                op_output.emit(positions.copy(), "positions_snapshot")
            t_pos_end = time.perf_counter()

            # Per-batch pred + indices export is opt-in (config field
            # vit_batch_writes). When enabled, hand off to BatchWriterOp via a
            # bounded FIFO; when disabled, skip entirely so the ViT branch
            # isn't gated by tiled's slow HTTPS PUT (~28 s per 33 MB pred).
            if self._enable_batch_writes:
                op_output.emit(
                    (self.batch_num, total_batches, chunk_size, pred, indices),
                    "vit_batch",
                )

            # Stitch in-memory and hand off to MosaicWriterOp. The downstream
            # input is capacity=1 + QueuePolicy.POP, so if the writer is mid-
            # tiled-PUT this snapshot replaces any stale one queued behind
            # it. We never block this thread on the tiled write.
            t_stitch_start = time.perf_counter()
            snap = self._stitch_batch(pred, indices)
            t_stitch_end = time.perf_counter()
            if snap is not None:
                snap = (*snap, total_batches, chunk_size)
                op_output.emit(snap, "mosaic_snapshot")
            t_emit_end = time.perf_counter()

            self._logger.info(
                "SaveViTResult: chunk %d/%d (%d frames) "
                "write_pos=%.1f ms stitch=%.1f ms emit=%.1f ms total=%.1f ms",
                self.batch_num + 1, total_batches, chunk_size,
                (t_pos_end - t_pos_start) * 1000,
                (t_stitch_end - t_stitch_start) * 1000,
                (t_emit_end - t_stitch_end) * 1000,
                (t_emit_end - t0) * 1000,
            )
            self.batch_num += 1
        except Exception:
            self._logger.exception("SaveViTResult.compute failed")


class MosaicWriterOp(Operator):
    """Writes the latest ViT mosaic snapshot to tiled on its own scheduler
    thread.

    Each tiled HTTPS PUT of the mosaic node takes ~30 s end-to-end. Running
    that on the upstream ``SaveViTResult`` thread would gate the whole ViT
    branch on the tiled write. Instead, ``SaveViTResult`` emits a copy of
    the in-memory ``(mosaic, counts)`` arrays per batch; this op's input
    connector is ``capacity=1`` with ``QueuePolicy.POP``, so newer
    snapshots evict older ones whenever the writer is busy. The pipeline
    keeps stitching at full ViT cadence; tiled sees whatever slice of
    progress it can keep up with.

    This op is the only writer for ``<run>/vit/mosaic``. The pred + indices
    per-batch outputs stay synchronous on ``SaveViTResult`` because
    intermediate batches there can't be dropped — every batch's data is
    unique.
    """

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self._logger = logging.getLogger("holoptycho.MosaicWriterOp")
        # First write of a run goes through the full-canvas path so the fill
        # colour gets painted across the whole buffer (including never-stitched
        # regions). Subsequent writes patch only the bbox of newly-placed
        # patches, which is ~30-100× cheaper over WAN. Reset on new-scan
        # detection in SaveViTResult (smallest index < max seen).
        self._first_write_done = False
        self._last_seen_batch_num = -1

    def setup(self, spec: OperatorSpec):
        # capacity=1 + POP gives single-slot, latest-wins semantics: while
        # this op is mid-write, any newer snapshot from upstream replaces
        # the queued one. We never hold more than one stale snapshot.
        spec.input("snapshot").connector(
            IOSpec.ConnectorType.DOUBLE_BUFFER,
            capacity=1,
            policy=IOSpec.QueuePolicy.POP,
        )

    def compute(self, op_input, op_output, context):
        try:
            snap = op_input.receive("snapshot")
        except Exception:
            self._logger.exception("MosaicWriterOp.receive failed")
            return
        if snap is None:
            return
        try:
            (mosaic, counts, mosaic_amp, counts_amp, batch_num, pixel_size_m,
             canvas_origin_um, bbox, total_batches, chunk_size) = snap
            t0 = time.perf_counter()

            # Reset on new-scan detection: SaveViTResult resets batch_num to 0
            # at the start of each scan. We mirror that here so the next first
            # write goes through the full-canvas path again.
            if batch_num < self._last_seen_batch_num:
                self._first_write_done = False
            self._last_seen_batch_num = batch_num

            # Threshold counts at 0.5 (not 0) to suppress FFT-leakage tails
            # from the Fourier-shift placement, which deposit tiny non-zero
            # counts well outside the patch footprints.
            def _normalise_full(m: np.ndarray, c: np.ndarray) -> np.ndarray:
                valid = c >= 0.5
                if valid.any():
                    avg = m / np.where(valid, c, 1.0)
                    fill = float(np.median(avg[valid]))
                    return np.where(valid, avg, fill).astype(np.float32)
                return np.zeros_like(m, dtype=np.float32)

            def _normalise_sub(m: np.ndarray, c: np.ndarray) -> np.ndarray:
                # Write avg_sub directly; unfilled pixels inside the bbox get
                # the local average (close to the seeded fill), and the visual
                # seam is negligible since the bbox tightly hugs new patches.
                valid_sub = c >= 0.5
                return (m / np.where(valid_sub, c, 1.0)).astype(np.float32)

            if not self._first_write_done:
                # Full-canvas write seeds the buffer with the fill colour
                # everywhere so unfilled regions don't render as black.
                normalised = _normalise_full(mosaic, counts)
                t_norm = time.perf_counter()
                _writer.write_vit_mosaic(
                    normalised,
                    batch_num=batch_num,
                    pixel_size_m=pixel_size_m,
                    canvas_origin_um=canvas_origin_um,
                )
                if mosaic_amp is not None and counts_amp is not None:
                    _writer.write_vit_amp_mosaic(
                        _normalise_full(mosaic_amp, counts_amp),
                        batch_num=batch_num,
                        pixel_size_m=pixel_size_m,
                        canvas_origin_um=canvas_origin_um,
                    )
                t_done = time.perf_counter()
                self._first_write_done = True
                self._logger.info(
                    "MosaicWriterOp: chunk %d/%d (%d frames) FULL "
                    "normalize=%.0f ms write=%.0f ms",
                    batch_num + 1, total_batches, chunk_size,
                    (t_norm - t0) * 1000, (t_done - t_norm) * 1000,
                )
                return

            # Incremental path: normalise + patch only the bbox of patches
            # placed in this batch (phase + amp in parallel).
            y0, y1, x0, x1 = bbox
            if y1 <= y0 or x1 <= x0:
                return  # empty bbox; nothing to write
            normalised_sub = _normalise_sub(mosaic[y0:y1, x0:x1], counts[y0:y1, x0:x1])
            t_norm = time.perf_counter()
            _writer.patch_vit_mosaic(
                normalised_sub,
                offset_yx=(y0, x0),
                batch_num=batch_num,
            )
            if mosaic_amp is not None and counts_amp is not None:
                _writer.patch_vit_amp_mosaic(
                    _normalise_sub(mosaic_amp[y0:y1, x0:x1], counts_amp[y0:y1, x0:x1]),
                    offset_yx=(y0, x0),
                    batch_num=batch_num,
                )
            t_done = time.perf_counter()
            self._logger.info(
                "MosaicWriterOp: chunk %d/%d (%d frames) bbox=%dx%d "
                "normalize=%.0f ms write=%.0f ms",
                batch_num + 1, total_batches, chunk_size,
                y1 - y0, x1 - x0,
                (t_norm - t0) * 1000, (t_done - t_norm) * 1000,
            )
        except Exception:
            self._logger.exception("MosaicWriterOp.compute failed")


class PositionsWriterOp(Operator):
    """Writes the latest positions_um snapshot to tiled on its own scheduler
    thread.

    Each tiled HTTPS PUT of positions_um is ~700 ms — measured via
    SaveViTResult's per-phase timing breakdown to be ~85% of the total
    SaveViTResult.compute time. Running that inline prevented the pipeline
    from keeping up with the 200 Hz publish rate.

    Same drop-policy semantics as ``MosaicWriterOp``: positions_um is fully
    overwritten on each write (it always contains the latest cumulative
    snapshot), so dropping intermediate snapshots is fine — only the most
    recent matters. ``capacity=1`` + ``QueuePolicy.POP`` evicts older
    snapshots while we're mid-PUT.
    """

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self._logger = logging.getLogger("holoptycho.PositionsWriterOp")

    def setup(self, spec: OperatorSpec):
        spec.input("snapshot").connector(
            IOSpec.ConnectorType.DOUBLE_BUFFER,
            capacity=1,
            policy=IOSpec.QueuePolicy.POP,
        )

    def compute(self, op_input, op_output, context):
        try:
            positions = op_input.receive("snapshot")
        except Exception:
            self._logger.exception("PositionsWriterOp.receive failed")
            return
        if positions is None:
            return
        try:
            t0 = time.perf_counter()
            _writer.write_positions(positions)
            # Always emit meter-unit x/y alongside the existing micron-unit
            # array under <run>/diffraction/. Tiny extra payload (~160 KB)
            # and gives ptycho-vit's loader the SI form it expects without
            # a unit conversion or a per-run feature flag.
            _writer.write_probe_positions_m(
                x_m=positions[:, 0] * 1e-6,
                y_m=positions[:, 1] * 1e-6,
            )
            self._logger.info(
                "PositionsWriterOp: wrote in %.0f ms",
                (time.perf_counter() - t0) * 1000,
            )
        except Exception:
            self._logger.exception("PositionsWriterOp.compute failed")


class BatchWriterOp(Operator):
    """Writes per-batch ViT (pred, indices) to tiled on its own scheduler thread.

    Each pred is ``(64, 2, 256, 256)`` float32 = ~33 MB; the tiled HTTPS PUT
    takes ~30 s. Running that on the upstream ``SaveViTResult`` thread would
    gate the whole ViT branch on the tiled write — exactly the bottleneck
    that was making a 50-second replay take ~80 minutes.

    Unlike ``MosaicWriterOp`` (single-slot, drop-on-overrun), this writer must
    NOT drop anything: every batch's ``pred``/``indices`` is unique data that
    offline analysts re-stitch from. The input connector is a bounded FIFO
    (``capacity=32``), so up to ~1 GB of pred buffers can queue in RAM while
    tiled drains. If the queue stays full, holoscan's
    ``DownstreamMessageAffordableCondition`` backpressures upstream so we
    never lose data.
    """

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self._logger = logging.getLogger("holoptycho.BatchWriterOp")

    def setup(self, spec: OperatorSpec):
        spec.input("batch").connector(
            IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32
        )

    def compute(self, op_input, op_output, context):
        try:
            msg = op_input.receive("batch")
        except Exception:
            self._logger.exception("BatchWriterOp.receive failed")
            return
        if msg is None:
            return
        try:
            batch_num, total_batches, chunk_size, pred, indices = msg
            t0 = time.perf_counter()
            _writer.write_vit(batch_num=batch_num, pred=pred, indices=indices)
            self._logger.info(
                "BatchWriterOp: chunk %d/%d (%d frames) wrote in %.0f ms",
                batch_num + 1, total_batches, chunk_size,
                (time.perf_counter() - t0) * 1000,
            )
        except Exception:
            self._logger.exception("BatchWriterOp.compute failed")
