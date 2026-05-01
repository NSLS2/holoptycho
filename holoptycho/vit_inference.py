"""
PtychoViT TensorRT inference operator for Holoscan pipeline.

Runs PtychoViT neural network inference in parallel with the iterative
PtychoRecon solver. Takes preprocessed diffraction amplitudes from
ImagePreprocessorOp (or InitSimul in simulate mode), runs TRT inference
via the ``ptychoml`` package, and saves predicted amplitude/phase patches
to disk.

No PyTorch imports — uses TensorRT + PyCUDA via ptychoml (safe for
NSLS-II container).

Usage:
    See ptycho_holo.py for wiring into PtychoApp / PtychoSimulApp.
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
        """Create the ptychoml inference session."""
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
        self.engine_batch_size = read_engine_batch_size(self.engine_path)
        self._logger.info(
            "ptychoml.PtychoViTInference created: engine=%s gpu=%d engine_batch=%d",
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
        inner_crop: int = 64,
        canvas_pad: int = 64,
        fourier_pad: int = 32,
        phase_channel_index: int = 1,
        **kwargs,
    ):
        super().__init__(fragment, *args, **kwargs)
        self.batch_num = 0
        self.max_index_seen = -1
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

        # Lazy state — built on first batch once we know the model's patch
        # size (``pred.shape[-1]``). Reset on each new scan.
        self._mosaic: np.ndarray | None = None
        self._counts: np.ndarray | None = None
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
        self._canvas_origin_um = None
        self._half_h = 0
        self._half_w = 0
        self._grow_log_count = 0
        # Tell the writer to drop the old mosaic node so the new run starts
        # with a fresh shape rather than failing the shape check.
        try:
            _writer.delete_vit_mosaic()
        except Exception:
            self._logger.exception("delete_vit_mosaic on reset failed (continuing)")

    def _ensure_canvas(self, patch_h: int, patch_w: int, positions_um: np.ndarray) -> bool:
        """Allocate canvas + counts on first batch. Returns True if ready."""
        if self._mosaic is not None:
            return True
        finite = np.isfinite(positions_um).all(axis=1)
        if not finite.any():
            # No finite positions yet — defer until PandA catches up.
            return False
        ps = self._pixel_size_m
        cropped_h = patch_h - 2 * self._inner_crop
        cropped_w = patch_w - 2 * self._inner_crop
        if cropped_h <= 0 or cropped_w <= 0:
            self._logger.error(
                "inner_crop=%d too large for patch %dx%d; disabling stitching",
                self._inner_crop, patch_h, patch_w,
            )
            self._stitch_enabled = False
            return False

        # Canvas covers max(observed, commanded) extent + one cropped half-
        # patch + canvas_pad on each side. Using observed lets us absorb
        # encoder overshoot during settling rows that exceed the commanded
        # range; falling back to commanded protects the live case where
        # only a few first-row positions have arrived and observed is a
        # small subset of the eventual scan.
        half_h = cropped_h // 2
        half_w = cropped_w // 2
        x_min_um = float(np.nanmin(positions_um[finite, 0]))
        x_max_um = float(np.nanmax(positions_um[finite, 0]))
        y_min_um = float(np.nanmin(positions_um[finite, 1]))
        y_max_um = float(np.nanmax(positions_um[finite, 1]))
        x_range_um = max(x_max_um - x_min_um, self._x_range_um)
        y_range_um = max(y_max_um - y_min_um, self._y_range_um)
        canvas_h = (
            int(np.ceil(y_range_um * 1e-6 / ps))
            + 2 * half_h + 2 + 2 * self._canvas_pad
        )
        canvas_w = (
            int(np.ceil(x_range_um * 1e-6 / ps))
            + 2 * half_w + 2 + 2 * self._canvas_pad
        )
        # Origin is the um-coordinate that maps to canvas pixel (0, 0):
        # the observed minimum minus the (half-patch + canvas_pad) buffer.
        origin_x_um = x_min_um - (half_w + 1 + self._canvas_pad) * ps * 1e6
        origin_y_um = y_min_um - (half_h + 1 + self._canvas_pad) * ps * 1e6

        self._mosaic = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        self._counts = np.zeros_like(self._mosaic)
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

    def _grow_canvas_if_needed(self, batch_positions_um: np.ndarray) -> bool:
        """Grow canvas + counts to fit ``batch_positions_um`` if any of those
        positions would push past the current canvas's per-side buffer.

        Old data is copied to its correct offset in the new (larger) canvas.
        Returns True if the canvas was grown (the writer should drop the old
        tiled node before the next write so the shape change takes effect).
        """
        if self._mosaic is None or self._canvas_origin_um is None:
            return False
        ps = self._pixel_size_m
        oy_um, ox_um = self._canvas_origin_um
        canvas_h, canvas_w = self._mosaic.shape
        # The "valid placement" window inside the canvas — the buffer we
        # require so a patch centered at the canvas edge still fits.
        margin_y = self._half_h + 1 + self._canvas_pad
        margin_x = self._half_w + 1 + self._canvas_pad
        margin_y_um = margin_y * ps * 1e6
        margin_x_um = margin_x * ps * 1e6

        # Are these positions already inside the valid placement window?
        bx_min = float(np.nanmin(batch_positions_um[:, 0]))
        bx_max = float(np.nanmax(batch_positions_um[:, 0]))
        by_min = float(np.nanmin(batch_positions_um[:, 1]))
        by_max = float(np.nanmax(batch_positions_um[:, 1]))
        cur_y_lo = oy_um + margin_y_um
        cur_y_hi = oy_um + (canvas_h - margin_y) * ps * 1e6
        cur_x_lo = ox_um + margin_x_um
        cur_x_hi = ox_um + (canvas_w - margin_x) * ps * 1e6
        fits = (
            by_min >= cur_y_lo and by_max <= cur_y_hi
            and bx_min >= cur_x_lo and bx_max <= cur_x_hi
        )
        # Log the first 3 grow-checks and every check where fits=False so we
        # can verify the canvas actually expands when positions overflow.
        log_count = getattr(self, "_grow_log_count", 0)
        if log_count < 3 or not fits:
            self._logger.info(
                "ViT mosaic grow-check #%d: batch bounds y=[%.3f, %.3f] x=[%.3f, %.3f] "
                "vs canvas-valid y=[%.3f, %.3f] x=[%.3f, %.3f] -> fits=%s",
                log_count, by_min, by_max, bx_min, bx_max,
                cur_y_lo, cur_y_hi, cur_x_lo, cur_x_hi, fits,
            )
            self._grow_log_count = log_count + 1
        if fits:
            return False

        # Otherwise, expand the bounding box to include the new positions
        # plus the margin, then size + offset accordingly.
        new_y_lo = min(by_min, cur_y_lo)
        new_y_hi = max(by_max, cur_y_hi)
        new_x_lo = min(bx_min, cur_x_lo)
        new_x_hi = max(bx_max, cur_x_hi)
        new_origin_y_um = new_y_lo - margin_y_um
        new_origin_x_um = new_x_lo - margin_x_um
        new_canvas_h = int(np.ceil((new_y_hi + margin_y_um - new_origin_y_um) / (ps * 1e6)))
        new_canvas_w = int(np.ceil((new_x_hi + margin_x_um - new_origin_x_um) / (ps * 1e6)))
        new_mosaic = np.zeros((new_canvas_h, new_canvas_w), dtype=np.float32)
        new_counts = np.zeros_like(new_mosaic)
        # Where the old (0, 0) lands in the new array.
        offset_y = int(round((oy_um - new_origin_y_um) / (ps * 1e6)))
        offset_x = int(round((ox_um - new_origin_x_um) / (ps * 1e6)))
        new_mosaic[offset_y:offset_y + canvas_h, offset_x:offset_x + canvas_w] = self._mosaic
        new_counts[offset_y:offset_y + canvas_h, offset_x:offset_x + canvas_w] = self._counts

        self._logger.info(
            "ViT mosaic canvas grew %dx%d -> %dx%d, origin (%.3f, %.3f) -> (%.3f, %.3f) um",
            canvas_h, canvas_w, new_canvas_h, new_canvas_w,
            oy_um, ox_um, new_origin_y_um, new_origin_x_um,
        )
        self._mosaic = new_mosaic
        self._counts = new_counts
        self._canvas_origin_um = (new_origin_y_um, new_origin_x_um)
        # Shape changed — make the writer drop the old node so the next
        # write recreates it with the new shape rather than 422'ing.
        try:
            _writer.delete_vit_mosaic()
        except Exception:
            self._logger.exception("delete_vit_mosaic after grow failed (write will likely fail)")
        return True

    def _stitch_batch(self, pred: np.ndarray, indices: np.ndarray) -> None:
        if not self._stitch_enabled:
            return
        positions_um = self._positions_provider()
        if positions_um is None:
            return
        if not self._ensure_canvas(pred.shape[-1], pred.shape[-1], positions_um):
            return

        phase = pred[:, self._phase_channel_index].astype(np.float32, copy=False)
        if self._inner_crop > 0:
            c = self._inner_crop
            phase = phase[:, c:-c, c:-c]

        # Map per-frame um → canvas px. positions_um columns: 0=x, 1=y.
        sub = positions_um[indices]
        finite = np.isfinite(sub).all(axis=1)
        if not finite.any():
            return

        # Grow the canvas if any of these positions falls outside the
        # current canvas's per-side buffer. Old data is copied into the
        # new (larger) canvas at the right offset; the origin shifts so
        # we have to recompute pixel coords AFTER this call.
        self._grow_canvas_if_needed(sub[finite])

        ps = self._pixel_size_m
        oy_um, ox_um = self._canvas_origin_um
        positions_px = np.empty((finite.sum(), 2), dtype=np.float64)
        positions_px[:, 0] = (sub[finite, 1] - oy_um) * 1e-6 / ps   # y
        positions_px[:, 1] = (sub[finite, 0] - ox_um) * 1e-6 / ps   # x
        batch = phase[finite]

        try:
            self._mosaic, self._counts = stitch_batch_into(
                self._mosaic,
                self._counts,
                batch,
                positions_px,
                pad=self._fourier_pad,
            )
        except Exception:
            self._logger.exception("stitch_batch_into failed (skipping batch)")
            return

        # Fill unfilled regions with the median of the valid pixels rather
        # than NaN — tiled's PNG renderer treats NaN as 0, which would pull
        # the colormap range down to 0 and render the actual signal at the
        # bright end with no contrast. Threshold counts at 0.5 (not 0) to
        # suppress FFT-leakage tails from the Fourier-shift placement, which
        # deposit tiny non-zero counts well outside the patch footprints.
        valid = self._counts >= 0.5
        if valid.any():
            avg = self._mosaic / np.where(valid, self._counts, 1.0)
            fill = float(np.median(avg[valid]))
            normalised = np.where(valid, avg, fill).astype(np.float32)
        else:
            normalised = np.zeros_like(self._mosaic, dtype=np.float32)
        _writer.write_vit_mosaic(
            normalised,
            batch_num=self.batch_num,
            pixel_size_m=self._pixel_size_m,
            canvas_origin_um=self._canvas_origin_um,
        )

    def setup(self, spec: OperatorSpec):
        spec.input("results").connector(
            IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32
        )

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

            # Write positions BEFORE the per-batch container so any WebSocket
            # subscriber that wakes on the new batch sees an already-fresh
            # positions_um and can stitch with the real per-frame positions.
            if self._positions_provider is not None:
                positions = self._positions_provider()
                if positions is not None:
                    _writer.write_positions(positions)

            _writer.write_vit(
                batch_num=self.batch_num,
                pred=pred,
                indices=indices,
            )

            # Accumulate this batch's phase into the running mosaic and
            # publish the normalised result. On error this is logged but
            # does not stop the per-batch publish above.
            self._stitch_batch(pred, indices)

            self.batch_num += 1
        except Exception:
            pass
