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
        self._logger.info(
            "ViT mosaic canvas allocated: %dx%d px (%.2f x %.2f um), origin=(%.3f, %.3f) um, "
            "cropped patch=%dx%d",
            canvas_h, canvas_w,
            canvas_h * ps * 1e6, canvas_w * ps * 1e6,
            origin_y_um, origin_x_um,
            cropped_h, cropped_w,
        )
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
