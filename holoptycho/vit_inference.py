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
        self._logger.info(
            "ptychoml.PtychoViTInference created: engine=%s gpu=%d",
            self.engine_path,
            self.gpu,
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
        t0 = time.perf_counter()
        pred, _ = self._session.predict(diff_amp)
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
    """Save VIT predictions to disk as per-batch files (O(1) per batch).

    Keeps state in memory (no counter file). Detects new scans by watching
    for frame indices that reset to 0, then clears old batch files.

    Writes:
        vit_batch_000000_pred.npy    — predictions for this batch [B, 2, H, W]
        vit_batch_000000_indices.npy — frame indices for this batch [B]
        vit_pred_latest.npy          — most recent batch (atomic write)

    Concatenate after scan:
        preds = np.concatenate([np.load(f) for f in sorted(glob('vit_batch_*_pred.npy'))])
    """

    def __init__(self, fragment, *args, save_dir="/data/users/Holoscan", **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.save_dir = save_dir
        self.batch_num = 0
        self.max_index_seen = -1

    def setup(self, spec: OperatorSpec):
        spec.input("results").connector(
            IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32
        )

    def _clear_old_batches(self):
        """Remove previous scan's batch files."""
        import glob as globmod
        for pattern in ["vit_batch_*_pred.npy", "vit_batch_*_indices.npy"]:
            for f in globmod.glob(os.path.join(self.save_dir, pattern)):
                try:
                    os.remove(f)
                except OSError:
                    pass

    def compute(self, op_input, op_output, context):
        try:
            results = op_input.receive("results")
            if results is None:
                return
            pred, indices = results

            os.makedirs(self.save_dir, exist_ok=True)

            # Detect new scan: if smallest index in this batch is less than
            # what we've seen, a new scan has started
            min_idx = int(indices.min())
            if min_idx < self.max_index_seen and self.batch_num > 0:
                self._clear_old_batches()
                self.batch_num = 0
                self.max_index_seen = -1

            self.max_index_seen = max(self.max_index_seen, int(indices.max()))

            # Save batch files
            np.save(
                os.path.join(self.save_dir, f"vit_batch_{self.batch_num:06d}_pred.npy"),
                pred,
            )
            np.save(
                os.path.join(self.save_dir, f"vit_batch_{self.batch_num:06d}_indices.npy"),
                indices,
            )

            # Atomic write of latest batch for quick inspection
            tmp = os.path.join(self.save_dir, "_vit_pred_latest.tmp.npy")
            np.save(tmp, pred)
            os.replace(tmp, os.path.join(self.save_dir, "vit_pred_latest.npy"))

            self.batch_num += 1
        except Exception:
            pass
