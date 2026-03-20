"""
PtychoViT TensorRT inference operator for Holoscan pipeline.

Runs PtychoViT neural network inference in parallel with the iterative
PtychoRecon solver. Takes preprocessed diffraction amplitudes from
ImagePreprocessorOp (or InitSimul in simulate mode), runs TRT inference,
and saves predicted amplitude/phase patches to disk.

No PyTorch imports — uses TensorRT + PyCUDA only (safe for NSLS-II container).

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

    Inputs:
        diff_amp:      [B, H, W] float32 — preprocessed diffraction amplitude
        image_indices: [B] int32 — frame indices (for correlating with scan positions)

    Outputs:
        vit_result: tuple(pred, indices) where pred is [B, 2, H, W] or [B, H, W]

    Parameters:
        engine_path:       Path to .engine file (must match batch size B)
        gpu:               CUDA device ordinal (default 1; leave 0 for PtychoRecon)
        output_save_dir:   Directory for saving predictions (default /data/users/Holoscan)
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

        # Lazy-initialized in first compute()
        self._initialized = False
        self.cuda_ctx = None
        self.trt_context = None
        self.trt_inputs = None
        self.trt_outputs = None
        self.trt_bindings = None
        self.trt_stream = None
        self.expected_input_shape = None
        self.expected_output_shape = None

        # Stats
        self.n_batches = 0
        self.total_infer_time = 0.0

    def _init_engine(self):
        """Initialize TRT engine and CUDA context. Called once on first compute()."""
        import pycuda.driver as drv

        drv.init()
        if self.gpu == 0:
            self._logger.warning(
                "VIT running on GPU 0 — same as PtychoRecon (CuPy). "
                "PyCUDA + CuPy on the same GPU from different threads can cause "
                "CUDA context crashes. Use gpu=1 on multi-GPU systems."
            )
        self.cuda_ctx = drv.Device(self.gpu).make_context()
        self._logger.info(
            "PyCUDA context created on GPU %d (%s)",
            self.gpu,
            drv.Device(self.gpu).name(),
        )

        from .edgePtychoViT.helper_trt import load_engine, allocate_io_buffers

        engine = load_engine(self.engine_path)
        self.trt_context = engine.create_execution_context()
        self.trt_inputs, self.trt_outputs, self.trt_bindings, self.trt_stream = (
            allocate_io_buffers(engine)
        )

        self.expected_input_shape = tuple(self.trt_inputs[0]["shape"])
        self.expected_output_shape = tuple(self.trt_outputs[0]["shape"])
        self._logger.info(
            "TRT engine loaded: %s | input=%s | output=%s",
            self.engine_path,
            self.expected_input_shape,
            self.expected_output_shape,
        )
        self._initialized = True

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
        if not self._initialized:
            self._init_engine()

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
                    if self.cuda_ctx:
                        self.cuda_ctx.pop()
                    self.engine_path = new_path
                    self._initialized = False
                    os.remove(reload_file)
                    self._init_engine()
                    self._logger.info("Engine reload complete: %s", new_path)
                else:
                    os.remove(reload_file)
            except Exception as e:
                self._logger.warning("Engine reload failed: %s", e)

        # --- Prepare input ---
        # Model was trained on unshifted diffraction amplitudes (DC at corners).
        # In simulate mode (diffamp path): data arrives unshifted — no action needed.
        # In live mode / simulate raw_data path: data arrives fftshift'd by
        # ImagePreprocessorOp / InitSimul — undo the shift so model sees DC at corners.
        # For even-sized arrays, fftshift == ifftshift, so applying fftshift
        # to already-shifted data returns it to unshifted.
        if self._data_is_shifted:
            diff_amp = np.fft.fftshift(diff_amp, axes=(1, 2))
        B_actual = diff_amp.shape[0]
        H_data = diff_amp.shape[1]
        W_data = diff_amp.shape[2]
        B_engine = self.expected_input_shape[0]
        H_engine = self.expected_input_shape[2]
        W_engine = self.expected_input_shape[3]

        model_input = diff_amp[:, np.newaxis, :, :]  # [B_actual, 1, H_data, W_data]

        # Spatial padding: if data is smaller than engine input (e.g. 128x128 data,
        # 256x256 model), center-pad with zeros.  Crop output back after inference.
        self._spatial_pad = None
        if H_data != H_engine or W_data != W_engine:
            pad_h = H_engine - H_data
            pad_w = W_engine - W_data
            if pad_h < 0 or pad_w < 0:
                self._logger.error(
                    "Data spatial dims (%d,%d) larger than engine (%d,%d). "
                    "Cannot run inference.", H_data, W_data, H_engine, W_engine,
                )
                return
            top = pad_h // 2
            left = pad_w // 2
            self._spatial_pad = (top, top + H_data, left, left + W_data)
            padded = np.zeros(
                (B_actual, 1, H_engine, W_engine), dtype=np.float32
            )
            padded[:, :, top:top + H_data, left:left + W_data] = model_input
            model_input = padded
            if self.n_batches == 0:
                self._logger.info(
                    "Spatial pad: %dx%d -> %dx%d (center-padded)",
                    H_data, W_data, H_engine, W_engine,
                )

        # Pad final batch if smaller than engine batch size (e.g. scan_points % B != 0)
        if B_actual < B_engine:
            pad = np.zeros((B_engine - B_actual, 1, H_engine, W_engine), dtype=np.float32)
            model_input = np.concatenate([model_input, pad], axis=0)
            self._logger.info(
                "Final batch: padded %d -> %d frames", B_actual, B_engine
            )
        elif B_actual > B_engine:
            self._logger.error(
                "Batch too large: input %d vs engine %d. "
                "Check that ONNX batch size matches ImageBatchOp batchsize.",
                B_actual, B_engine,
            )
            return

        model_input = np.ascontiguousarray(model_input, dtype=np.float32)

        # --- Run TRT inference ---
        np.copyto(self.trt_inputs[0]["host"], model_input.ravel())

        from .edgePtychoViT.helper_trt import infer, reshape_output_flat

        t0 = time.perf_counter()
        output_flat = np.array(
            infer(
                self.trt_context,
                self.trt_inputs,
                self.trt_outputs,
                self.trt_bindings,
                self.trt_stream,
                cuda_context=self.cuda_ctx,
            )[0]
        )
        dt = time.perf_counter() - t0

        # --- Reshape output and strip padding ---
        pred = reshape_output_flat(output_flat, batch_size=B_engine, height=H_engine, width=W_engine)
        pred = pred[:B_actual]  # strip padding rows if final batch was padded

        # Crop spatial padding back to original data dimensions
        if self._spatial_pad is not None:
            top, bot, left, right = self._spatial_pad
            if pred.ndim == 4:  # [B, 2, H, W]
                pred = pred[:, :, top:bot, left:right]
            else:  # [B, H, W]
                pred = pred[:, top:bot, left:right]

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
        if self.cuda_ctx is not None:
            try:
                self.cuda_ctx.pop()
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
