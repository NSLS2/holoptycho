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
import re
import time
from pathlib import Path

import numpy as np

from holoscan.core import Operator, OperatorSpec, ConditionType, IOSpec
from .mosaic_stitch import stitch_batch_livestitch_into, stitch_batch_nearest


def find_onnx_for_engine(engine_path: str):
    """Return the ONNX path for a compiled .engine file, or None.

    Convention: engine  {dir}/{model_name}_v{version}.engine
                onnx    {dir}/onnx/{model_name}/{version}/*.onnx
    """
    ep = Path(engine_path)
    m = re.fullmatch(r"(.+?)_v(\d+)\.engine", ep.name)
    if not m:
        return None
    model_name, version = m.group(1), m.group(2)
    onnx_dir = ep.parent / "onnx" / model_name / version
    onnx_files = sorted(onnx_dir.glob("*.onnx"))
    return onnx_files[0] if onnx_files else None


def inner_crop_from_onnx(onnx_path, threshold: float = 0.50):
    """Derive ``inner_crop`` from the probe stored in an ONNX model.

    The probe defines which output pixels carry meaningful signal. For a
    circular probe of radius R pixels the largest inscribed square has
    half-side R/sqrt(2), giving::

        inner_crop = patch_size // 2 - floor(R / sqrt(2))

    Returns None if the ONNX cannot be loaded or a probe tensor is not found.
    """
    try:
        import onnx
        import onnx.numpy_helper as nph
    except ImportError:
        return None
    try:
        model = onnx.load(str(onnx_path))
    except Exception:
        return None

    init_map = {i.name: nph.to_array(i) for i in model.graph.initializer}
    out_names = [o.name for o in model.graph.output]

    if "probe_real" in init_map and "probe_imag" in init_map:
        p_re = init_map["probe_real"]
        p_im = init_map["probe_imag"]
    else:
        probe_cands = [
            init_map[n] for n in out_names
            if n != "output" and n in init_map and init_map[n].ndim == 2
        ]
        if len(probe_cands) < 2:
            return None
        p_re, p_im = probe_cands[0], probe_cands[1]

    if p_re.shape != p_im.shape or p_re.ndim != 2:
        return None

    amp = np.sqrt(p_re.astype(np.float64) ** 2 + p_im.astype(np.float64) ** 2)
    patch_h, patch_w = amp.shape
    cy, cx = patch_h / 2.0, patch_w / 2.0
    y_idx, x_idx = np.ogrid[:patch_h, :patch_w]
    r = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    support = r[amp >= threshold * amp.max()]
    if not len(support):
        return None
    radius = float(support.max())
    inscribed_half = radius / np.sqrt(2)
    inner_crop = int(np.floor(min(patch_h, patch_w) / 2.0 - inscribed_half))
    return max(0, min(inner_crop, min(patch_h, patch_w) // 4))


class PtychoViTInferenceOp(Operator):
    """Holoscan operator that runs PtychoViT TRT inference on diffraction batches.

    Delegates TRT engine loading, buffer allocation, and inference (including
    fftshift auto-detection, spatial padding, and final-batch padding) to
    ``ptychoml.PtychoViTInference``. This operator is a thin Holoscan adapter
    around that session.

    Inputs:
        diff_amp:      [B, H, W] float32 — ViT-normalised diffraction amplitude
                       (scale=10000, fftshift=False). Wired from
                       ImagePreprocessorOp.diff_amp_vit in ptycho_holo.py.
        image_indices: [B] int32 — frame indices

    Outputs:
        vit_result: tuple(pred, indices)
                    pred is [B, 2, H, W] (amplitude + phase) or [B, H, W]

    Parameters:
        engine_path: Path to .engine file
        gpu:         CUDA device ordinal (default 1; leave 0 for PtychoRecon)
        fftshift:    DC-convention override for the ptychoml session.
                     Default None lets ptychoml auto-detect per batch — robust
                     against upstream changes to fftshift policy. Pass True/False
                     only to force a fixed convention.
    """

    def __init__(
        self,
        fragment,
        *args,
        engine_path: str,
        gpu: int = 1,
        output_save_dir: str = "/data/users/Holoscan",
        fftshift: bool | None = None,
        **kwargs,
    ):
        super().__init__(fragment, *args, **kwargs)
        self._logger = logging.getLogger("PtychoViTInferenceOp")
        self.engine_path = engine_path
        self.gpu = gpu
        self.output_save_dir = output_save_dir
        self._fftshift = fftshift

        # Lazy-initialized on first compute() — PyCUDA contexts are
        # thread-local so the session must be created on the scheduler
        # worker thread, not here in __init__.
        self._session = None

        # Stats
        self.n_batches = 0
        self.total_infer_time = 0.0

    def _init_session(self):
        """Create and eagerly initialize the ptychoml inference session.

        Forces TRT engine deserialization at startup so the first batch is
        not slowed by the ~1–2 s load cost.
        """
        from ptychoml import PtychoViTInference

        if self.gpu == 0:
            self._logger.warning(
                "VIT running on GPU 0. On multi-GPU systems, prefer gpu=1 to "
                "keep PyCUDA (ViT) and CuPy (PtychoRecon) on separate devices."
            )
        self._session = PtychoViTInference(
            engine_path=self.engine_path,
            gpu=self.gpu,
            fftshift=self._fftshift,  # None = auto-detect DC per batch
        )
        self._session._init_engine()
        self.engine_batch_size = int(self._session.expected_input_shape[0])
        self._logger.info(
            "ptychoml.PtychoViTInference ready: engine=%s gpu=%d batch=%d",
            self.engine_path, self.gpu, self.engine_batch_size,
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
        # Engine is loaded lazily in _compute_inner() on first call.
        # PyCUDA contexts are thread-local: creating the context here
        # (framework startup thread) makes it unavailable in compute()
        # (MultiThreadScheduler worker thread). Cost is paid on first batch.
        pass

    def compute(self, op_input, op_output, context):
        try:
            self._compute_inner(op_input, op_output, context)
        except Exception:
            self._logger.exception("VIT inference failed (pipeline continues)")

    def _compute_inner(self, op_input, op_output, context):
        # Lazy session init on the scheduler worker thread
        if self._session is None:
            self._init_session()

        diff_amp = op_input.receive("diff_amp")
        indices  = op_input.receive("image_indices")

        if diff_amp is None:
            return

        # --- Save batch-average diffraction amplitude for live monitor ---
        # diff_amp is diff_amp_vit: sqrt(I/norm*10000), DC at center (no fftshift).
        # Use .tmp.npy suffix so np.save does not append a second .npy extension.
        try:
            _tmp = os.path.join(self.output_save_dir, "diff_avg_latest.tmp.npy")
            np.save(_tmp, diff_amp.mean(axis=0))
            os.replace(_tmp, os.path.join(self.output_save_dir, "diff_avg_latest.npy"))
        except Exception:
            pass

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

        # --- Run inference via ptychoml ---
        # Split into engine-sized sub-batches if pipeline batch > engine batch.
        # ptychoml handles the final partial sub-batch via internal padding.
        ebs = self.engine_batch_size
        n   = diff_amp.shape[0]
        t0  = time.perf_counter()
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
                self.n_batches, dt * 1000, avg_ms, pred.shape,
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
    """Save VIT predictions to disk and build a live phase mosaic.

    Per-batch file writes (existing behaviour, unchanged):
        vit_batch_000000_pred.npy    — predictions [B, 2, H, W] or [B, H, W]
        vit_batch_000000_indices.npy — frame indices [B]
        vit_pred_latest.npy          — most recent batch (atomic write)

    Mosaic stitching (new — AI inference visualization):
        vit_mosaic_latest.npy        — normalized phase mosaic (float32, NaN=unseen)
        vit_mosaic_amp_latest.npy    — normalized amplitude mosaic (when available)

    Stitching is enabled only when ``positions_provider``, ``pixel_size_m``,
    ``x_range_um``, and ``y_range_um`` are all supplied (wired in
    ptycho_holo.config_ops / compose). If any are absent the operator falls
    back to per-batch file writes only.
    """

    def __init__(self, fragment, *args,
                 save_dir="/data/users/Holoscan",
                 working_directory="", sign="t1",
                 # --- Mosaic stitching parameters (AI inference visualization) ---
                 positions_provider=None,  # callable → point_proc.positions_um
                 pixel_size_m=None,        # sample pixel size in metres
                 x_range_um=None,          # commanded slow-axis scan range (µm)
                 y_range_um=None,          # commanded fast-axis scan range (µm)
                 inner_crop=None,          # edge-artifact trim; None → patch_size//4
                 min_overlap_count=0.5,    # min patch overlap to show a pixel
                 phase_channel_index=1,    # output channel index for phase (0 or 1)
                 overshoot_factor=1.2,     # canvas safety margin over commanded range
                 total_frames=0,           # total expected scan frames (for progress bar)
                 save_batch_files=False,   # save per-batch vit_batch_*.npy files
                 **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self._logger = logging.getLogger("SaveViTResult")
        self.save_dir = save_dir
        self.working_directory = working_directory
        self.sign = sign
        self.batch_num = 0
        self.max_index_seen = -1
        self._total_frames = max(0, int(total_frames))
        self.save_batch_files = bool(save_batch_files)
        self._save_every = 5  # write mosaic/progress every N batches

        # Mosaic stitching state
        self._positions_provider = positions_provider
        self._pixel_size_m = pixel_size_m
        self._x_range_um = x_range_um
        self._y_range_um = y_range_um
        self._inner_crop = inner_crop
        self._min_overlap_count = max(0.5, float(min_overlap_count))
        self._phase_channel_index = phase_channel_index
        self._overshoot_factor = overshoot_factor

        # Lazy canvas — built on first batch once patch size is known
        self._mosaic = None        # (H, W) float32 running sum (phase)
        self._counts = None        # (H, W) float32 overlap counts
        self._mosaic_amp = None    # (H, W) float32 running sum (amplitude)
        self._counts_amp = None
        self._canvas_origin_um = None  # (origin_y_um, origin_x_um)
        self._half_h = 0
        self._half_w = 0
        self._pending_frames = []  # (pred_hw_or_2hw, frame_idx) with NaN positions

        # Stitching only active when all required params are provided
        self._stitch_enabled = (
            positions_provider is not None
            and pixel_size_m is not None and pixel_size_m > 0
            and x_range_um is not None and y_range_um is not None
        )
        if not self._stitch_enabled:
            self._logger.info(
                "ViT mosaic stitching disabled (missing pixel_size_m / "
                "x_range_um / y_range_um / positions_provider); "
                "per-batch .npy writes still active"
            )

    def setup(self, spec: OperatorSpec):
        # Default connector with POP policy drops oldest messages when full.
        # DOUBLE_BUFFER ignores QueuePolicy and raises GXF_EXCEEDING_PREALLOCATED_SIZE.
        spec.input("results", policy=IOSpec.QueuePolicy.POP).condition(ConditionType.NONE)

    def start(self):
        """Pre-warm the stitch kernel to avoid a JIT delay on the first batch."""
        try:
            dummy_canvas    = np.zeros((32, 32), dtype=np.float32)
            dummy_counts    = np.zeros_like(dummy_canvas)
            dummy_patches   = np.zeros((1, 8, 8), dtype=np.float32)
            dummy_positions = np.array([[16.0, 16.0]], dtype=np.float64)
            stitch_batch_nearest(dummy_canvas, dummy_counts,
                                 dummy_patches, dummy_positions)
            self._logger.info("SaveViTResult: stitch kernel pre-warmed")
        except Exception:
            self._logger.exception("Stitch pre-warm failed (non-fatal)")

    # ------------------------------------------------------------------
    # Mosaic state management
    # ------------------------------------------------------------------

    def _reset_mosaic(self):
        self._mosaic = None
        self._counts = None
        self._mosaic_amp = None
        self._counts_amp = None
        self._canvas_origin_um = None
        self._half_h = 0
        self._half_w = 0
        self._pending_frames.clear()

    def _ensure_canvas(self, patch_h, patch_w, positions_um):
        """Allocate the mosaic canvas on the first batch. Returns True if ready."""
        if self._mosaic is not None:
            return True
        finite = np.isfinite(positions_um).all(axis=1)
        if not finite.any():
            return False  # no positions yet — defer

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

        ps = self._pixel_size_m
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

        # Estimate canvas centre accounting for biased early-scan sampling
        def _mid(obs_min, obs_max, obs_range, cmd_range, vals):
            if obs_range >= cmd_range * 0.5 or cmd_range <= 0:
                return 0.5 * (obs_min + obs_max)
            direction = float(np.sign(vals[-1] - vals[0])) or -1.0
            start = obs_max if direction < 0 else obs_min
            return start + direction * cmd_range / 2.0

        x_mid_um = _mid(x_min_um, x_max_um, x_obs_um, self._x_range_um,
                        positions_um[finite, 0])
        y_mid_um = _mid(y_min_um, y_max_um, y_obs_um, self._y_range_um,
                        positions_um[finite, 1])

        canvas_h = int(np.ceil(y_range_um * 1e-6 / ps)) + 2 * half_h + 2
        canvas_w = int(np.ceil(x_range_um * 1e-6 / ps)) + 2 * half_w + 2
        origin_x_um = x_mid_um - (canvas_w / 2.0) * ps * 1e6
        origin_y_um = y_mid_um - (canvas_h / 2.0) * ps * 1e6

        self._mosaic     = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        self._counts     = np.zeros_like(self._mosaic)
        self._mosaic_amp = np.zeros_like(self._mosaic)
        self._counts_amp = np.zeros_like(self._mosaic)
        self._canvas_origin_um = (origin_y_um, origin_x_um)
        self._half_h = half_h
        self._half_w = half_w
        self._logger.info(
            "ViT mosaic canvas: %dx%d px (%.2f x %.2f µm), "
            "origin=(%.3f, %.3f) µm, cropped patch=%dx%d",
            canvas_h, canvas_w,
            canvas_h * ps * 1e6, canvas_w * ps * 1e6,
            origin_y_um, origin_x_um, cropped_h, cropped_w,
        )
        return True

    def _stitch_batch(self, pred, indices):
        """Stitch this batch into the in-memory mosaic.

        Returns True if at least one patch was placed, False otherwise.
        Handles NaN positions by buffering frames for retry on the next batch.
        """
        if not self._stitch_enabled:
            return False
        positions_um = self._positions_provider()
        if positions_um is None:
            return False
        positions_um = positions_um.copy()  # snapshot before PointProcessorOp mutates

        # Merge buffered frames whose positions have now arrived
        if self._pending_frames:
            ready = [
                (p, i) for p, i in self._pending_frames
                if i < len(positions_um) and np.isfinite(positions_um[i]).all()
            ]
            self._pending_frames = [
                (p, i) for p, i in self._pending_frames
                if not (i < len(positions_um)
                        and np.isfinite(positions_um[i]).all())
            ]
            if ready:
                self._logger.debug(
                    "ViT mosaic: re-stitching %d buffered frames", len(ready)
                )
                extra_pred = np.stack([p for p, _ in ready], axis=0)
                extra_idx  = np.array([i for _, i in ready], dtype=indices.dtype)
                pred    = np.concatenate([pred, extra_pred], axis=0)
                indices = np.concatenate([indices, extra_idx])

        patch_h = pred.shape[-2]
        patch_w = pred.shape[-1]

        if not self._ensure_canvas(patch_h, patch_w, positions_um):
            # Canvas not ready — buffer all frames until positions arrive
            for i in range(len(indices)):
                self._pending_frames.append((pred[i].copy(), int(indices[i])))
            return False

        # Extract phase and amplitude channels
        if pred.ndim == 4:   # (B, 2, H, W)
            phase_patches = pred[:, self._phase_channel_index].astype(np.float32)
            amp_patches   = pred[:, 1 - self._phase_channel_index].astype(np.float32)
        else:                # (B, H, W) — single channel, treat as phase
            phase_patches = pred.astype(np.float32)
            amp_patches   = None

        # Inner crop to remove edge artifacts
        c = self._inner_crop
        if c > 0:
            phase_patches = phase_patches[:, c:-c, c:-c]
            if amp_patches is not None:
                amp_patches = amp_patches[:, c:-c, c:-c]

        # Map positions µm → canvas pixels; buffer NaN frames for retry
        sub    = positions_um[indices]
        finite = np.isfinite(sub).all(axis=1)
        if not finite.all():
            n_nan = int((~finite).sum())
            self._logger.debug(
                "ViT mosaic: %d/%d frames have NaN positions — buffering",
                n_nan, len(indices),
            )
            for i in np.where(~finite)[0]:
                self._pending_frames.append((pred[i].copy(), int(indices[i])))
        if not finite.any():
            return False

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
                "ViT mosaic: all frames in batch fall outside canvas — "
                "increase overshoot_factor"
            )
            return False

        positions_px = np.stack([py[in_bounds], px[in_bounds]], axis=1)
        idx_ok = np.where(finite)[0][in_bounds]

        self._mosaic, self._counts, _ = stitch_batch_livestitch_into(
            self._mosaic, self._counts, phase_patches[idx_ok], positions_px,
        )
        if amp_patches is not None:
            self._mosaic_amp, self._counts_amp, _ = stitch_batch_livestitch_into(
                self._mosaic_amp, self._counts_amp, amp_patches[idx_ok], positions_px,
            )
        return True

    # ------------------------------------------------------------------
    # Existing helpers (unchanged)
    # ------------------------------------------------------------------

    def _save_final(self, scan_num):
        """Save vit_mosaic_latest and vit_mosaic_amp_latest to the recon_data directory."""
        import shutil
        recon_data_dir = os.path.join(self.working_directory,
                                      './recon_result/S' + str(scan_num) + '/' + self.sign + '/recon_data/')
        mosaic_src     = os.path.join(self.save_dir, "vit_mosaic_latest.npy")
        mosaic_amp_src = os.path.join(self.save_dir, "vit_mosaic_amp_latest.npy")
        if not os.path.exists(mosaic_src):
            self._logger.warning("vit_mosaic_latest.npy not found in %s; nothing to save.", self.save_dir)
            return
        try:
            os.makedirs(recon_data_dir, exist_ok=True)
            print(f"VIT mosaic results are being saved to {recon_data_dir}")
            shutil.copy2(mosaic_src,
                         os.path.join(recon_data_dir, f'recon_{scan_num}_{self.sign}_vit_mosaic.npy'))
            if os.path.exists(mosaic_amp_src):
                shutil.copy2(mosaic_amp_src,
                             os.path.join(recon_data_dir, f'recon_{scan_num}_{self.sign}_vit_mosaic_amp.npy'))
            self._logger.info("VIT mosaic results saved to %s", recon_data_dir)
        except Exception:
            self._logger.exception("Failed to save final VIT mosaic results")

    def _clear_old_batches(self):
        """Remove previous scan's batch files."""
        import glob as globmod
        for pattern in ["vit_batch_*_pred.npy", "vit_batch_*_indices.npy"]:
            for f in globmod.glob(os.path.join(self.save_dir, pattern)):
                try:
                    os.remove(f)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Main compute loop
    # ------------------------------------------------------------------

    def compute(self, op_input, op_output, context):
        try:
            results = op_input.receive("results")
            if results is None:
                return
            pred, indices = results

            os.makedirs(self.save_dir, exist_ok=True)

            # Detect new scan: smallest index in this batch less than max seen
            min_idx = int(indices.min())
            if min_idx < self.max_index_seen and self.batch_num > 0:
                self._clear_old_batches()
                self.batch_num = 0
                self.max_index_seen = -1
                self._reset_mosaic()

            self.max_index_seen = max(self.max_index_seen, int(indices.max()))

            _do_save = (self.batch_num % self._save_every == 0)

            # --- Optional per-batch .npy writes (off by default) ---
            if self.save_batch_files:
                np.save(
                    os.path.join(self.save_dir, f"vit_batch_{self.batch_num:06d}_pred.npy"),
                    pred,
                )
                np.save(
                    os.path.join(self.save_dir, f"vit_batch_{self.batch_num:06d}_indices.npy"),
                    indices,
                )
                tmp = os.path.join(self.save_dir, "_vit_pred_latest.tmp.npy")
                np.save(tmp, pred)
                os.replace(tmp, os.path.join(self.save_dir, "vit_pred_latest.npy"))

            # --- Mosaic stitching (every batch) + throttled save ---
            if self._stitch_enabled:
                try:
                    stitched = self._stitch_batch(pred, indices)
                    if stitched and _do_save:
                        # Normalize phase mosaic and save atomically
                        valid = self._counts >= self._min_overlap_count
                        norm = np.where(
                            valid,
                            self._mosaic / np.where(valid, self._counts, 1.0),
                            np.nan,
                        ).astype(np.float32)
                        tmp_m = os.path.join(self.save_dir, "_vit_mosaic.tmp.npy")
                        np.save(tmp_m, norm)
                        os.replace(tmp_m, os.path.join(
                            self.save_dir, "vit_mosaic_latest.npy"))

                        # Normalize amplitude mosaic if available
                        if self._mosaic_amp is not None and self._counts_amp is not None:
                            valid_a = self._counts_amp >= self._min_overlap_count
                            norm_a = np.where(
                                valid_a,
                                self._mosaic_amp / np.where(valid_a, self._counts_amp, 1.0),
                                np.nan,
                            ).astype(np.float32)
                            tmp_a = os.path.join(self.save_dir,
                                                 "_vit_mosaic_amp.tmp.npy")
                            np.save(tmp_a, norm_a)
                            os.replace(tmp_a, os.path.join(
                                self.save_dir, "vit_mosaic_amp_latest.npy"))
                except Exception:
                    self._logger.exception(
                        "Mosaic stitching/save failed (non-fatal)"
                    )

            # --- Throttled progress write ---
            if _do_save:
                try:
                    _tmp_p = os.path.join(self.save_dir, "_vit_progress.tmp.npy")
                    np.save(_tmp_p, np.array([self.max_index_seen + 1, self._total_frames], dtype=np.int64))
                    os.replace(_tmp_p, os.path.join(self.save_dir, "vit_progress.npy"))
                except Exception:
                    pass

            self.batch_num += 1
        except Exception:
            self._logger.exception("SaveViTResult.compute failed")
