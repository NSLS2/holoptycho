import logging
import os
import sys
import time

import numpy as np
import cupy as cp

from ptychoml.preprocess import (
    apply_d4,
    apply_intensity_floor,
    crop_to_roi,
    inpaint_bad_pixels,
    preprocess_diffraction,
)

from .orientation import compute_pos_bases, reorient_d4

from holoscan.core import Operator, OperatorSpec, ConditionType, IOSpec
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler, EventBasedScheduler
from holoscan.logger import LogLevel, set_log_level
from holoscan.decorator import create_op, Input


def compute_center_box(headroom_batch, headroom_roi, nx, ny):
    """One-shot segmentation-based centered crop box (lossless auto-centering).

    Given a batch of (coordinate-corrected, global) detector frames cropped to
    ``headroom_roi`` — a ``[[y0,y1],[x0,x1]]`` window in global coords — find the
    diffraction beam and return the ``ny x nx`` crop box centered on it, in the
    same global convention as the operator's ``self.roi``.

    Segmentation: average the batch (protects against an odd empty/saturated
    first frame), mask saturated pixels (which would bias the centroid),
    threshold at 5% of peak to isolate the blob, run ``scipy.ndimage.label``,
    and take the centroid of the largest connected component. The box is clamped
    to stay inside ``headroom_roi``.

    Returns ``(box, clamped)`` where ``box`` is an integer ``np.array`` ROI, or
    ``(None, False)`` when no blob is found (caller falls back to the configured
    ROI). ``clamped`` is True when the beam sat beyond the headroom and the box
    was pushed to the window edge.
    """
    from scipy.ndimage import label, center_of_mass

    avg = headroom_batch.astype(np.float32).mean(axis=0)
    # Saturation mask: pixels at/near uint max are hot/bad and would bias the
    # centroid. Drop them from the segmentation input (floats: no saturation).
    try:
        sat = float(np.iinfo(headroom_batch.dtype).max) - 1.0
    except ValueError:
        sat = float('inf')
    masked = np.where(avg >= sat, 0.0, avg)
    peak = float(masked.max())
    if peak <= 0:
        return None, False
    binary = masked > (0.05 * peak)
    labels, n_obj = label(binary)
    if n_obj == 0:
        return None, False
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    cy, cx = center_of_mass(masked, labels, largest)  # window-local (row, col)

    Hh, Wh = avg.shape
    hy0 = int(headroom_roi[0, 0])
    hx0 = int(headroom_roi[1, 0])
    # Absolute raw-detector centroid, then box centered on it.
    y0 = int(round(hy0 + cy - ny / 2.0))
    x0 = int(round(hx0 + cx - nx / 2.0))
    # Clamp so the ny x nx box stays inside the headroom window.
    y0c = min(max(y0, hy0), hy0 + Hh - ny)
    x0c = min(max(x0, hx0), hx0 + Wh - nx)
    clamped = (y0c != y0) or (x0c != x0)
    return np.array([[y0c, y0c + ny], [x0c, x0c + nx]]), clamped


class ImageBatchOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.logger = logging.getLogger("ImageBatchOp")
        logging.basicConfig(level=logging.INFO)
        self.counter = 0

        # Local→global coordinate correction (a ptychoml D4 name) applied to the
        # WHOLE raw frame before cropping. Source-dependent: live Eiger ZMQ is
        # raw-local → 'fliplr'; replay-from-Tiled is already corrected → 'identity'.
        # Set from config by compose(). Default 'identity' is a zero-cost no-op
        # for safety before compose runs.
        self.detector_orientation = 'identity'
        self.batchsize = 0
        self.nx_prb = 0
        self.ny_prb = 0
        self.images_to_add = None #np.zeros((self.batchsize, 256, 256))
        self.indices_to_add = None #np.zeros(self.batchsize, dtype=np.int32)
        # ROI is normally set by compose(); default None so compute()'s guard
        # (`if self.roi is None: return`) is safe before that runs.
        self.roi = None

        # Opt-in lossless auto-centering (off by default — OFF path is identical
        # to the historical fixed-ROI crop). When on, the FIRST batch is buffered
        # at a larger headroom window, segmented to find the beam, and a centered
        # ny x nx crop box is computed once and reused for every batch (incl. the
        # first). Set from config by compose(). See compute_center_box().
        self.auto_center = False
        self.headroom = 0
        self._center_box = None       # cached ny x nx global ROI once computed
        self._hr_buf = None           # transient (batchsize, Hh, Wh) first-batch buffer
        self._headroom_roi = None     # global-coords headroom window

        # Per-second compute() throughput counters. See note in EigerZmqRxOp.
        self._diag_window_start = time.time()
        self._diag_calls = 0
        self._diag_batches_emitted = 0

    def flush(self,param):
        self.counter = 0
        self.roi = np.array(param)
        
    def setup(self, spec: OperatorSpec):
        # capacity=4096 (~4 s of buffer at 1000 fps) + REJECT propagates
        # backpressure upstream during the initial burst when this op is
        # spinning up — see EigerDecompressOp for the same rationale.
        spec.input("image").connector(
            IOSpec.ConnectorType.DOUBLE_BUFFER,
            capacity=4096,
            policy=IOSpec.QueuePolicy.REJECT,
        )
        spec.input("image_index").connector(
            IOSpec.ConnectorType.DOUBLE_BUFFER,
            capacity=4096,
            policy=IOSpec.QueuePolicy.REJECT,
        )
        spec.output("image_batch")
        spec.output("image_indices")
        
    def compute(self, op_input, op_output, context):
        self._diag_calls += 1
        now = time.time()
        if now - self._diag_window_start >= 1.0:
            self.logger.debug(
                "ImageBatchOp 1s: calls=%d batches_emitted=%d",
                self._diag_calls, self._diag_batches_emitted,
            )
            self._diag_window_start = now
            self._diag_calls = 0
            self._diag_batches_emitted = 0

        image = op_input.receive("image")
        image_index = op_input.receive("image_index")

        if self.roi is None:
            return

        # Step 1: coordinate-correct the WHOLE raw frame local->global once.
        # Everything below (ROI crop, auto-center segmentation) then works in
        # global coords, so the ROI is a plain crop and never needs flipping.
        # 'identity' (replay, Tiled already corrected) is a zero-cost no-op.
        if self.detector_orientation != 'identity':
            image = np.ascontiguousarray(apply_d4(image, self.detector_orientation))

        # Step 2 (auto-center variant): buffer the FIRST batch at a headroom
        # window and compute a beam-centered crop box before emitting anything.
        # Only entered when enabled and the box isn't computed yet; once set,
        # falls through to the plain crop below. OFF path never enters here.
        if self.auto_center and self._center_box is None:
            self._accumulate_first_batch(image, image_index, op_output)
            return

        # Step 2 (steady state + OFF path): plain crop to the active global ROI —
        # the computed centered box if auto-centering produced one, else the
        # configured global ROI. Output is always nx x ny.
        active_roi = self.roi if self._center_box is None else self._center_box
        image = crop_to_roi(image, active_roi)

        # Remove Bad pixels (-1 to unsigned int)
        image = np.array(image)  # crop_to_roi may return a view; don't mutate source
        image[image==np.iinfo(image.dtype).max] = 0

        self.images_to_add[self.counter, :, :] = image
        self.indices_to_add[self.counter] = image_index

        # sys.stderr.write(f"Received image {image_index}\n")

        if self.counter < (self.batchsize - 1):
            self.counter += 1
        else:
            op_output.emit(self.images_to_add.copy(), "image_batch")
            op_output.emit(self.indices_to_add.copy(), "image_indices")
            self.counter = 0
            self._diag_batches_emitted += 1

    def _headroom_roi_for(self, frame_shape):
        """Global-coords headroom window to search for the beam. ``headroom < 0``
        searches the WHOLE frame (used when no crop ROI was given — the beam can
        be anywhere in the full detector). Otherwise the configured ROI grown by
        ``headroom`` on each side, clamped to the frame bounds (known only at
        compute time)."""
        H, W = int(frame_shape[0]), int(frame_shape[1])
        M = int(self.headroom)
        if M < 0:
            return np.array([[0, H], [0, W]])
        y0, y1 = int(self.roi[0, 0]), int(self.roi[0, 1])
        x0, x1 = int(self.roi[1, 0]), int(self.roi[1, 1])
        return np.array([
            [max(0, y0 - M), min(H, y1 + M)],
            [max(0, x0 - M), min(W, x1 + M)],
        ])

    def _accumulate_first_batch(self, image, image_index, op_output):
        """Buffer the first batch at the headroom window, then on the full batch
        compute the centered crop box and emit the batch. ``image`` is already
        coordinate-corrected (global) by compute(), so the box and crop are in
        global coords — no flipping anywhere."""
        if self._hr_buf is None:
            self._headroom_roi = self._headroom_roi_for(image.shape)
            hy0, hy1 = int(self._headroom_roi[0, 0]), int(self._headroom_roi[0, 1])
            hx0, hx1 = int(self._headroom_roi[1, 0]), int(self._headroom_roi[1, 1])
            self._hr_buf = np.zeros(
                (self.batchsize, hy1 - hy0, hx1 - hx0), dtype=image.dtype
            )

        # Crop the (global) headroom window for segmentation; zero bad pixels to
        # match the steady-state path. crop_to_roi may return a view — copy.
        hr = np.array(crop_to_roi(image, self._headroom_roi))
        hr[hr == np.iinfo(image.dtype).max] = 0
        self._hr_buf[self.counter, :, :] = hr
        self.indices_to_add[self.counter] = image_index

        if self.counter < (self.batchsize - 1):
            self.counter += 1
            return

        # First batch full — compute the centered ny x nx box (derive nx/ny from
        # the configured ROI so it matches images_to_add's allocation).
        ny = int(self.roi[0, 1] - self.roi[0, 0])
        nx = int(self.roi[1, 1] - self.roi[1, 0])
        box, clamped = compute_center_box(self._hr_buf, self._headroom_roi, nx, ny)
        if box is None:
            self._center_box = np.array(self.roi)
            self.logger.warning(
                "Auto-centering: no diffraction blob in the first batch; "
                "falling back to the configured ROI."
            )
        else:
            self._center_box = box
            if clamped:
                self.logger.warning(
                    "Auto-centering: beam beyond the ±%d px headroom; crop box "
                    "clamped to the window edge. Adjust batch_x0/y0 or headroom.",
                    int(self.headroom),
                )
            self.logger.info(
                "Auto-centering: centered crop box rows %d:%d cols %d:%d",
                int(box[0, 0]), int(box[0, 1]), int(box[1, 0]), int(box[1, 1]),
            )

        # Emit the buffered first batch cropped to the centered box. The buffer
        # is already coordinate-corrected, so slicing it to the box is exactly
        # what the steady-state crop produces — batch 0 matches all later batches.
        hy0 = int(self._headroom_roi[0, 0])
        hx0 = int(self._headroom_roi[1, 0])
        ry0, ry1 = int(self._center_box[0, 0]) - hy0, int(self._center_box[0, 1]) - hy0
        rx0, rx1 = int(self._center_box[1, 0]) - hx0, int(self._center_box[1, 1]) - hx0
        for i in range(self.batchsize):
            self.images_to_add[i, :, :] = self._hr_buf[i, ry0:ry1, rx0:rx1]
        op_output.emit(self.images_to_add.copy(), "image_batch")
        op_output.emit(self.indices_to_add.copy(), "image_indices")
        self.counter = 0
        self._hr_buf = None
        self._diag_batches_emitted += 1


class ImagePreprocessorOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.logger = logging.getLogger("ImagePreprocessorOp")
        logging.basicConfig(level=logging.INFO)
        # self.roi = np.array(roi)
        self.detmap_threshold = 0
        self.badpixels = None
        # NOTE: diffraction auto-centering lives in ImageBatchOp now (a lossless
        # centered crop computed once on the first batch — see
        # compute_center_box). This operator no longer shifts frames; it receives
        # them already centered, so the intensity tap and model input stay in
        # lockstep exactly as before.
        # Geometry knobs for the two output branches.
        #   tap_orient: D4 element applied to the intensity tap (saved /dp).
        #               Default 'antitranspose' matches the historical HXN
        #               anti-diagonal flip applied to bring data into the
        #               beamline operator's view; the saved DP looks how the
        #               operator expects.
        #   dp_orient:  D4 element on the model-input branch. Default
        #               'identity' — the frame is already in the global
        #               coordinate system (detector_orientation). Set
        #               dp_orient='auto' in the config to opt in to the ViT
        #               orientation-autodetect sweep, which overwrites this.
        #   fftshift_dp: model-input DC-convention control. ``None``
        #               (default) lets ptychoml auto-detect via
        #               ``detect_dc_at_corner`` and shift only when the
        #               central beam is at the corners. Override with
        #               ``True``/``False`` from the scan config if a
        #               specific dataset misbehaves; otherwise leave alone.
        # All three are settable from the scan config; see ptycho_holo.py.
        self.tap_orient = 'antitranspose'
        self.dp_orient = 'identity'
        self.fftshift_dp: bool | None = None
        # Intensity normalization passed straight through to
        # ptychoml.preprocess_diffraction so each DP gets scaled by the
        # same constant the offline pipeline used. ``normalization`` is the
        # per-scan max intensity (hot pixels excluded) — see
        # ptychoml.compute_intensity_normalization. The default of 1e5 is a
        # placeholder; in production the scan JSON overrides it per-scan.
        self.normalization = 1.0e5
        self.scale = 1.0e4
        # Photon-count threshold for hot-pixel zeroing (None = disabled).
        # 50000 matches hxn_to_vit.py's default.
        self.hot_pixel_count_threshold = None

        # Raw-intensity buffer for orientation auto-detect. PtychoViTInferenceOp
        # reads this to run ptychoml.autodetect_orientation on the first batch
        # of frames that has enough finite scan positions, then updates
        # dp_orient. Entries are (processed_images, indices) tuples of pre-D4
        # frames. Capped at _AUTODETECT_BUF_MAX_FRAMES total frames; cleared
        # when the sweep is done (it sets _autodetect_done=True).
        self._autodetect_buf: list = []
        self._autodetect_done: bool = False
        self._AUTODETECT_BUF_MAX_FRAMES: int = 256
        super().__init__(*args, **kwargs)

        # Per-second compute() throughput counters. See note in EigerZmqRxOp.
        self._diag_window_start = time.time()
        self._diag_calls = 0
        self._diag_total_ms = 0.0

    def setup(self, spec: OperatorSpec):
        spec.input("image_batch").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)
        spec.input("image_indices_in").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)
        spec.output("diff_amp")
        # Separate RAW-amplitude output for the iterative engine (normalization=1,
        # scale=1 => sqrt(intensity)). The ViT branch consumes the vit-normalized
        # ``diff_amp``; the engine must NOT — its frozen warm-start probe has a
        # fixed amplitude scale, so the ViT normalization collapses |object|.
        spec.output("diff_amp_engine")
        spec.output("image_indices")
        # Detector-frame intensity tap, captured before rot90/fftshift/floor/sqrt.
        # Consumed only by FrameWriterOp when fine_tune writes are enabled.
        spec.output("intensity")
        # The dp_orient this batch was actually preprocessed with, emitted in
        # lockstep with diff_amp. ImageSendOp needs it per batch (not read live
        # from this op) because the orientation auto-detect mutates dp_orient at
        # runtime while up to capacity=32 batches sit queued downstream.
        spec.output("dp_orient_used")

    def compute(self, op_input, op_output, context):
        t0 = time.perf_counter()
        self._diag_calls += 1
        now = time.time()
        if now - self._diag_window_start >= 1.0:
            self.logger.debug(
                "ImagePreprocessorOp 1s: calls=%d total=%.1f ms",
                self._diag_calls, self._diag_total_ms,
            )
            self._diag_window_start = now
            self._diag_calls = 0
            self._diag_total_ms = 0.0

        images = op_input.receive("image_batch")
        indices = op_input.receive("image_indices_in")
        
        processed_images = np.asarray(images)

        # self.badpixels is shape (2, K) with rows=[row_indices, col_indices];
        # transpose to (K, 2) for inpaint_bad_pixels' coords format.
        inpaint_bad_pixels(processed_images, self.badpixels.T)

        # Buffer pre-D4 frames for the one-shot orientation auto-detect.
        # autodetect_orientation sweeps the D4 candidates itself, so it needs
        # the frames *before* tap_orient/dp_orient are applied. Once
        # PtychoViTInferenceOp has run the sweep it sets _autodetect_done=True.
        if not self._autodetect_done:
            n_buffered = sum(f.shape[0] for f, _ in self._autodetect_buf)
            if n_buffered < self._AUTODETECT_BUF_MAX_FRAMES:
                self._autodetect_buf.append(
                    (processed_images.copy(), indices.copy())
                )

        # Tap branch: apply the configured D4 to put the saved intensity
        # into whatever orientation the operator wants to see on the
        # dashboard. Default 'antitranspose' reproduces the historical HXN
        # anti-diagonal flip (transpose ∘ flip-both-axes); set
        # tap_orient='identity' to save raw detector frames. Contiguous
        # copy so the emitted buffer doesn't alias processed_images, which
        # is mutated in place below.
        tap = np.ascontiguousarray(apply_d4(processed_images, self.tap_orient))
        op_output.emit(tap, "intensity")

        # Model branch: delegate the entire normalize → mask → sqrt → D4 →
        # fftshift sequence to ptychoml.preprocess_diffraction. Bad pixels
        # are already inpainted above; the intensity floor (low-threshold)
        # stays a holoptycho-side knob applied before the call because
        # preprocess_diffraction doesn't expose it. fftshift=None (the
        # default for this op) lets ptychoml auto-detect the central beam
        # position and shift only when needed.
        if self.detmap_threshold > 0:
            apply_intensity_floor(processed_images, self.detmap_threshold)
        # Snapshot dp_orient once so the preprocess call and the stamp emitted
        # below can't disagree (the autodetect mutates it from another thread).
        dp_orient = self.dp_orient
        diff_amp = preprocess_diffraction(
            processed_images,
            normalization=self.normalization,
            scale=self.scale,
            hot_pixel_count_threshold=self.hot_pixel_count_threshold,
            dp_orient=dp_orient,
            fftshift=self.fftshift_dp,
        )

        op_output.emit(diff_amp, "diff_amp")
        # Iterative-engine branch: feed RAW sqrt(intensity) (normalization=1,
        # scale=1), matching the holoscan-framework split (a dedicated
        # ``self.normalization = 1.0`` for the engine vs a separate
        # ``vit_normalization`` for the model) and the offline_engine_test.py
        # reference. ``normalization``/``scale`` are pure global scalars inside
        # the sqrt, so recover the raw amplitude by rescaling ``diff_amp``
        # (= sqrt((I/normalization)*scale)) rather than re-running the
        # D4/fftshift: sqrt((I/n)*s) * sqrt(n/s) = sqrt(I). A frozen warm-start
        # probe has a fixed scale, so feeding the ViT-normalized amplitude
        # instead collapses |object|.
        engine_amp = diff_amp * np.float32((self.normalization / self.scale) ** 0.5)
        op_output.emit(engine_amp, "diff_amp_engine")
        op_output.emit(indices, "image_indices")
        op_output.emit(dp_orient, "dp_orient_used")
        self._diag_total_ms += (time.perf_counter() - t0) * 1000.0

class PointProcessorOp(Operator):
    def __init__(self, *args, x_direction = -1., y_direction = -1.,
                 swap_xy: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("PointProcessorOp")
        logging.basicConfig(level=logging.INFO)

        # Axis swap on the (x, y) columns of ``positions_um`` only — affects
        # the ViT/mosaic path, not the iterative engine's ``point_info``
        # (fixed convention the engine was tuned for). Set when the
        # orientation auto-detector picks a winning candidate with swap_xy.
        self.swap_xy = bool(swap_xy)

        self.point_info = None
        self.point_info_target = None
        # Per-frame scan positions in microns, post-conversion. Assigned by
        # compose() with shape (nz, 2). Filled by process_point_info as the
        # PandA stream arrives. Read by SaveViTResult to publish positions
        # alongside ViT batches so downstream stitching uses real positions
        # rather than a deterministic raster (matches live_compare_viewer.py,
        # which loaded H5 `points`).
        self.positions_um = None

        self.angle_correction_flag = True
        self.angle = 0

        self.buffer = []
        self.raw_data = np.zeros((2,0),dtype = np.int32)
        self.frame_id_list = np.zeros((0,),dtype = np.int32)

        self.next_pack_frame_number = 0
        self.raw_data_pointer = 0

        self.pos_loaded_num = 0
        self.pos_ready_num = 0

        # Hardcode
        self.min_points = 300
        self.max_points = 20000
        self.x_direction = x_direction
        self.y_direction = y_direction
        # Relative sign factors (iterative_direction * shared_direction, ±1)
        # applied ONLY to the iterative engine's point_info stream. 1.0 =
        # iterative follows the shared x/y_direction (byte-identical default);
        # the ViT positions_um always uses the shared convention. Set from
        # config x/y_direction_iterative by compose().
        self.x_sign_rel = 1.0
        self.y_sign_rel = 1.0
        self.pos_x_base = None
        self.pos_y_base = None
        self.x_range_um = 2.
        self.y_range_um = 2.
        self.x_pixel_m = 5e-9
        self.y_pixel_m = 5e-9
        self.nx_prb = 180
        self.ny_prb = 180
        self.obj_pad = 30
        # Engine object-array dims (StreamingRecon.nx_obj/ny_obj), wired by
        # compose() in iterative/both mode. Used to (a) center the scan in the
        # object array so position undershoot below the first-chunk minimum
        # has real margin instead of obj_pad//2 (= 2 px!) and (b) clamp
        # out-of-bounds crop windows instead of handing the engine negative
        # indices (cudaErrorIllegalAddress in the gather/scatter kernels).
        self.obj_nx_limit = None
        self.obj_ny_limit = None
        self._oob_points = 0
        # Engine point capacity (StreamingRecon.num_points_l). The engine's
        # GPU buffers are sized to live_num_points_max (default 8192), NOT
        # the scan's nz — writes past it are out-of-bounds GPU memory.
        self.target_max_points = None
        self.x_ratio = 0
        self.y_ratio = 0

        self.simulate_positions = False

    def flush(self,param):
        self.buffer = []
        self.raw_data = np.zeros((2,0),dtype = np.int32)
        self.frame_id_list = np.zeros((0,),dtype = np.int32)

        self.next_pack_frame_number = 0
        self.raw_data_pointer = 0

        self.pos_loaded_num = 0
        self.pos_ready_num = 0

        self.pos_x_base = None
        self.pos_y_base = None

        self.x_range_um = np.abs(param[0])
        self.y_range_um = np.abs(param[1])

        self.x_ratio = param[2]
        self.y_ratio = param[3]

        self.min_points = param[4]
        self.angle = param[5]

        self.simulate_positions = param[6]

        if self.simulate_positions: #Generate all positions at flush
            nx = int(param[7])
            ny = int(param[8])
            x_range_sign = param[0]
            y_range_sign = param[1]
            self.pos0_simul = np.tile(np.linspace(0,x_range_sign,\
                nx+1)[:-1],[ny,1]).reshape((int(nx*ny),)) * self.x_direction
            self.pos1_simul = np.tile(np.linspace(0,y_range_sign,\
                ny+1)[:-1],[nx,1]).T.reshape((int(nx*ny),)) * self.y_direction

        
    def setup(self, spec: OperatorSpec):
        spec.input("pointOp_in").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)

        # An option to deal with the ugly hack:
        # spec.input("pointOp_in").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)
        # spec.input("image_indices_in").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)
        
        # spec.multi_port_condition(
        #     kind=ConditionType.MULTI_MESSAGE_AVAILABLE,
        #     port_names=["pointOp_in", "image_indices_in"],
        #     sampling_mode="SumOfAll",
        #     min_sum=1,
        # )

        spec.output("pos_ready_num").condition(ConditionType.NONE)
    
    def search_next_frame_in_buffer(self):
        for ind,data in enumerate(self.buffer):
            if data[0] == self.next_pack_frame_number:
                self.raw_data = np.concatenate((self.raw_data,data[1]),axis=1)
                self.next_pack_frame_number += 1
                self.buffer.pop(ind)
                return True
        return False
    
    def process_point_info(self):

        if (self.pos_loaded_num+1)*self.upsample <= self.raw_data.shape[1]:

            if self.raw_data.shape[1] > self.min_points * self.upsample:
                
                p_total_num = self.raw_data.shape[1]//self.upsample

                if not self.simulate_positions:
                    
                    praw0 = np.reshape(self.raw_data[0,self.pos_loaded_num*self.upsample:p_total_num*self.upsample],
                                    (p_total_num-self.pos_loaded_num,self.upsample))
                    pos0 = np.mean(praw0,axis=1,dtype = np.float64)
                    praw1 = np.reshape(self.raw_data[1,self.pos_loaded_num*self.upsample:p_total_num*self.upsample],
                                    (p_total_num-self.pos_loaded_num,self.upsample))
                    pos1 = np.mean(praw1,axis=1,dtype = np.float64)


                    pos0 = pos0*self.x_ratio*self.x_direction
                    pos1 = pos1*self.y_ratio*self.y_direction

                    if self.angle_correction_flag:
                        # print('rescale x axis...')
                        if np.abs(self.angle) <= 45.:
                            pos0 *= np.abs(np.cos(self.angle*np.pi/180.))
                        else:
                            pos0 *= np.abs(np.sin(self.angle*np.pi/180.))

                        if self.angle <= -45.:
                            pos0 *= -1
                else:
                    pos0 = self.pos0_simul[self.pos_loaded_num:p_total_num]
                    pos1 = self.pos1_simul[self.pos_loaded_num:p_total_num]


                # Iterative-only sign override: the engine's point_info stream
                # may run with flipped axis conventions relative to the shared
                # pos0/pos1 (which always keep the configured x/y_direction and
                # feed positions_um / the ViT mosaic). A trailing ±1 commutes
                # with all the scalar ratio/angle factors above.
                pos0_pts = pos0 if self.x_sign_rel == 1.0 else pos0 * self.x_sign_rel
                pos1_pts = pos1 if self.y_sign_rel == 1.0 else pos1 * self.y_sign_rel

                if self.pos_x_base is None or self.pos_y_base is None:
                    self.pos_x_base, self.pos_y_base = compute_pos_bases(
                        pos0_pts, pos1_pts, self.y_range_um
                    )
                    # The bases anchor the scan at the very edge of the object
                    # array (margin = obj_pad//2 px below the FIRST CHUNK's
                    # minimum). Real scans undershoot that minimum (settling
                    # rows, serpentine turnarounds, encoder jitter), producing
                    # negative window indices. The engine's object array is
                    # allocated much larger than the scan extent — center the
                    # scan in it so the slack becomes symmetric margin.
                    if self.obj_nx_limit:
                        ext_x = np.ceil(abs(self.x_range_um) * 1e-6 / self.x_pixel_m)
                        ext_y = np.ceil(abs(self.y_range_um) * 1e-6 / self.y_pixel_m)
                        mx = max(0, int((self.obj_nx_limit - self.nx_prb - ext_x) // 2))
                        my = max(0, int((self.obj_ny_limit - self.ny_prb - ext_y) // 2))
                        self.pos_x_base -= mx * self.x_pixel_m * 1e6
                        self.pos_y_base -= my * self.y_pixel_m * 1e6
                        logging.info(
                            "point_info: centered scan in object array "
                            "(margin %d px x, %d px y; obj %dx%d)",
                            mx, my, self.obj_nx_limit, self.obj_ny_limit,
                        )

                points0 = np.round((pos0_pts-self.pos_x_base)*1.e-6/self.x_pixel_m)
                points1 = np.round((pos1_pts-self.pos_y_base)*1.e-6/self.y_pixel_m)

                points0 = points0 + self.nx_prb / 2 + self.obj_pad//2
                points1 = points1 + self.ny_prb / 2 + self.obj_pad//2

                # Clamp windows into the engine's object array. A clamped
                # frame reconstructs at a slightly wrong location, but an
                # unclamped one indexes outside the object buffers and kills
                # the whole run with cudaErrorIllegalAddress.
                if self.obj_nx_limit:
                    oob = (
                        (points0 < self.nx_prb // 2)
                        | (points0 > self.obj_nx_limit - self.nx_prb // 2)
                        | (points1 < self.ny_prb // 2)
                        | (points1 > self.obj_ny_limit - self.ny_prb // 2)
                    )
                    n_oob = int(np.count_nonzero(oob))
                    if n_oob:
                        self._oob_points += n_oob
                        logging.warning(
                            "point_info: clamped %d/%d out-of-bounds windows "
                            "(total %d) — x px [%g, %g], y px [%g, %g], "
                            "obj %dx%d",
                            n_oob, len(points0), self._oob_points,
                            float(points0.min()), float(points0.max()),
                            float(points1.min()), float(points1.max()),
                            self.obj_nx_limit, self.obj_ny_limit,
                        )
                    points0 = np.clip(
                        points0, self.nx_prb // 2,
                        self.obj_nx_limit - self.nx_prb // 2,
                    )
                    points1 = np.clip(
                        points1, self.ny_prb // 2,
                        self.obj_ny_limit - self.ny_prb // 2,
                    )

                for i in range(self.pos_loaded_num,p_total_num):
                    index = i-self.pos_loaded_num
                    if i < self.max_points:
                        self.point_info[i,:] = np.array([(int(points0[index] - self.nx_prb//2), int(points0[index] + self.nx_prb//2), \
                                        int(points1[index] - self.ny_prb//2), int(points1[index] + self.ny_prb//2))]\
                                        ,dtype = np.int32)

                # Mirror the freshly-converted per-frame positions into the
                # buffer that downstream consumers (SaveViTResult → tiled
                # writer → synaps-dash mosaic stitcher) read. Stored in
                # microns, NaN where not yet populated.
                if self.positions_um is not None:
                    end = min(p_total_num, self.positions_um.shape[0])
                    take = end - self.pos_loaded_num
                    if take > 0:
                        # hxn_to_vit convention (pos_map=-y,-x): col 0 = slow
                        # axis (INENC3/pos1 = y_range), col 1 = fast axis
                        # (INENC2/pos0 = x_range). swap_xy=True reverts to the
                        # old (fast→col0, slow→col1) assignment.
                        col_x, col_y = (0, 1) if self.swap_xy else (1, 0)
                        self.positions_um[self.pos_loaded_num:end, col_x] = pos0[:take]
                        self.positions_um[self.pos_loaded_num:end, col_y] = pos1[:take]

                self.pos_loaded_num = p_total_num
                
    def send_points_to_recon(self):

        for i in range(self.pos_ready_num,self.frame_id_list.shape[0]):
            # print('loaded', self.pos_loaded_num)
            if self.pos_loaded_num > self.frame_id_list[i]:
                fid = self.frame_id_list[i]
                # point_info_target is the iterative engine's GPU buffer; it's
                # None in vit-only mode (no engine), so skip the device copy.
                # Its capacity is target_max_points (num_points_l), which can
                # be far smaller than the scan — never write past it.
                if (
                    fid < self.max_points
                    and self.point_info_target is not None
                    and (
                        self.target_max_points is None
                        or self.pos_ready_num < self.target_max_points
                    )
                ):
                    self.point_info_target[self.pos_ready_num,:] = cp.array(self.point_info[fid,:],\
                                                                            dtype = np.int32, order='C')
                # sys.stderr.write(f'{self.point_info[fid,:]}'+'\n')
                self.pos_ready_num += 1
            else:
                break


    def compute(self, op_input, op_output, context):

        data = op_input.receive("pointOp_in")

        # Ugly hack
        if isinstance(data,tuple):
        # if data:            # <---- this is the option to deal with the ugly hack
            # received raw panda data
            # sys.stderr.write('Recv pos data frame'+str(data[0])+'\n')
            if data[0] == self.next_pack_frame_number:
                #concat right away
                self.raw_data = np.concatenate((self.raw_data,data[1]),axis=1)
                self.next_pack_frame_number += 1
            else:
                # store in buffer
                self.buffer.append(data)
            
            while self.search_next_frame_in_buffer():
                pass

            self.process_point_info()
        else:
        # data = op_input.receive("image_indices_in")             # <---- this is the option to deal with the ugly hack
        # if data:
            # received frame ids
            self.frame_id_list = np.concatenate((self.frame_id_list,data),axis=0)

        self.send_points_to_recon()
        op_output.emit(self.pos_ready_num,"pos_ready_num")

class ImageSendOp(Operator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.logger = logging.getLogger("ImageSendOp")
        logging.basicConfig(level=logging.INFO)

        self.diff_d_target = None
        self.max_points = 20000
        # Engine point capacity (StreamingRecon.num_points_l); see compute().
        self.target_max_points = None
        self._engine_full_logged = False
        self.frame_ready_num = 0
        # Iterative-only absolute D4 orientation (config dp_orient_iterative).
        # None = disabled: the engine receives diff_amp exactly as produced
        # with the shared dp_orient. Must NOT default to the config dp_orient —
        # the orientation auto-detect changes dp_orient at runtime and the
        # engine should follow it unless an explicit override was requested.
        self.dp_orient_iterative = None
        self._warned_nonsquare = False

    def flush(self,param):
        self.frame_ready_num = 0


    def setup(self, spec: OperatorSpec):
        spec.input("diff_amp").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)
        spec.input("image_indices").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)
        # Per-batch dp_orient stamp from ImagePreprocessorOp; always received
        # (to drain the queue), only used when dp_orient_iterative is set.
        spec.input("dp_orient_used").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)
        spec.output("frame_ready_num").condition(ConditionType.NONE)
        spec.output("image_indices_out").condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):

        diff_d = op_input.receive("diff_amp")
        indices = op_input.receive("image_indices")
        dp_orient_used = op_input.receive("dp_orient_used")

        nframe = diff_d.shape[0]

        # diff_d_target is the iterative engine's GPU buffer; None in vit-only
        # mode (no engine), so skip the device copy. Its capacity is
        # target_max_points (= StreamingRecon.num_points_l, default 8192) —
        # this can be far SMALLER than max_points (= the scan's nz), and
        # writing past it is an out-of-bounds GPU write
        # (cudaErrorIllegalAddress at frame num_points_l + 1).
        _engine_capacity = self.target_max_points or self.max_points
        if self.diff_d_target is not None and (self.frame_ready_num + nframe) > _engine_capacity \
                and not self._engine_full_logged:
            self.logger.warning(
                "Engine point buffer full (%d points) — further frames bypass "
                "the iterative engine (ViT/tap unaffected). Raise config "
                "live_num_points_max (GPU memory permitting) to cover more "
                "of the scan.", _engine_capacity,
            )
            self._engine_full_logged = True
        if self.diff_d_target is not None and (self.frame_ready_num + nframe) <= _engine_capacity:
            # Iterative-only re-orientation: compose the inverse of the D4 this
            # batch was preprocessed with and the configured target. For even
            # square frames this equals having run preprocess_diffraction with
            # dp_orient_iterative directly (D4 commutes with fftshift there).
            # Rebinds only — diff_d is the same object the ViT branch consumes,
            # so it must never be mutated in place. ascontiguousarray because
            # the memcpy below reads raw ctypes memory.
            if (
                self.dp_orient_iterative is not None
                and dp_orient_used != self.dp_orient_iterative
            ):
                reoriented = reorient_d4(
                    diff_d, dp_orient_used, self.dp_orient_iterative
                )
                if reoriented.shape == diff_d.shape:
                    diff_d = np.ascontiguousarray(reoriented)
                elif not self._warned_nonsquare:
                    self.logger.warning(
                        "dp_orient_iterative %r relative to %r would change the "
                        "frame shape %s (non-square frames + transpose-family "
                        "relative transform); skipping re-orientation.",
                        self.dp_orient_iterative, dp_orient_used,
                        diff_d.shape[-2:],
                    )
                    self._warned_nonsquare = True
            # DC-convention bridge: the shared diff_amp carries the natural
            # diffraction-pattern layout (beam/DC at the CENTER — the ViT
            # convention and what the dashboard shows). The engine's kernels
            # do plain unshifted FFTs and compare amplitudes POINTWISE
            # (dm_update_amp1: same flat index into fft output and diff), so
            # its copy must be fftshifted to DC-at-corner — the same shift
            # HXN_databroker applies when building diffamp for this engine,
            # and the old holoscan-framework feed applied before it. It was
            # lost in the ptychoml preprocess migration (confirmed with the
            # beamline scientists). fftshift == ifftshift for even nx/ny.
            # Rebind, never mutate in place — the ViT branch consumes the
            # same diff_d object.
            diff_d = np.ascontiguousarray(
                np.fft.fftshift(diff_d, axes=(-2, -1))
            )
            # Debug dump: the first 20 DPs exactly as the engine receives
            # them (post-preprocess, post-fftshift, pre-memcpy). Set
            # HOLOPTYCHO_DUMP_ENGINE_INPUTS=<dir> on the server; fires once.
            _dump_dir = os.environ.get("HOLOPTYCHO_DUMP_ENGINE_INPUTS")
            if _dump_dir and not getattr(self, "_feed_dumped", False):
                _path = os.path.join(_dump_dir, "engine_feed_dps.npy")
                np.save(_path, diff_d[:20])
                self.logger.info(
                    "Dumped %d engine-feed DPs to %s", min(20, nframe), _path
                )
                self._feed_dumped = True
            diff_d_target = self.diff_d_target[self.frame_ready_num:self.frame_ready_num+nframe]

            cp.cuda.runtime.memcpy(diff_d_target.data.ptr,diff_d.ctypes.data,diff_d.nbytes,cp.cuda.runtime.memcpyHostToDevice)

        self.frame_ready_num += nframe

        op_output.emit(indices,"image_indices_out")
        op_output.emit(self.frame_ready_num,"frame_ready_num")
