import logging
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

from holoscan.core import Operator, OperatorSpec, ConditionType, IOSpec
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler, EventBasedScheduler
from holoscan.logger import LogLevel, set_log_level
from holoscan.decorator import create_op, Input


def crop_flipped_roi(image, roi):
    """Crop a horizontally-flipped Eiger2 frame, mirroring the column ROI.

    ``roi[1]`` stores the column window in raw-frame (pre-flip) coordinates,
    measured from the left. ``np.flip(image, 1)`` reverses the columns, so
    the raw col start/stop must be mirrored to positions measured from the
    right of the flipped frame, using the actual post-flip frame width
    (only known at compute time, not when the ROI is set at compose time).
    Rows (``roi[0]``) are unaffected by the horizontal flip.
    """
    image = np.flip(image, 1)
    W = image.shape[1]
    r0, r1 = int(roi[0, 0]), int(roi[0, 1])
    c0_raw, c1_raw = int(roi[1, 0]), int(roi[1, 1])
    return image[r0:r1, W - c1_raw : W - c0_raw]


class ImageBatchOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.logger = logging.getLogger("ImageBatchOp")
        logging.basicConfig(level=logging.INFO)
        self.counter = 0

        self.flip_image = False
        self.batchsize = 0
        self.nx_prb = 0
        self.ny_prb = 0
        self.images_to_add = None #np.zeros((self.batchsize, 256, 256))
        self.indices_to_add = None #np.zeros(self.batchsize, dtype=np.int32)

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

        # For Eiger2 detector: horizontal flip mirrors the column ROI.
        if self.flip_image:
            image = crop_flipped_roi(image, self.roi)
        else:
            image = crop_to_roi(image, self.roi)

        # Remove Bad pixels (-1 to unsigned int)
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
            
class ImagePreprocessorOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.logger = logging.getLogger("ImagePreprocessorOp")
        logging.basicConfig(level=logging.INFO)
        # self.roi = np.array(roi)
        self.detmap_threshold = 0
        self.badpixels = None
        # Once-per-run auto-centering. Computed on the average of the first
        # batch via scipy connected-component segmentation; same shift then
        # applied to every subsequent batch. ``None`` = not yet computed,
        # ``False`` = disabled by config. See ``_compute_centering_shift``.
        # Disabled by default: this shift is NOT part of ptychoml's preprocessing
        # pipeline and moves the diffraction beam away from the position the model
        # was trained on. Enable only if the detector ROI is uncalibrated and the
        # beam is significantly off-centre.
        self.auto_center = False
        self._center_shift: tuple[int, int] | None = None
        # Geometry knobs for the two output branches.
        #   tap_orient: D4 element applied to the intensity tap (saved /dp).
        #               Default 'antitranspose' matches the historical HXN
        #               anti-diagonal flip applied to bring data into the
        #               beamline operator's view; the saved DP looks how the
        #               operator expects.
        #   dp_orient:  D4 element on the model-input branch. Default
        #               'rot90_cw' reproduces the old chain
        #               (antidiag flip ∘ rot90 ∘ transpose) for backwards
        #               compatibility; orientation auto-detect will overwrite
        #               this once it's wired up.
        #   fftshift_dp: model-input DC-convention control. ``None``
        #               (default) lets ptychoml auto-detect via
        #               ``detect_dc_at_corner`` and shift only when the
        #               central beam is at the corners. Override with
        #               ``True``/``False`` from the scan config if a
        #               specific dataset misbehaves; otherwise leave alone.
        # All three are settable from the scan config; see ptycho_holo.py.
        self.tap_orient = 'antitranspose'
        self.dp_orient = 'rot90_cw'
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

    def _compute_centering_shift(self, batch: np.ndarray) -> tuple[int, int]:
        """Segmentation-based one-shot centering offset.

        Averages the input batch (typically 64 frames; protects against the
        odd empty/saturated first frame), masks hot pixels at detector
        saturation, thresholds at 5% of peak to isolate the diffraction
        blob, and runs ``scipy.ndimage.label`` to find connected components.
        The centroid of the largest one is taken as the "scan center"; the
        returned shift translates that centroid to the canvas centre.

        Falls back to ``(0, 0)`` if no object meets the threshold (e.g.
        truly empty first batch) — subsequent batches stay un-shifted in
        that case, matching the un-centered behaviour we had before.
        """
        from scipy.ndimage import label, center_of_mass

        avg = batch.astype(np.float32).mean(axis=0)
        # Saturation mask: pixels at/near uint max are hot/bad and would
        # bias the centroid. Drop them out of the segmentation input.
        try:
            sat = float(np.iinfo(batch.dtype).max) - 1.0
        except ValueError:
            sat = float('inf')  # floats — no saturation concept
        masked = np.where(avg >= sat, 0.0, avg)
        peak = float(masked.max())
        if peak <= 0:
            self.logger.warning(
                "Auto-centering: average frame has no positive signal; "
                "skipping shift",
            )
            return (0, 0)
        binary = masked > (0.05 * peak)
        labels, n_obj = label(binary)
        if n_obj == 0:
            self.logger.warning(
                "Auto-centering: no connected component above 5%% of peak; "
                "skipping shift",
            )
            return (0, 0)
        # "Find 1 object": largest connected component by pixel count.
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        largest = int(np.argmax(sizes))
        cy, cx = center_of_mass(masked, labels, largest)
        h, w = batch.shape[1], batch.shape[2]
        dy = int(round(h / 2.0 - cy))
        dx = int(round(w / 2.0 - cx))
        self.logger.info(
            "Auto-centering: %d component(s) found, largest=%d px at "
            "(y=%.1f, x=%.1f); shifting batch by (dy=%d, dx=%d)",
            n_obj, int(sizes[largest]), cy, cx, dy, dx,
        )
        return (dy, dx)

    @staticmethod
    def _apply_shift(batch: np.ndarray, dy: int, dx: int) -> np.ndarray:
        """Translate every frame in ``batch`` by ``(dy, dx)``; zero-fill the
        wrap-around band (so we don't bring in random distant intensity).
        ``np.roll`` is vectorised over the batch dim — much faster than
        per-frame ``scipy.ndimage.shift``."""
        if dy == 0 and dx == 0:
            return batch
        shifted = np.roll(batch, shift=(dy, dx), axis=(1, 2))
        if dy > 0:
            shifted[:, :dy, :] = 0
        elif dy < 0:
            shifted[:, dy:, :] = 0
        if dx > 0:
            shifted[:, :, :dx] = 0
        elif dx < 0:
            shifted[:, :, dx:] = 0
        return shifted

    def setup(self, spec: OperatorSpec):
        spec.input("image_batch").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)
        spec.input("image_indices_in").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)
        spec.output("diff_amp")
        spec.output("image_indices")
        # Detector-frame intensity tap, captured before rot90/fftshift/floor/sqrt.
        # Consumed only by FrameWriterOp when fine_tune writes are enabled.
        spec.output("intensity")

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

        # One-shot segmentation-based centering: compute a (dy, dx) shift
        # from the first batch's averaged frame, then apply that same shift
        # to every subsequent batch. Affects both the intensity tap (saved
        # to tiled <run>/diffraction/dp) and the model input, so what the
        # operator sees on the dashboard matches what the ViT model sees.
        if self.auto_center and self._center_shift is None:
            self._center_shift = self._compute_centering_shift(processed_images)
        if self._center_shift is not None:
            dy, dx = self._center_shift
            processed_images = self._apply_shift(processed_images, dy, dx)

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
        diff_amp = preprocess_diffraction(
            processed_images,
            normalization=self.normalization,
            scale=self.scale,
            hot_pixel_count_threshold=self.hot_pixel_count_threshold,
            dp_orient=self.dp_orient,
            fftshift=self.fftshift_dp,
        )

        op_output.emit(diff_amp, "diff_amp")
        op_output.emit(indices, "image_indices")
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
        self.pos_x_base = None
        self.pos_y_base = None
        self.x_range_um = 2.
        self.y_range_um = 2.
        self.x_pixel_m = 5e-9
        self.y_pixel_m = 5e-9
        self.nx_prb = 180
        self.ny_prb = 180
        self.obj_pad = 30
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

                
                if self.pos_x_base is None:
                    self.pos_x_base = np.min(pos0)

                if self.pos_y_base is None:
                    self.pos_y_base = pos1[0]
                    if pos1[-1]<pos1[0]:
                        self.pos_y_base -= self.y_range_um

                points0 = np.round((pos0-self.pos_x_base)*1.e-6/self.x_pixel_m)
                points1 = np.round((pos1-self.pos_y_base)*1.e-6/self.y_pixel_m)

                points0 = points0 + self.nx_prb / 2 + self.obj_pad//2
                points1 = points1 + self.ny_prb / 2 + self.obj_pad//2

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
                if fid < self.max_points and self.point_info_target is not None:
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
        self.frame_ready_num = 0
    
    def flush(self,param):
        self.frame_ready_num = 0
        
    
    def setup(self, spec: OperatorSpec):
        spec.input("diff_amp").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)
        spec.input("image_indices").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)
        spec.output("frame_ready_num").condition(ConditionType.NONE)
        spec.output("image_indices_out").condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):

        diff_d = op_input.receive("diff_amp")
        indices = op_input.receive("image_indices")

        nframe = diff_d.shape[0]

        # diff_d_target is the iterative engine's GPU buffer; None in vit-only
        # mode (no engine), so skip the device copy.
        if self.diff_d_target is not None and (self.frame_ready_num + nframe) < self.max_points:
            diff_d_target = self.diff_d_target[self.frame_ready_num:self.frame_ready_num+nframe]
            
            cp.cuda.runtime.memcpy(diff_d_target.data.ptr,diff_d.ctypes.data,diff_d.nbytes,cp.cuda.runtime.memcpyHostToDevice)

        self.frame_ready_num += nframe
        
        op_output.emit(indices,"image_indices_out")
        op_output.emit(self.frame_ready_num,"frame_ready_num")
