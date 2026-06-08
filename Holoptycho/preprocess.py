import logging
import sys
import threading
import time

import numpy as np
import cupy as cp

_cuda_thread_local = threading.local()

from holoscan.core import Operator, OperatorSpec, ConditionType, IOSpec
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler, EventBasedScheduler
from holoscan.logger import LogLevel, set_log_level
from holoscan.decorator import create_op, Input


def _preprocess_diffraction(images, *, normalization=1.0, scale=1.0,
                             hot_pixel_count_threshold=None, fftshift=True):
    """Convert raw detector intensity to diffraction amplitude.

    Steps (in order):
      1. Hot-pixel zeroing  — counts > hot_pixel_count_threshold → 0  (AI inference;
                              disabled by default so old behaviour is preserved)
      2. sqrt((intensity / normalization) * scale) → float32 amplitude
         normalization / scale are AI-inference knobs; defaults 1.0 / 1.0 reproduce
         the old np.sqrt(images, dtype=float32) exactly.
      3. np.rot90(amplitude, axes=(2,1))  — same as old hardcoded rot90; shared by
         both iterative reconstruction and AI inference.
      4. fftshift on last two axes        — same as old hardcoded fftshift; shared.

    With all defaults the output is numerically equivalent to the previous chain:
        np.rot90(images, axes=(2,1)) → fftshift(axes=(1,2)) → sqrt(float32)
    """
    working = np.asarray(images, dtype=np.float64)
    # Step 1: hot-pixel ceiling (AI inference tuning; disabled by default)
    if hot_pixel_count_threshold is not None:
        working[working > float(hot_pixel_count_threshold)] = 0.0
    # Step 2: amplitude
    amplitude = np.sqrt(
        (working / float(normalization)) * float(scale)
    ).astype(np.float32)
    # Step 3: rotation — rot90_cw == old np.rot90(images, axes=(2,1))
    amplitude = np.ascontiguousarray(np.rot90(amplitude, axes=(2, 1)))
    # Step 4: fftshift
    if fftshift:
        amplitude = np.fft.fftshift(amplitude, axes=(-2, -1))
    return amplitude


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

        # Per-second throughput counters for diagnostic logging
        self._diag_window_start = time.time()
        self._diag_calls = 0
        self._diag_batches_emitted = 0
 
    def flush(self,param):
        self.counter = 0
        self.roi = np.array(param)
        
    def setup(self, spec: OperatorSpec):
        spec.input("flush",policy=IOSpec.QueuePolicy.POP).condition(ConditionType.NONE)
        spec.input("image").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=256)
        spec.input("image_index").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=256)
        spec.output("image_batch")
        spec.output("image_indices")
        
    def compute(self, op_input, op_output, context):
        # Per-second diagnostic logging
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

        param = op_input.receive('flush')
        if param:
            self.flush(param)

        image = op_input.receive("image")
        image_index = op_input.receive("image_index")

        if self.roi is None:
            return

        # For Eiger2 detector
        if self.flip_image:
            image = np.flip(image, 1)
            # After flipping along axis=1 the ROI column indices (measured on
            # the un-flipped raw frame) must be mirrored so the crop picks the
            # same physical region in the now-flipped frame.
            W = image.shape[1]
            r0, r1         = int(self.roi[0, 0]), int(self.roi[0, 1])
            c0_raw, c1_raw = int(self.roi[1, 0]), int(self.roi[1, 1])
            image = image[r0:r1, W - c1_raw : W - c0_raw]
        else:
            image = image[self.roi[0, 0]:self.roi[0, 1],
                          self.roi[1, 0]:self.roi[1, 1]]

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
        self.detmap_threshold = 0
        self.badpixels = None

        # --- AI inference amplitude scaling ---
        # normalization: per-scan max intensity (hot pixels excluded).
        #   Default 1.0 → diff_amp = sqrt(images), identical to old behaviour.
        #   Set to the actual per-scan max to enable ViT-style normalisation.
        #   Does NOT affect iterative reconstruction (relative amplitudes only).
        self.normalization = 1.0
        # scale: global multiplicative factor after sqrt.
        #   Default 1.0 → no change. ptycho-vit training uses 1e4; set to match
        #   whatever value the ViT engine was trained with.
        self.scale = 1.0
        # hot_pixel_count_threshold: zero photon counts above this before sqrt.
        #   Default None = disabled. Complements detmap_threshold (low-end floor).
        #   Affects both iterative and AI paths via the shared diff_amp output.
        self.hot_pixel_count_threshold = None

        # --- Shared orientation / DC convention ---
        # fftshift_dp: apply fftshift to diff_amp (shared by both paths).
        #   True = always apply, matching the old hardcoded fftshift.
        #   PtychoViTInferenceOp's data_is_shifted=True undoes this for ViT.
        #   Keep True unless deliberately changing the DC convention.
        self.fftshift_dp = True

        # --- ViT normalization (computed online from first N frames) ---
        # vit_normalization: per-scan max intensity used to scale diff_amp_vit.
        #   None until computed. First vit_norm_frames raw frames are accumulated;
        #   after that the max (excluding hot pixels) is set here and held for
        #   the rest of the scan.  The first few batches use 1.0 as a placeholder.
        self.vit_normalization = None
        self.vit_normalization_guess = 1000
        self._vit_norm_buffer = []          # raw frame accumulator (pre-transform)
        self._vit_norm_seen = 0             # total frames accumulated so far
        self._vit_norm_frames = 5         # compute once this many frames seen
        self._vit_norm_hot_threshold = 50000.0  # exclude pixels above this count

        # Per-second timing diagnostics
        self._diag_window_start = time.time()
        self._diag_calls = 0
        self._diag_total_ms = 0.0
        
    def setup(self, spec: OperatorSpec):
        spec.input("image_batch").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)
        spec.input("image_indices_in").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)
        spec.output("diff_amp")
        spec.output("image_indices")
        # ViT-specific amplitude: scale=10000, fftshift=False (AI inference path).
        # ConditionType.NONE so the operator is not blocked when this port is
        # not connected (iterative-only mode).
        spec.output("diff_amp_vit").condition(ConditionType.NONE)
        
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

        images  = op_input.receive("image_batch")
        indices = op_input.receive("image_indices_in")

        # Reset vit_normalization at the start of each new scan so it is
        # recomputed from fresh data.  Detecting scan boundary by frame index
        # resetting to 0 is robust without requiring a flush port.
        if int(indices[0]) == 0:
            self.vit_normalization = None
            self._vit_norm_buffer.clear()
            self._vit_norm_seen = 0

        processed_images = np.asarray(images)

        # --- Bad pixel inpainting (BOTH paths) ---
        # Border-clamped: fixes silent empty-window bug when a bad pixel sits
        # at row 0 or col 0 (negative slice index in old code → empty window).
        # Functionally identical to old loop for all interior pixels.
        if self.badpixels is not None and self.badpixels.ndim == 2 and self.badpixels.shape[1] > 0:
            h, w = processed_images.shape[-2], processed_images.shape[-1]
            for bd in self.badpixels.T:
                r, c = int(bd[0]), int(bd[1])
                r0, r1 = max(r - 1, 0), min(r + 2, h)
                c0, c1 = max(c - 1, 0), min(c + 2, w)
                processed_images[:, r, c] = np.median(
                    processed_images[:, r0:r1, c0:c1], axis=(2, 1)
                )

        # --- Noise floor threshold (BOTH paths; unchanged from old code) ---
        if self.detmap_threshold > 0:
            processed_images = processed_images.copy()
            processed_images[processed_images < self.detmap_threshold] = 0

        # --- Reconstruction + model branch → diff_amp (BOTH paths) ---
        # With all defaults this is numerically equivalent to the old chain:
        #   np.rot90(images, axes=(2,1)) → fftshift(axes=(1,2)) → sqrt(float32)
        # To enable ViT-style normalisation set self.normalization to the
        # per-scan max intensity; self.scale to the training scale (e.g. 1e4).
        # The iterative reconstruction is insensitive to constant amplitude
        # scaling, so adjusting these does not affect iterative results.
        diff_amp = _preprocess_diffraction(
            processed_images,
            normalization=self.normalization,
            scale=self.scale,
            hot_pixel_count_threshold=self.hot_pixel_count_threshold,
            fftshift=self.fftshift_dp,
        )

        op_output.emit(diff_amp, "diff_amp")
        op_output.emit(indices, "image_indices")

        # --- ViT branch → diff_amp_vit (AI inference path only) ---
        # Online vit_normalization: accumulate raw frames until vit_norm_frames
        # seen, then compute max excluding hot pixels.  Before normalization is
        # ready the first few batches use vit_norm=1.0 as a placeholder — this
        # is a small fraction of a typical scan and acceptable for live display.
        if self.vit_normalization is None:
            self._vit_norm_buffer.append(processed_images.copy())
            self._vit_norm_seen += processed_images.shape[0]
            if self._vit_norm_seen >= self._vit_norm_frames:
                all_raw = np.concatenate(self._vit_norm_buffer, axis=0)
                mask = all_raw < self._vit_norm_hot_threshold
                self.vit_normalization = (
                    float(all_raw[mask].max()) if mask.any() else self.vit_normalization_guess
                )
                self._vit_norm_buffer.clear()
                self.logger.info(
                    "vit_normalization computed: %.1f from %d frames",
                    self.vit_normalization, self._vit_norm_seen,
                )
        vit_norm = self.vit_normalization if self.vit_normalization is not None else self.vit_normalization_guess
        # fftshift=False: DC stays at center (natural detector position after
        # rot90). ptychoml.PtychoViTInference uses fftshift=None to auto-detect
        # DC position, so it will correctly no-op here and the model sees
        # DC at center — matching the 01Holoscan/holoptycho training convention.
        diff_amp_vit = _preprocess_diffraction(
            processed_images,
            normalization=vit_norm,
            scale=10000.0,
            hot_pixel_count_threshold=self.hot_pixel_count_threshold,
            fftshift=False,
        )
        op_output.emit(diff_amp_vit, "diff_amp_vit")

        self._diag_total_ms += (time.perf_counter() - t0) * 1000.0

class PointProcessorOp(Operator):
    def __init__(self, *args, x_direction = -1., y_direction = -1., **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("PointProcessorOp")
        logging.basicConfig(level=logging.INFO)

        self.point_info = None
        self.point_info_target = None

        # positions_um: (nz, 2) float64 array of per-frame scan positions in
        # microns. Allocated externally in ptycho_holo.config_ops() once nz is
        # known. Col 0 = slow axis (pos1/INENC3), col 1 = fast axis (pos0/INENC2).
        # Filled by process_point_info() as PandA data arrives.
        # Read by SaveViTResult to stitch ViT patches at real scan positions.
        # None until allocated; position tracking silently disabled if None.
        # AI inference path only — iterative path uses point_info/point_info_target.
        self.positions_um = None
        # swap_xy: exchange col 0 and col 1 in positions_um for non-standard
        # axis conventions. False = default HXN (slow→col0, fast→col1).
        self.swap_xy = False

        self.angle_correction_flag = True
        self.angle = 0

        self.upsample = 10
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

        # Reset positions_um to NaN for the new scan (AI inference path)
        if self.positions_um is not None:
            self.positions_um[:] = np.nan

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
        spec.input("flush",policy=IOSpec.QueuePolicy.POP).condition(ConditionType.NONE)
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

                # Populate positions_um with freshly-converted per-frame
                # positions in microns (AI inference path — mosaic stitching).
                # Silent no-op if positions_um has not been allocated.
                # Must run before pos_loaded_num is updated so the slice
                # [pos_loaded_num:end] refers to the newly processed range.
                if self.positions_um is not None:
                    end  = min(p_total_num, self.positions_um.shape[0])
                    take = end - self.pos_loaded_num
                    if take > 0:
                        # HXN convention: col 0 = slow axis (pos1/INENC3),
                        #                 col 1 = fast axis (pos0/INENC2).
                        # swap_xy=True reverses this for non-standard setups.
                        col_fast, col_slow = (0, 1) if self.swap_xy else (1, 0)
                        self.positions_um[self.pos_loaded_num:end, col_fast] = pos0[:take]
                        self.positions_um[self.pos_loaded_num:end, col_slow] = pos1[:take]

                self.pos_loaded_num = p_total_num
                
    def send_points_to_recon(self):

        for i in range(self.pos_ready_num,self.frame_id_list.shape[0]):
            # print('loaded', self.pos_loaded_num)
            if self.pos_loaded_num > self.frame_id_list[i]:
                fid = self.frame_id_list[i]
                if fid < self.max_points:
                    self.point_info_target[self.pos_ready_num,:] = cp.array(self.point_info[fid,:],\
                                                                            dtype = np.int32, order='C')
                # sys.stderr.write(f'{self.point_info[fid,:]}'+'\n')
                self.pos_ready_num += 1
            else:
                break


    def compute(self, op_input, op_output, context):
        if not getattr(_cuda_thread_local, "initialized", False):
            cp.cuda.Device(self.point_info_target.device.id).use()
            _cuda_thread_local.initialized = True

        param = op_input.receive('flush')
        if param:
            self.flush(param)

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
        spec.input("flush",policy=IOSpec.QueuePolicy.POP).condition(ConditionType.NONE)
        spec.input("diff_amp").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)
        spec.input("image_indices").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)
        spec.output("frame_ready_num").condition(ConditionType.NONE)
        spec.output("image_indices_out").condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        if not getattr(_cuda_thread_local, "initialized", False):
            cp.cuda.Device(self.diff_d_target.device.id).use()
            _cuda_thread_local.initialized = True

        param = op_input.receive('flush')
        if param:
            self.flush(param)

        diff_d = op_input.receive("diff_amp")
        indices = op_input.receive("image_indices")

        nframe = diff_d.shape[0]


        if self.diff_d_target is not None and (self.frame_ready_num + nframe) < self.max_points:
            diff_d_target = self.diff_d_target[self.frame_ready_num:self.frame_ready_num+nframe]
            
            cp.cuda.runtime.memcpy(diff_d_target.data.ptr,diff_d.ctypes.data,diff_d.nbytes,cp.cuda.runtime.memcpyHostToDevice)

        self.frame_ready_num += nframe
        
        op_output.emit(indices,"image_indices_out")
        op_output.emit(self.frame_ready_num,"frame_ready_num")
