import logging
import sys
import time

import numpy as np
import cupy as cp

from holoscan.core import Operator, OperatorSpec, ConditionType, IOSpec
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler, EventBasedScheduler
from holoscan.logger import LogLevel, set_log_level
from holoscan.decorator import create_op, Input

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
        spec.input("image").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=256)
        spec.input("image_index").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=256)
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

        # For Eiger2 detector
        if self.flip_image:
            image = np.flip(image,1)


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
        # self.roi = np.array(roi)
        self.detmap_threshold = 0
        self.badpixels = None
        super().__init__(*args, **kwargs)

        # Per-second compute() throughput counters. See note in EigerZmqRxOp.
        self._diag_window_start = time.time()
        self._diag_calls = 0
        self._diag_total_ms = 0.0

    def setup(self, spec: OperatorSpec):
        spec.input("image_batch").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)
        spec.input("image_indices_in").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)
        spec.output("diff_amp")
        spec.output("image_indices")

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
        
        for bd in self.badpixels.T:
            x = int(bd[0])
            y = int(bd[1])
            processed_images[:, x, y] = np.median(processed_images[:, x-1:x+2, y-1:y+2], axis=(2, 1))
        
        # processed_images = processed_images[:, self.roi[0,0]:self.roi[0,1], self.roi[1,0]:self.roi[1,1]]
        processed_images = np.rot90(processed_images, axes=(2,1))
        processed_images = np.fft.fftshift(processed_images, axes=(1,2))
        # processed_images = np.transpose(processed_images,[0,2,1])
        if self.detmap_threshold > 0:
            processed_images[processed_images<self.detmap_threshold] = 0
        diff_amp = np.sqrt(processed_images, dtype = np.float32 ,order='C')

        op_output.emit(diff_amp, "diff_amp")
        op_output.emit(indices, "image_indices")
        self._diag_total_ms += (time.perf_counter() - t0) * 1000.0

class PointProcessorOp(Operator):
    def __init__(self, *args, x_direction = -1., y_direction = -1., **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("PointProcessorOp")
        logging.basicConfig(level=logging.INFO)

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
                        self.positions_um[self.pos_loaded_num:end, 0] = pos0[:take]
                        self.positions_um[self.pos_loaded_num:end, 1] = pos1[:take]

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


        if (self.frame_ready_num + nframe) < self.max_points:
            diff_d_target = self.diff_d_target[self.frame_ready_num:self.frame_ready_num+nframe]
            
            cp.cuda.runtime.memcpy(diff_d_target.data.ptr,diff_d.ctypes.data,diff_d.nbytes,cp.cuda.runtime.memcpyHostToDevice)

        self.frame_ready_num += nframe
        
        op_output.emit(indices,"image_indices_out")
        op_output.emit(self.frame_ready_num,"frame_ready_num")
