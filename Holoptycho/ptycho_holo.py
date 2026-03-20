import logging
from time import sleep
import time

import sys
import os

import numpy as np
import cupy as cp
from numba import cuda

try:
    from hxntools.motor_info import motor_table
except ModuleNotFoundError:
    motor_table = None  # OK for simulate mode; live mode requires hxntools


from ..ptycho.utils import parse_config
from ..ptycho.recon_ptycho_gui import recon_thread

# from nsls2ptycho.core.ptycho.recon_ptycho_gui import create_recon_object, deal_with_init_prb
# from nsls2ptycho.core.ptycho.utils import parse_config
# from nsls2ptycho.core.ptycho_param import Param

from holoscan.core import Application, Operator, OperatorSpec, ConditionType, IOSpec
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler, EventBasedScheduler
from holoscan.logger import LogLevel, set_log_level
from holoscan.decorator import create_op

from .datasource import parse_args, EigerZmqRxOp, PositionRxOp, EigerDecompressOp
from .preprocess import ImageBatchOp, ImagePreprocessorOp, PointProcessorOp, ImageSendOp
from .liverecon_utils import parse_scan_header
from .live_simulation import InitSimul
from .vit_inference import PtychoViTInferenceOp, SaveViTResult

class InitRecon(Operator):
    def __init__(self, *args, param, batchsize,min_points,scan_header_file, **kwargs):
        super().__init__(*args,**kwargs)
        self.scan_num = None
        self.scan_header_file = scan_header_file
        p = parse_scan_header(self.scan_header_file)
        if p:
            self.scan_num = p.scan_num

        self.roi_ptyx0 = param.batch_x0
        self.roi_ptyy0 = param.batch_y0
        self.nx = param.nx
        self.ny = param.ny
        self.batchsize = batchsize
        self.min_points = min_points

        self.angle_correction_flag = True

    def setup(self,spec):
        spec.output("flush_pos_rx").condition(ConditionType.NONE)
        spec.output("flush_image_batch").condition(ConditionType.NONE)
        spec.output("flush_image_send").condition(ConditionType.NONE)
        spec.output("flush_pos_proc").condition(ConditionType.NONE)
        spec.output("flush_pty").condition(ConditionType.NONE)

    def compute(self,op_input,op_output,context):
        p = parse_scan_header(self.scan_header_file)

        if p:
            if self.scan_num != p.scan_num:

                print(f"New scan num: {p.scan_num}")

                self.scan_num = p.scan_num
                nz = p.nz - p.nz%self.batchsize

                if self.angle_correction_flag:
                    # print('rescale x axis based on rotation angle...')
                    if np.abs(p.angle) <= 45.:
                        p.x_range *= np.abs(np.cos(p.angle*np.pi/180.))
                    else:
                        p.x_range *= np.abs(np.sin(p.angle*np.pi/180.))

                # New scan
                op_output.emit((motor_table[p.x_motor][2],motor_table[p.y_motor][2]),'flush_pos_rx')
                op_output.emit([[p.det_roiy0 + self.roi_ptyy0, \
                                           p.det_roiy0 + self.roi_ptyy0 + self.ny],\
                                          [p.det_roix0 + self.roi_ptyx0, \
                                           p.det_roix0 + self.roi_ptyx0 + self.nx]],\
                                            'flush_image_batch')
                op_output.emit(True,'flush_image_send')
                op_output.emit((p.x_range,p.y_range,motor_table[p.x_motor][1],motor_table[p.y_motor][1],p.x_num*2,p.angle,p.x_motor == 'ssx',p.x_num,p.y_num),'flush_pos_proc')
                op_output.emit((p.scan_num,p.x_range,p.y_range,np.maximum(p.x_num*2,self.min_points),nz),'flush_pty')
        sleep(0.05)

# class PtychoCtrl(Operator):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args,**kwargs)
#         self.pos_ready_num = 0
#         self.frame_ready_num = 0

#     def setup(self,spec):
#         spec.input("ctrl_input").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)
#         spec.output("ready_num")

#     def compute(self,op_input,op_output,context):
#         data = op_input.receive("ctrl_input")
#         if data:
#             if data[0] == "pos":
#                 # print(f"Recv pos {data[1]}")
#                 self.pos_ready_num = data[1]
            
#             if data[0] == "frame":
#                 # print(f"Recv frame {data[1]}")
#                 self.frame_ready_num = data[1]
#         else:
#             print(f"Recv pos {self.pos_ready_num} frame {self.frame_ready_num}")
#             op_output.emit(np.minimum(self.pos_ready_num,self.frame_ready_num),"ready_num")


class PtychoRecon(Operator):
    def __init__(self, *args, param=None, **kwargs):
        super().__init__(*args,**kwargs)

        self.recon, rank = recon_thread(param)
        self.recon.setup()
        self.recon.keep_obj0()

        self.num_points_min = 300
        self.it = 0
        self.it_last_update = np.inf
        self.it_ends_after = 30
        self.pos_ready_num = 0
        self.frame_ready_num = 0
        self.points_total = 0
        self.timestamp_iter = []
        self.num_points_recv_iter = []
    
    def flush(self,param):

        print(f'flush ptycho recon {str(param[0])}')
        self.it = 0
        self.it_last_update = np.inf
        self.pos_ready_num = 0
        self.frame_ready_num = 0
        self.probe_initialized = False # not self.recon.init_prb_flag

        self.recon.num_points_recon = 0

        self.recon.scan_num = str(param[0])
        self.recon.x_range_um = np.abs(param[1])
        self.recon.y_range_um = np.abs(param[2])

        self.num_points_min = param[3]
        self.points_total = param[4]

        if self.num_points_min < self.recon.gpu_batch_size:
            self.num_points_min = self.recon.gpu_batch_size

        self.timestamp_iter = []
        self.num_points_recv_iter = []

        # nx_obj_new = int(self.recon.nx_prb + np.ceil(self.recon.x_range_um*1e-6/self.recon.x_pixel_m) + self.recon.obj_pad)
        # ny_obj_new = int(self.recon.ny_prb + np.ceil(self.recon.y_range_um*1e-6/self.recon.y_pixel_m) + self.recon.obj_pad)

        self.recon.new_obj()
        print('reload shared memory')

        # self.recon.init_mmap()

    def setup(self,spec):
        spec.input("flush",policy=IOSpec.QueuePolicy.POP).condition(ConditionType.NONE)
        spec.input("pos_ready_num",policy=IOSpec.QueuePolicy.POP).condition(ConditionType.NONE)
        spec.input("frame_ready_num",policy=IOSpec.QueuePolicy.POP).condition(ConditionType.NONE)
        #spec.input("ready_num")
        spec.output("save_live_result").condition(ConditionType.NONE)
        spec.output("output").condition(ConditionType.NONE)

    def compute(self,op_input,op_output,context):

        flush_param = op_input.receive('flush')
        if flush_param:
            self.flush(flush_param)

        # ready_num = op_input.receive("ready_num")

        # self.recon.num_points_recon = int(ready_num)

        pos_ready_num = op_input.receive("pos_ready_num")
        

        if pos_ready_num:
            self.pos_ready_num = int(pos_ready_num)

        frame_ready_num = op_input.receive("frame_ready_num")

        if frame_ready_num:
            self.frame_ready_num = int(frame_ready_num)

        if self.it - self.it_last_update < self.it_ends_after and self.points_total>0:
            print(f"Recv pos {self.pos_ready_num} frame {self.frame_ready_num} / {self.points_total} {self.num_points_min}")

        ready_num = np.minimum(self.pos_ready_num,self.frame_ready_num)

        ready_num = np.minimum(self.recon.num_points_l,ready_num)

        cp.cuda.Device(self.recon.gpu).use()
        cuda.select_device(self.recon.gpu)
        # cp.cuda.set_pinned_memory_allocator()

        if ready_num > self.recon.num_points_recon and self.num_points_min < np.inf:
            if np.ceil(self.recon.x_range_um*1e-6/self.recon.x_pixel_m)*np.ceil(self.recon.y_range_um*1e-6/self.recon.x_pixel_m)/self.points_total > 16:
                self.recon.clear_obj_tail(self.recon.num_points_recon,ready_num)
            self.recon.num_points_recon = ready_num
            if ready_num > np.minimum(self.recon.num_points_l,self.points_total)*0.97:
                self.it_last_update = self.it
        
        if self.recon.num_points_recon > self.num_points_min:
            if not self.probe_initialized:
                self.recon.init_live_prb(self.recon.num_points_recon)
                if self.recon.prb_prop_dist_um != 0:
                    self.recon.propagate_prb()
                self.probe_initialized = True

            self.timestamp_iter.append(time.time())
            self.num_points_recv_iter.append(self.recon.num_points_recon)
            self.recon.one_iter(self.it)

            if self.it % 10 == 0:
                it_mmap = self.it % self.recon.n_iterations
                op_output.emit((self.recon.mmap_prb[it_mmap],self.recon.mmap_obj[it_mmap],self.it,self.recon.scan_num),"save_live_result")

            self.it += 1

        sleep(0.1)


            # #save
            # if self.recon.num_points_recon >= 2500:
            #     print('saving..')
            #     np.save('diff_d.npy',self.recon.diff_d.get())
            #     np.save('point_info_d.npy',self.recon.point_info_d.get())
        
        if self.it - self.it_last_update >= self.it_ends_after and self.num_points_min<np.inf:
            self.num_points_min = np.inf
            op_output.emit((self.recon,self.timestamp_iter,self.num_points_recv_iter),"output")
        sys.stdout.flush()
        sys.stderr.flush()
        
@create_op(inputs="results")
def SaveLiveResult(results):
    try:
        np.save('/data/users/Holoscan/prb_live.npy',results[0])
        np.save('/data/users/Holoscan/obj_live.npy',results[1])
        with open('/data/users/Holoscan/iteration','w') as f:
            f.write('%d\n'%results[2])
        scan_num = results[3] # For future use
    except:
        pass

@create_op(inputs="output")
def SaveResult(output):
    print('Live recon done! Saving results..')
    output[0].save_recon()
    save_dir = output[0].save_recon_flow()
    np.save(save_dir+'/timestamp_iter',np.array(output[1]))
    np.save(save_dir+'/num_points_recv_iter',np.array(output[2]))
    print('Saving results done.')
    return
    
class PtychoSimulApp(Application):
    def __init__(self, *args, config_path=None, **kwargs):
        super().__init__(*args,**kwargs)

        self.config_path = config_path
        self.param = parse_config(self.config_path)
        self.gpu = self.param.gpus[0]

    def config_ops(self,param):

        nx_prb = self.pty.recon.nx_prb
        ny_prb = self.pty.recon.ny_prb
        nz = self.pty.recon.num_points

        self.image_send.diff_d_target = self.pty.recon.diff_d
        self.image_send.max_points = nz

        self.point_proc.point_info = np.zeros((nz,4),dtype = np.int32)
        self.point_proc.point_info_target = self.pty.recon.point_info_d

        self.point_proc.min_points = self.min_points
        self.point_proc.max_points = nz
        self.point_proc.x_direction = self.pty.recon.x_direction
        self.point_proc.y_direction = self.pty.recon.y_direction
        self.point_proc.x_range_um = self.pty.recon.x_range_um
        self.point_proc.y_range_um = self.pty.recon.y_range_um
        self.point_proc.x_pixel_m = self.pty.recon.x_pixel_m
        self.point_proc.y_pixel_m = self.pty.recon.y_pixel_m
        self.point_proc.nx_prb = nx_prb
        self.point_proc.ny_prb = ny_prb
        self.point_proc.obj_pad = self.pty.recon.obj_pad

        self.point_proc.angle_correction_flag = param.angle_correction_flag
        self.init.angle_correction_flag = param.angle_correction_flag

        self.pty.num_points_min = self.min_points



    def compose(self):

        self.param.live_recon_flag = True

        self.batchsize = 64
        self.min_points = 256

        self.flip_image = True #According to detector settings

        self.image_send = ImageSendOp(self, name="image_send")
        self.point_proc = PointProcessorOp(self, x_direction=self.param.x_direction, y_direction=self.param.y_direction, name="point_proc")

        self.init = InitSimul(self,param=self.param,batchsize = self.batchsize, min_points = self.min_points, name='InitSimul')

        # Additional margin to avoid wrapping
        self.param.x_range += 5
        self.param.y_range += 5
        # self.init_recon = InitRecon(self)
        self.pty = PtychoRecon(self,param=self.param,name='pty')
        # self.pty_ctrl = PtychoCtrl(self)
        self.param.x_range -= 5
        self.param.y_range -= 5


        # Temp
        self.o = SaveResult(self,name='out')
        self.live_result = SaveLiveResult(self,name='live_result')

        self.config_ops(self.param)

        # --- PtychoViT inference (parallel to iterative recon) ---
        # Simulate diffamp path: data is unshifted (DC at corners) — no undo needed
        self.vit = PtychoViTInferenceOp(
            self,
            engine_path="/models/ptycho_vit_amp_phase_b64.engine",
            gpu=1,
            data_is_shifted=False,
            name="vit_inference",
        )
        self.vit_save = SaveViTResult(self, name="vit_save")

        self.add_flow(self.init,self.image_send,{("flush_image_send","flush"),("diff_amp","diff_amp"),("image_indices","image_indices")})

        self.add_flow(self.init,self.point_proc,{("pointRx_out","pointOp_in")})
        self.add_flow(self.image_send,self.point_proc,{("image_indices_out","pointOp_in")})

        self.add_flow(self.image_send,self.pty,{("frame_ready_num","frame_ready_num")})
        self.add_flow(self.point_proc,self.pty,{("pos_ready_num","pos_ready_num")})

        self.add_flow(self.init,self.point_proc,{("flush_pos_proc","flush")})
        self.add_flow(self.init,self.pty,{("flush_pty","flush")})

        self.add_flow(self.pty,self.live_result,{("save_live_result","results")})
        self.add_flow(self.pty,self.o,{("output","output")})

        # VIT: branch off InitSimul's diff_amp (fan-out, parallel to image_send)
        self.add_flow(self.init, self.vit, {("diff_amp", "diff_amp"), ("image_indices", "image_indices")})
        self.add_flow(self.vit, self.vit_save, {("vit_result", "results")})


class PtychoApp(Application):
    def __init__(self, *args, config_path=None, **kwargs):
        super().__init__(*args,**kwargs)

        self.config_path = config_path
        self.param = parse_config(self.config_path)
        self.gpu = self.param.gpus[0]

    def config_ops(self,param):

        nx_prb = self.pty.recon.nx_prb
        ny_prb = self.pty.recon.ny_prb
        nz = self.pty.recon.num_points

        self.image_batch.roi = None
        self.image_batch.batchsize = self.batchsize
        self.image_batch.flip_image = self.flip_image
        self.image_batch.nx_prb = nx_prb
        self.image_batch.ny_prb = ny_prb
        self.image_batch.images_to_add = np.zeros((self.batchsize, nx_prb, ny_prb), dtype = np.uint32)
        self.image_batch.indices_to_add = np.zeros(self.batchsize, dtype=np.int32)

        self.image_proc.detmap_threshold = 0
        self.image_proc.badpixels = np.array([])

        self.image_send.diff_d_target = self.pty.recon.diff_d
        self.image_send.max_points = nz

        self.point_proc.point_info = np.zeros((nz,4),dtype = np.int32)
        self.point_proc.point_info_target = self.pty.recon.point_info_d

        self.point_proc.min_points = self.min_points
        self.point_proc.max_points = nz
        self.point_proc.x_direction = self.pty.recon.x_direction
        self.point_proc.y_direction = self.pty.recon.y_direction
        self.point_proc.x_range_um = self.pty.recon.x_range_um
        self.point_proc.y_range_um = self.pty.recon.y_range_um
        self.point_proc.x_pixel_m = self.pty.recon.x_pixel_m
        self.point_proc.y_pixel_m = self.pty.recon.y_pixel_m
        self.point_proc.nx_prb = nx_prb
        self.point_proc.ny_prb = ny_prb
        self.point_proc.obj_pad = self.pty.recon.obj_pad

        self.point_proc.angle_correction_flag = param.angle_correction_flag
        self.init.angle_correction_flag = param.angle_correction_flag

        self.pty.num_points_min = self.min_points



    def compose(self):

        self.param.live_recon_flag = True
        

        self.batchsize = 64
        self.min_points = 256

        self.flip_image = True #According to detector settings

        self.eiger_zmq_rx = EigerZmqRxOp(self,os.environ['SERVER_STREAM_SOURCE'], name="eiger_zmq_rx")
        self.eiger_decompress = EigerDecompressOp(self, name="eiger_decompress")
        self.pos_rx = PositionRxOp(self,endpoint = os.environ['PANDA_STREAM_SOURCE'], ch1 = "/INENC2.VAL.Value", ch2 = "/INENC3.VAL.Value", upsample_factor=10,
                                   name="pos_rx")

        self.image_batch = ImageBatchOp(self, name="image_batch")
        self.image_proc = ImagePreprocessorOp(self, name="image_proc")
        self.image_send = ImageSendOp(self, name="image_send")
        self.point_proc = PointProcessorOp(self, x_direction=self.param.x_direction, y_direction=self.param.y_direction, name="point_proc")

        # self.init_recon = InitRecon(self)
        self.pty = PtychoRecon(self,param=self.param,name='pty')
        # self.pty_ctrl = PtychoCtrl(self)

        self.init = InitRecon(self,param=self.param,batchsize = self.batchsize, min_points = self.min_points,scan_header_file=os.environ.get('SCAN_HEADER_FILE', '/nsls2/data2/hxn/legacy/users/startup_parameters/scan_header.txt'))

        # Temp
        self.o = SaveResult(self,name='out')
        self.live_result = SaveLiveResult(self,name='live_result')

        self.config_ops(self.param)

        # --- PtychoViT inference (parallel to iterative recon) ---
        # Live mode: ImagePreprocessorOp applies fftshift — undo it for model
        self.vit = PtychoViTInferenceOp(
            self,
            engine_path="/models/ptycho_vit_amp_phase_b64.engine",
            gpu=1,
            data_is_shifted=True,
            name="vit_inference",
        )
        self.vit_save = SaveViTResult(self, name="vit_save")

        self.add_flow(self.eiger_zmq_rx,self.eiger_decompress,{("image_index_encoding","image_index_encoding")})
        self.add_flow(self.eiger_decompress,self.image_batch,{("decompressed_image","image"),("image_index","image_index")})
        self.add_flow(self.image_batch,self.image_proc,{("image_batch","image_batch"),("image_indices","image_indices_in")})
        self.add_flow(self.image_proc,self.image_send,{("diff_amp","diff_amp"),("image_indices","image_indices")})
        
        self.add_flow(self.pos_rx,self.point_proc,{("pointRx_out","pointOp_in")})
        self.add_flow(self.image_send,self.point_proc,{("image_indices_out","pointOp_in")})

        self.add_flow(self.image_send,self.pty,{("frame_ready_num","frame_ready_num")})
        self.add_flow(self.point_proc,self.pty,{("pos_ready_num","pos_ready_num")})

        self.add_flow(self.init,self.pos_rx,{("flush_pos_rx","flush")})
        self.add_flow(self.init,self.image_batch,{("flush_image_batch","flush")})
        self.add_flow(self.init,self.image_send,{("flush_image_send","flush")})
        self.add_flow(self.init,self.point_proc,{("flush_pos_proc","flush")})
        self.add_flow(self.init,self.pty,{("flush_pty","flush")})

        # pool1 = self.make_thread_pool("pool1", 1)
        # pool1.add(self.eiger_zmq_rx, True)
    
        # pool2 = self.make_thread_pool("pool2", 7)
        # pool2.add(self.pos_rx, True)
        # pool2.add(self.image_batch, True)
        # pool2.add(self.image_proc, True)
        # pool2.add(self.image_send, True)
        # pool2.add(self.point_proc, True)
        # pool2.add(self.pty, True)
        # pool2.add(self.o, True)
        

        # self.add_flow(self.init_recon,self.pty_ctrl,{("init","ctrl_input")})
        # self.add_flow(self.image_send,self.pty_ctrl,{("frame_ready_num","ctrl_input")})
        # self.add_flow(self.point_proc,self.pty_ctrl,{("pos_ready_num","ctrl_input")})
        # self.add_flow(self.pty_ctrl,self.pty,{("ready_num","ready_num")})
        # self.add_flow(self.pty,self.pty_ctrl,{("ctrl","ctrl_input")})


        self.add_flow(self.pty,self.live_result,{("save_live_result","results")})
        self.add_flow(self.pty,self.o,{("output","output")})

        # VIT: branch off ImagePreprocessorOp's diff_amp (fan-out, parallel to image_send)
        self.add_flow(self.image_proc, self.vit, {("diff_amp", "diff_amp"), ("image_indices", "image_indices")})
        self.add_flow(self.vit, self.vit_save, {("vit_result", "results")})


def main():
    if len(sys.argv) == 1: # started from commmandline
        # raise NotImplementedError("No config file for Holoptycho")
        config_path = '/eiger_dir/ptycho_holo/ptycho_config.txt'
    elif len(sys.argv) >= 2: # started from GUI
        config_path = sys.argv[1]
    #config = parse_args()

    param = parse_config(config_path)
    gpu = param.gpus[0]
    cp.cuda.Device(gpu).use()
    cuda.select_device(gpu)
    cp.cuda.set_pinned_memory_allocator()

    if len(sys.argv) >= 3 and sys.argv[2] == 'simulate':
        app = PtychoSimulApp(config_path=config_path)
    else:
        app = PtychoApp(config_path=config_path)
    
    #app.config()
    
    # scheduler = EventBasedScheduler(
    #             app,
    #             worker_thread_number=16,
    #             stop_on_deadlock=True,
    #             stop_on_deadlock_timeout=500,
    #             name="event_based_scheduler",
    #         )
    # app.scheduler(scheduler)
    
    scheduler = MultiThreadScheduler(
                app,
                worker_thread_number=11,
                check_recession_period_ms=0.001,
                stop_on_deadlock=True,
                stop_on_deadlock_timeout=500,
                # strict_job_thread_pinning=True,
                name="multithread_scheduler",
            )
    app.scheduler(scheduler)

    app.run()
    
    
