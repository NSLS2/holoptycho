import logging
from time import sleep

import numpy as np
import cupy as cp


import sys
sys.path.append('/ptycho_gui/')

from nsls2ptycho2.core.ptycho.utils import parse_config
from nsls2ptycho2.core.ptycho.recon_ptycho_gui import recon_thread

# from nsls2ptycho.core.ptycho.recon_ptycho_gui import create_recon_object, deal_with_init_prb
# from nsls2ptycho.core.ptycho.utils import parse_config
# from nsls2ptycho.core.ptycho_param import Param

from holoscan.core import Application, Operator, OperatorSpec
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler, EventBasedScheduler
from holoscan.logger import LogLevel, set_log_level
from holoscan.decorator import create_op

from pipeline_source import parse_args
from pipeline_preprocess import PreprocAppBase

import nvtx

# class ReconOp(Operator):
#     def __init__(self, *args, param=None, **kwargs):
#         self.logger = logging.getLogger("ReconOp")
#         logging.basicConfig(level=logging.INFO)
#         super().__init__(*args, **kwargs)
#         self.param = param
#         self.recon = create_recon_object(param)
        
#     def setup(self, spec):
#         spec.input("batch")
#         spec.output("result")
    
#     def compute(self, op_input, op_output, context):
#         tensor = op_input.receive("batch")
#         diff_d_to_add = cp.asarray(tensor["diff_amp"])
#         points_to_add = cp.asarray(tensor["points"])
        
#         if np.all(self.recon.prb == 0): # if probe is trivial it means that it was not fully initialized
#             with nvtx.annotate("deal_with_init_prb", color="red"):
#                 prb_init = deal_with_init_prb(self.recon, self.param, diff_d_to_add)
#                 self.recon.prb += cp.asnumpy(prb_init.astype(self.recon.complex_precision))
#                 self.recon.prb_d[:, :] += prb_init.astype(self.recon.complex_precision)

#         with nvtx.annotate("self.recon.update_arrays", color="green"):
#             self.recon.update_arrays(diff_d_to_add, points_to_add * -1)

#         self.recon.recon_ptycho_run()
#         output = self.recon.fetch_obj_ave()

#         op_output.emit(output, "result")


# class ReconResultOp(Operator):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.logger = logging.getLogger("ReconResultOp")
#         logging.basicConfig(level=logging.INFO)
#         self.batched_result = None
        
#     def setup(self, spec: OperatorSpec):
#         spec.input("in")
#         spec.output("out")
        
#     def compute(self, op_input, op_output, context):
#         obj = op_input.receive("in")
#         if self.batched_result is None:
#             self.batched_result = obj.copy()
#         else:
#             # this implies that at most two batches overlap
#             self.batched_result = np.nanmean(
#                 np.array([self.batched_result, obj.copy()]), axis=0)
#         out = np.nan_to_num(self.batched_result, nan=0.0)
#         op_output.emit(out, "out")


# @create_op(inputs="result")
# def sink_func(result):
#     print(f"SinkOp received the result from reconstruction {np.sum(result>0)=}")



class source_det(Operator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.done = False

    def setup(self,spec):
        spec.output("det_frame")

    def compute(self,op_input,op_output,context):
        if not self.done:
            diff_d = np.load('diff_d.npy')
            op_output.emit(diff_d,"det_frame")
            self.done = True
        

class source_pos(Operator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.done = False

    def setup(self,spec):
        spec.output("panda_pos")

    def compute(self,op_input,op_output,context):
        if not self.done:
            point_info_d = np.load('point_info_d.npy')
            op_output.emit(point_info_d,"panda_pos")
            self.done = True


class PtychoRecon(Operator):
    def __init__(self,*args, **kwargs):
        super().__init__(*args,**kwargs)

        param = parse_config('./ptycho_holo/ptycho_config.txt')
        param.live_recon_flag = True

        self.recon, rank = recon_thread(param)
        self.recon.setup()


    def setup(self,spec):
        spec.input("batch")
        spec.output("output")

    def compute(self,op_input,op_output,context):
        # diff_d = op_input.receive("det_frame")
        # point_info_d = op_input.receive("panda_pos")

        data = op_input.receive("batch")
        diff_d = cp.asarray(data["diff_amp"],dtype=np.float32)   #500x256x256
        points = cp.asarray(data["points"])     #2x500

        pos0 = points[0]
        pos1 = points[1]
        
        if self.recon.pos_x_base is None:
            self.recon.pos_x_base = np.min(pos0)

        if self.recon.pos_y_base is None:
            self.recon.pos_y_base = pos1[0]
            if pos1[-1]<pos1[0]:
                self.recon.pos_y_base -= self.recon.y_range_um

        points0 = np.round((pos0-self.recon.pos_x_base)*1.e-6/self.recon.x_pixel_m)
        points1 = np.round((pos1-self.recon.pos_y_base)*1.e-6/self.recon.y_pixel_m)

        #self.points[0] = np.floor(self.points[0]*1.e-6/self.x_pixel_m) + (np.random.rand()<(self.points[0]*1.e-6/self.x_pixel_m - np.floor(self.points[0]*1.e-6/self.x_pixel_m)))
        #self.points[1] = np.floor(self.points[1]*1.e-6/self.y_pixel_m) + (np.random.rand()<(self.points[1]*1.e-6/self.y_pixel_m - np.floor(self.points[1]*1.e-6/self.y_pixel_m)))
        
        points0 = points0 + self.recon.nx_prb / 2 + self.recon.obj_pad//2
        points1 = points1 + self.recon.ny_prb / 2 + self.recon.obj_pad//2

        for index in range(points.shape[1]):
            self.recon.point_info_d[index,:] = cp.array([(int(points0[index] - self.recon.nx_prb//2), int(points0[index] + self.recon.nx_prb//2), \
                            int(points1[index] - self.recon.ny_prb//2), int(points1[index] + self.recon.ny_prb//2))]\
                                , dtype=np.int32, order='C')
            

        self.recon.num_points_recon = diff_d.shape[0]

        cp.copyto(self.recon.diff_d[0:self.recon.num_points_recon],diff_d)
        #cp.cuda.runtime.memcpy(self.recon.diff_d[0:self.recon.num_points_recon].data.ptr,diff_d.ctypes.data,diff_d.nbytes,cp.cuda.runtime.memcpyHostToDevice)

        diff_d = np.load('./ptycho_holo/diff_d.npy')
        point_info_d = np.load('./ptycho_holo/point_info_d.npy')
        cp.cuda.runtime.memcpy(self.recon.diff_d.data.ptr,diff_d.ctypes.data,diff_d.nbytes,cp.cuda.runtime.memcpyHostToDevice)
        # cp.cuda.runtime.memcpy(self.recon.point_info_d.data.ptr,point_info_d.ctypes.data,point_info_d.nbytes,cp.cuda.runtime.memcpyHostToDevice)

        self.recon.live_update_plan_last()

        for it in range(20):
            self.recon.one_iter(it)

        op_output.emit(self.recon.obj, "output")

@create_op(inputs="output")
def SaveResult(output):
    np.save('output.npy',output)
    return
    

class PtychoApp(PreprocAppBase):
    def compose(self):
        eiger_zmq_rx, pos_rx, gather_op = super().compose()


        pty = PtychoRecon(self,name='pty')
        o = SaveResult(self,name='out')
        self.add_flow(gather_op, pty, {("batch1", "batch")})
        self.add_flow(pty,o)


        # sdet = source_det(self,name='sdet')
        # spos = source_pos(self,name='spos')
        # pty = PtychoRecon(self,name='pty')
        # self.add_flow(sdet,pty,{('det_frame','det_frame')})
        # self.add_flow(spos,pty,{('panda_pos','panda_pos')})

# class PtychoAppBase(PreprocAppBase):
#     def compose(self):
#         recon_param = parse_config(self.kwargs("recon_op")["ptycho_config_path"], Param())
#         recon_param.working_directory = self.kwargs("recon_op")["working_directory"]
        
#         eiger_zmq_rx, pos_rx, gather_op = super().compose()
        
#         num_parallel_streams = self.kwargs("gather_op")["num_parallel_streams"]
#         # Create N reconstruction operators
#         recon_ops = []
#         for i in range(1, num_parallel_streams + 1):
#             recon = ReconOp(self, 
#                            param=recon_param,
#                            name=f"recon{i}")
#             recon_ops.append(recon)
#             # Connect gather_op output to this recon operator
#             self.add_flow(gather_op, recon, {(f"batch{i}", "batch")})
        
#         # Create single batch stacker for combining all results
#         recon_result_stacker = ReconResultOp(self, name="recon_result_stacker")
        
#         # Connect all recon operators to the same batch stacker
#         for recon in recon_ops:
#             self.add_flow(recon, recon_result_stacker, {("result", "in")})
        
#         return eiger_zmq_rx, pos_rx, recon_result_stacker
        

# class PtychoApp(PtychoAppBase):
#     def compose(self):
#         _, _, recon_result_stacker = super().compose()
#         sink = sink_func(self, name="sink")
#         self.add_flow(recon_result_stacker, sink)


if __name__ == "__main__":# eiger_ip, eiger_port, msg_format, simulate_position_data_stream, position_data_path = parse_source_args()
    config = parse_args()

    app = PtychoApp()
    app.config(config)
    
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
                worker_thread_number=16,
                check_recession_period_ms=0.001,
                stop_on_deadlock=True,
                stop_on_deadlock_timeout=500,
                name="multithread_scheduler",
            )
    app.scheduler(scheduler)
    
    app.run()
    
    
