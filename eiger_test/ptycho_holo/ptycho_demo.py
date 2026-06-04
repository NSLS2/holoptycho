import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from holoscan.core import Application, Operator, OperatorSpec, Tracker, IOSpec
from holoscan.decorator import create_op
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler, EventBasedScheduler

import sys
sys.path.append('/ptycho_gui/')

from nsls2ptycho2.core.ptycho.utils import parse_config
from nsls2ptycho2.core.ptycho.recon_ptycho_gui import recon_thread

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

        param = parse_config('./ptycho_config.txt')
        param.live_recon_flag = True

        self.recon, rank = recon_thread(param)
        self.recon.setup()

    def setup(self,spec):
        spec.input("det_frame")
        spec.input("panda_pos")
        spec.output("output")

    def compute(self,op_input,op_output,context):
        diff_d = op_input.receive("det_frame")
        point_info_d = op_input.receive("panda_pos")

        cp.cuda.runtime.memcpy(self.recon.diff_d.data.ptr,diff_d.ctypes.data,diff_d.nbytes,cp.cuda.runtime.memcpyHostToDevice)
        cp.cuda.runtime.memcpy(self.recon.point_info_d.data.ptr,point_info_d.ctypes.data,point_info_d.nbytes,cp.cuda.runtime.memcpyHostToDevice)

        self.recon.num_points_recon = diff_d.shape[0]
        self.recon.live_update_plan_last()

        for it in range(50):
            self.recon.one_iter(it)

        op_output.emit(self.recon.obj, "output")

@create_op(inputs="output")
def SaveResult(output):
    np.save('output.npy',output)
    return
    

class TestApp(Application):
    def compose(self):
        sdet = source_det(self,name='sdet')
        spos = source_pos(self,name='spos')
        pty = PtychoRecon(self,name='pty')
        o = SaveResult(self,name='out')
        self.add_flow(sdet,pty,{('det_frame','det_frame')})
        self.add_flow(spos,pty,{('panda_pos','panda_pos')})
        self.add_flow(pty,o)
        
if __name__ == "__main__":
    app = TestApp()
    scheduler = MultiThreadScheduler(
            app,
            worker_thread_number = 3,
            max_duration_ms = -1,
            name = "multi",
            )
    app.scheduler(scheduler)
    app.run()
