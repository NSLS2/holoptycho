import logging
from time import sleep

import numpy as np
import cupy as cp


from nsls2ptycho.core.ptycho.recon_ptycho_gui import create_recon_object, deal_with_init_prb
from nsls2ptycho.core.ptycho.utils import parse_config
from nsls2ptycho.core.ptycho_param import Param

from holoscan.core import Application, Operator, OperatorSpec
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler, EventBasedScheduler
from holoscan.logger import LogLevel, set_log_level
from holoscan.decorator import create_op

from pipeline_source import parse_args
from pipeline_preprocess import PreprocAppBase

import nvtx

class ReconOp(Operator):
    def __init__(self, *args, param=None, **kwargs):
        self.logger = logging.getLogger("ReconOp")
        logging.basicConfig(level=logging.INFO)
        super().__init__(*args, **kwargs)
        self.param = param
        self.recon = create_recon_object(param)
        
    def setup(self, spec):
        spec.input("batch")
        spec.output("result")
    
    def compute(self, op_input, op_output, context):
        tensor = op_input.receive("batch")
        diff_d_to_add = cp.asarray(tensor["diff_amp"])
        points_to_add = cp.asarray(tensor["points"])
        
        if np.all(self.recon.prb == 0): # if probe is trivial it means that it was not fully initialized
            with nvtx.annotate("deal_with_init_prb", color="red"):
                prb_init = deal_with_init_prb(self.recon, self.param, diff_d_to_add)
                self.recon.prb += cp.asnumpy(prb_init.astype(self.recon.complex_precision))
                self.recon.prb_d[:, :] += prb_init.astype(self.recon.complex_precision)

        with nvtx.annotate("self.recon.update_arrays", color="green"):
            self.recon.update_arrays(diff_d_to_add, points_to_add * -1)

        self.recon.recon_ptycho_run()
        output = self.recon.fetch_obj_ave()

        op_output.emit(output, "result")


class ReconResultOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("ReconResultOp")
        logging.basicConfig(level=logging.INFO)
        self.batched_result = None
        
    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")
        
    def compute(self, op_input, op_output, context):
        obj = op_input.receive("in")
        if self.batched_result is None:
            self.batched_result = obj.copy()
        else:
            # this implies that at most two batches overlap
            self.batched_result = np.nanmean(
                np.array([self.batched_result, obj.copy()]), axis=0)
        out = np.nan_to_num(self.batched_result, nan=0.0)
        op_output.emit(out, "out")


@create_op(inputs="result")
def sink_func(result):
    print(f"SinkOp received the result from reconstruction {np.sum(result>0)=}")


class PtychoAppBase(PreprocAppBase):
    def compose(self):
        recon_param = parse_config(self.kwargs("recon_op")["ptycho_config_path"], Param())
        recon_param.working_directory = self.kwargs("recon_op")["working_directory"]
        
        eiger_zmq_rx, pos_rx, gather_op = super().compose()
        
        num_parallel_streams = self.kwargs("gather_op")["num_parallel_streams"]
        # Create N reconstruction operators
        recon_ops = []
        for i in range(1, num_parallel_streams + 1):
            recon = ReconOp(self, 
                           param=recon_param,
                           name=f"recon{i}")
            recon_ops.append(recon)
            # Connect gather_op output to this recon operator
            self.add_flow(gather_op, recon, {(f"batch{i}", "batch")})
        
        # Create single batch stacker for combining all results
        recon_result_stacker = ReconResultOp(self, name="recon_result_stacker")
        
        # Connect all recon operators to the same batch stacker
        for recon in recon_ops:
            self.add_flow(recon, recon_result_stacker, {("result", "in")})
        
        return eiger_zmq_rx, pos_rx, recon_result_stacker
        

class PtychoApp(PtychoAppBase):
    def compose(self):
        _, _, recon_result_stacker = super().compose()
        sink = sink_func(self, name="sink")
        self.add_flow(recon_result_stacker, sink)


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
    
    
