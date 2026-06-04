import logging
import socket
import zmq
from argparse import ArgumentParser

import numpy as np
import numpy.typing as npt
import cupy as cp
import json
import cbor2
import pprint
import traceback
import h5py
import time

from holoscan.core import Application, Operator, OperatorSpec, Tracker
from holoscan.decorator import create_op
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler, EventBasedScheduler

@create_op(outputs=("in1","in2"))
def source():
    return np.random.rand(),np.random.rand()

@create_op(inputs="in1",outputs=("out1"))
def op1(in1):
    print('op_start')
    time.sleep(2)
    print('op_end')
    return in1

@create_op(inputs=("out1"))
def emit(out1):
    print(out1)

class TestApp(Application):
    def compose(self):
        s = source(self,name='source')
        o1 = op1(self,name='op1')
        o2 = op1(self,name='op2')
        e = emit(self,name='emit')
        self.add_flow(s,o1,{("in1","in1")})
        self.add_flow(s,o2,{("in2","in1")})
        self.add_flow(o1,e,{("out1","out1")})
        self.add_flow(o2,e,{("out1","out1")})
        
if __name__ == "__main__":
    app = TestApp()
    scheduler = MultiThreadScheduler(
            app,
            worker_thread_number = 1,
            name = "multi",
            )
#    scheduler = GreedyScheduler(
#            app,
#            max_duration_ms = -1,
#            name = "multi",
#            )
    app.scheduler(scheduler)
    app.run_async()
#    app2 = TestApp()
#    scheduler = MultiThreadScheduler(
#            app2,
#            worker_thread_number = 1,
#            name = "multi",
#            )
#    app2.scheduler(scheduler)
#    app2.run_async()
