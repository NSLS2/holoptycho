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

class source1(Operator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.cnt = 0

    def setup(self,spec):
        spec.output("in1")

    def compute(self,op_input,op_output,context):
        op_output.emit(self.cnt,"in1")

@create_op(outputs=("in1"))
def source():
    return 2

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
        self.s = source1(self,name='source')
        self.o = op1(self,name='op1')
        self.e = emit(self,name='emit')
        self.add_flow(self.s,self.o,{("in1","in1")})
        self.add_flow(self.o,self.e,{("out1","out1")})
        
if __name__ == "__main__":
    app = TestApp()
    app.run_async()
    #time.sleep(0.1)
    app.s.cnt += 1
