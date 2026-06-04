import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
print(comm.Get_rank())

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
        self.it = np.array([0],dtype = np.float32)
        self.cnt = 0

    def setup(self,spec):
        spec.output("out1")
        spec.output("out2")
        spec.output("out3")

    def compute(self,op_input,op_output,context):
        if self.cnt<3:
            print(f'source {self.cnt+1}')
            if (MPI.COMM_WORLD.Get_rank() == 0):
                time.sleep(1)
                self.it[0] = np.array(np.random.rand())
            MPI.COMM_WORLD.Bcast(self.it,root=0)
            op_output.emit(float(self.it[0]),f"out{self.cnt+1}")
            self.cnt += 1
        time.sleep(1)
        return

class op1(Operator):
    def __init__(self,*args, oid=0, **kwargs):
        super().__init__(*args,**kwargs)
        self.oid = int(oid)

    def setup(self,spec):
        spec.input("in")
        spec.output("id")
        spec.output("out")

    def compute(self,op_input,op_output,context):
        d = op_input.receive("in")
        print(f"op_start {self.oid}")
        time.sleep(1)
        print(f"op_end {self.oid}")
        op_output.emit(d, "out")
        op_output.emit(self.oid, "id")

class out(Operator):
    def setup(self,spec):
        spec.input("in")
        spec.input("id")
        spec.output("out1")
        spec.output("out2")
        spec.output("out3")

    def compute(self,op_input,op_output,context):
        d = op_input.receive("in")
        oid = op_input.receive("id")
        print(f'out {oid} {d}')
        op_output.emit(d, f"out{oid+1}")


class TestApp(Application):
    def compose(self):
        s = source1(self,name='source')
        os=[]
        for i in range(3):
            o = op1(self,oid=i, name=f'op{i}')
            os.append(o)
        e = out(self,name='emit')
        for i in range(3):
            print('ok')
            self.add_flow(s,os[i],{(f"out{i+1}","in")})
            self.add_flow(os[i],e,{("out","in")})
            self.add_flow(os[i],e,{("id","id")})
            self.add_flow(e,os[i],{(f"out{i+1}","in")})
        
if __name__ == "__main__":
    app = TestApp()
    scheduler = MultiThreadScheduler(
            app,
            worker_thread_number = 3,
            max_duration_ms = -1,
            name = "multi",
            stop_on_deadlock = False
            )
#    scheduler = GreedyScheduler(
#            app,
#            max_duration_ms = -1,
#            name = "greedy",
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

