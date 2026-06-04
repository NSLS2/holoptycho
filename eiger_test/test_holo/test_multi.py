import numpy as np
import time

from holoscan.core import Application, Operator, OperatorSpec, Tracker, IOSpec
from holoscan.decorator import create_op
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler, EventBasedScheduler

class source1(Operator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.cnt = 0

    def setup(self,spec):
        spec.output("out1")
        spec.output("out2")
        spec.output("out3")

    def compute(self,op_input,op_output,context):
        if self.cnt<3:
            op_output.emit(np.random.rand(),f"out{self.cnt+1}")
            self.cnt += 1
        else:
            pass
        time.sleep(1)

class op1(Operator):
    def __init__(self,*args, oid=0, **kwargs):
        super().__init__(*args,**kwargs)
        self.oid = int(oid)

    def setup(self,spec):
        spec.input("in")
        spec.output("out")

    def compute(self,op_input,op_output,context):
        d = op_input.receive("in")
        print(d)
        print(f"op_start {self.oid+1}")
        time.sleep(3-self.oid)
        print(f"op_end {self.oid+1}")
        op_output.emit(d, "out")

class out(Operator):
    def setup(self,spec):
        spec.input("in").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=3)

    def compute(self,op_input,op_output,context):
        d = op_input.receive("in")
        print(f'out {d}')

class TestApp(Application):
    def compose(self):
        s = source1(self,name='source')
        os=[]
        for i in range(3):
            o = op1(self,oid=i, name=f'op{i}')
            os.append(o)
        e = out(self,name='output')
        for i in range(3):
            self.add_flow(s,os[i],{(f"out{i+1}","in")})
            self.add_flow(os[i],e,{("out","in")})
        
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
