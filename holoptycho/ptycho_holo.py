import logging
import ast
import threading
import uuid
from time import sleep
import time

import sys
import os


# Set by runner.stop() to ask the running pipeline to flush, save final
# results, and stop iterating. PtychoRecon.compute() checks this on each tick
# and trips the natural-termination path. Cleared by PtychoRecon.flush() at
# the start of each new run.
_finish_event = threading.Event()

# Set by PtychoRecon.compute() right after fragment.stop_execution() returns,
# meaning the natural-termination path completed (write_final landed, scheduler
# winding down). The subprocess SIGABRT handler reads this to convert a
# Holoscan teardown crash into rc=0.
_work_complete = threading.Event()

import numpy as np
import cupy as cp
from numba import cuda

try:
    from hxntools.motor_info import motor_table
except ModuleNotFoundError:
    motor_table = None  # OK for simulate mode; live mode requires hxntools


# Ptycho is imported purely as a library. The streaming-reconstruction
# state machine lives locally in ``streaming_recon.StreamingPtychoRecon``
# and uses ptycho only for stateless kernel libraries (cupy_util,
# prop_class_asm, cupy_collection, numba_collection). See streaming_recon.py
# for the architectural note.
from types import SimpleNamespace

from ptycho.utils import parse_config as _ptycho_parse_config

from .streaming_recon import StreamingPtychoRecon


def _coerce_config_value(value):
    if not isinstance(value, str):
        return value
    if value == "":
        return value
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _apply_config_overrides(param, config_overrides):
    if not config_overrides:
        return param
    for key, value in config_overrides.items():
        setattr(param, key, _coerce_config_value(value))
    return param


def parse_config(config_path, config_overrides=None):
    """Parse a ptycho config file into a SimpleNamespace.

    Wraps ``ptycho.utils.parse_config`` (which requires a mutable ``param``
    object to populate) by supplying a fresh ``SimpleNamespace`` so callers
    don't need to construct a Param themselves. Holoptycho calls this with
    a single positional argument.
    """
    param = SimpleNamespace(gui=False)
    param = _ptycho_parse_config(config_path, param)
    return _apply_config_overrides(param, config_overrides)

from holoscan.core import Application, Operator, OperatorSpec, ConditionType, IOSpec
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler, EventBasedScheduler
from holoscan.logger import LogLevel, set_log_level
from holoscan.decorator import create_op

from .datasource import parse_args, EigerZmqRxOp, PositionRxOp, EigerDecompressOp
from .preprocess import ImageBatchOp, ImagePreprocessorOp, PointProcessorOp, ImageSendOp
from .liverecon_utils import parse_scan_header
from .live_simulation import InitSimul
from .vit_inference import PtychoViTInferenceOp, SaveViTResult
from .tiled_writer import get_writer

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

        self.param = param

        # The streaming reconstruction engine owns all GPU state that was
        # previously behind ``recon_thread`` / ``ptycho_trans``. It imports
        # ptycho's kernel libraries directly; see streaming_recon.py.
        self.recon = StreamingPtychoRecon(config=param)
        # Allocate GPU buffers once per process. ``live_num_points_max``
        # comes from the config; fall back to a reasonable upper bound if
        # not set.
        num_points_max = int(getattr(param, "live_num_points_max", 0)) or 8192
        self.recon.gpu_setup(num_points_max=num_points_max)

        self.num_points_min = 300
        self.it = 0
        self.it_last_update = np.inf
        self.it_ends_after = 30
        self.n_iterations = int(getattr(param, "n_iterations", 500))
        self.pos_ready_num = 0
        self.frame_ready_num = 0
        self.points_total = 0
        self.timestamp_iter = []
        self.num_points_recv_iter = []

        self._logger = logging.getLogger("PtychoRecon")
        self._iteration_started_logged = False
        self._last_recv_state = (-1, -1)
        self._stop_execution_called = False

    def flush(self,param):

        print(f'flush ptycho recon {str(param[0])}')
        self.it = 0
        self.it_last_update = np.inf
        self.pos_ready_num = 0
        self.frame_ready_num = 0
        self.probe_initialized = False
        self._iteration_started_logged = False
        self._last_recv_state = (-1, -1)
        self._stop_execution_called = False
        _finish_event.clear()
        _work_complete.clear()

        # Reset the engine for a new scan region. This combines what the
        # HXN_development code called ``new_obj()`` + ``flush_live_recon()``:
        # re-dimension the object, zero the accumulation buffers, and
        # random-init the object.
        self.recon.reset_for_scan(
            scan_num=str(self.param.scan_num),
            x_range_um=np.abs(param[1]),
            y_range_um=np.abs(param[2]),
            num_points_max=param[3],
        )

        self.num_points_min = param[3]
        self.points_total = param[4]

        if self.num_points_min < self.recon.gpu_batch_size:
            self.num_points_min = self.recon.gpu_batch_size

        self.timestamp_iter = []
        self.num_points_recv_iter = []
        print('reload shared memory')

    def setup(self,spec):
        spec.input("pos_ready_num",policy=IOSpec.QueuePolicy.POP).condition(ConditionType.NONE)
        spec.input("frame_ready_num",policy=IOSpec.QueuePolicy.POP).condition(ConditionType.NONE)
        #spec.input("ready_num")
        spec.output("save_live_result").condition(ConditionType.NONE)
        spec.output("output").condition(ConditionType.NONE)

    def compute(self,op_input,op_output,context):

        # External stop request: trip the natural-termination path so SaveResult
        # fires on this tick (write_final lands in Tiled), then go quiescent.
        if _finish_event.is_set() and self.num_points_min < np.inf:
            self.n_iterations = self.it
            self._logger.info("Finish requested — flushing and saving final results")
            _finish_event.clear()

        # If termination was emitted on a previous tick, SaveResult has had
        # at least one full tick to run (its message is queued; the scheduler
        # ticks downstream ops between our compute() calls). Now ask Holoscan
        # to exit the run loop so app.run_async() Future resolves and the
        # API status flips to "finished".
        if self.num_points_min == np.inf and not self._stop_execution_called:
            self._logger.info("Calling fragment.stop_execution() to release run_async")
            try:
                self.fragment.stop_execution()
            except Exception:
                self._logger.exception("stop_execution() raised")
            self._stop_execution_called = True
            _work_complete.set()

        pos_ready_num = op_input.receive("pos_ready_num")
        

        if pos_ready_num:
            self.pos_ready_num = int(pos_ready_num)

        frame_ready_num = op_input.receive("frame_ready_num")

        if frame_ready_num:
            self.frame_ready_num = int(frame_ready_num)

        if self.it - self.it_last_update < self.it_ends_after and self.points_total > 0:
            state = (self.pos_ready_num, self.frame_ready_num)
            if state != self._last_recv_state:
                self._logger.info(
                    "Recv pos %d frame %d / %d (threshold %d)",
                    self.pos_ready_num, self.frame_ready_num,
                    self.points_total, self.num_points_min,
                )
                self._last_recv_state = state

        ready_num = np.minimum(self.pos_ready_num,self.frame_ready_num)

        ready_num = np.minimum(self.recon.num_points_l,ready_num)

        cp.cuda.Device(self.recon.gpu).use()
        cuda.select_device(self.recon.gpu)
        # cp.cuda.set_pinned_memory_allocator()

        if ready_num > self.recon.num_points_recon and self.num_points_min < np.inf:
            if np.ceil(self.recon.x_range_um*1e-6/self.recon.x_pixel_m)*np.ceil(self.recon.y_range_um*1e-6/self.recon.x_pixel_m)/self.points_total > 16:
                self.recon.clear_region(self.recon.num_points_recon, ready_num)
            self.recon.num_points_recon = ready_num
            if ready_num > np.minimum(self.recon.num_points_l,self.points_total)*0.97:
                self.it_last_update = self.it

        if self.recon.num_points_recon >= self.num_points_min:
            if not self._iteration_started_logged:
                self._logger.info(
                    "Iterative recon threshold crossed: num_points_recon=%d (threshold=%d)",
                    int(self.recon.num_points_recon),
                    int(self.num_points_min),
                )
                self._iteration_started_logged = True

            if not self.probe_initialized:
                self._logger.info("Initializing probe (num_points=%d)", int(self.recon.num_points_recon))
                self.recon.initial_probe(self.recon.num_points_recon)
                if self.recon.prb_prop_dist_um != 0:
                    self._logger.info("Propagating probe (prop_dist_um=%g)", float(self.recon.prb_prop_dist_um))
                    self.recon.propagate_probe()
                self.probe_initialized = True
                self._logger.info("Probe initialized; entering iteration loop")

            self.timestamp_iter.append(time.time())
            self.num_points_recv_iter.append(self.recon.num_points_recon)
            self.recon.iter_once(self.it)
            self.recon._track_iter(self.it)

            if self.it % 10 == 0:
                prb_snap, obj_snap, it_num, scan_num = self.recon.snapshot()
                op_output.emit((prb_snap, obj_snap, it_num, scan_num), "save_live_result")
                self._logger.info(
                    "Iteration %d (num_points_recon=%d)",
                    self.it, int(self.recon.num_points_recon),
                )

            self.it += 1

        sleep(0.1)


            # #save
            # if self.recon.num_points_recon >= 2500:
            #     print('saving..')
            #     np.save('diff_d.npy',self.recon.diff_d.get())
            #     np.save('point_info_d.npy',self.recon.point_info_d.get())
        
        # Terminate when we've either (a) finished collecting data and run
        # `it_ends_after` more iterations, or (b) hit the configured iteration
        # cap. Both gated on `num_points_min < inf` so we only fire once.
        finished_collecting = (
            self.it - self.it_last_update >= self.it_ends_after
        )
        hit_iteration_cap = self.it >= self.n_iterations
        if (finished_collecting or hit_iteration_cap) and self.num_points_min < np.inf:
            reason = "iteration cap" if hit_iteration_cap else "data collection complete"
            self._logger.info(
                "Iterative recon finishing (%s, it=%d, n_iterations=%d)",
                reason, self.it, self.n_iterations,
            )
            self.num_points_min = np.inf
            op_output.emit((self.recon,self.timestamp_iter,self.num_points_recv_iter),"output")
        sys.stdout.flush()
        sys.stderr.flush()
        
# Module-level writer — initialized once when the pipeline starts.
# Uses TiledWriter if TILED_BASE_URL + TILED_API_KEY are set, otherwise
# falls back to .npy file writes.
# NOTE: The .npy fallback is deprecated and will be removed in a future release.
#       Set TILED_BASE_URL and TILED_API_KEY to use the Tiled backend.
_writer = get_writer()

@create_op(inputs="results")
def SaveLiveResult(results):
    try:
        _writer.write_live(
            iteration=results[2],
            probe=results[0],
            obj=results[1],
        )
    except Exception:
        pass

@create_op(inputs="output")
def SaveResult(output):
    print('Live recon done! Saving results..')
    engine = output[0]
    _writer.write_final(
        probe=np.asarray(engine.prb_mode),
        obj=np.asarray(engine.obj_mode),
        timestamps=np.array(output[1]),
        num_points=np.array(output[2]),
    )
    print('Saving results done.')
    
class PtychoSimulApp(Application):
    def __init__(self, *args, config_path=None, config_overrides=None, engine_path=None, **kwargs):
        super().__init__(*args,**kwargs)

        self.config_path = config_path
        self.param = parse_config(self.config_path, config_overrides=config_overrides)
        self.gpu = self.param.gpus[0]
        self.engine_path = engine_path or "/models/ptycho_vit_amp_phase_b64.engine"

    def config_ops(self,param):

        nx_prb = self.pty.recon.nx_prb
        ny_prb = self.pty.recon.ny_prb
        nz = self.pty.recon.num_points

        self.image_send.diff_d_target = self.pty.recon.diff_d
        self.image_send.max_points = nz

        self.point_proc.point_info = np.zeros((nz,4),dtype = np.int32)
        self.point_proc.point_info_target = self.pty.recon.point_info_d
        self.point_proc.positions_um = np.full((nz, 2), np.nan, dtype=np.float64)

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
        # Prefer a second GPU for PyCUDA/TRT when available, but fall back to
        # the recon GPU on single-GPU nodes instead of hard-failing.
        vit_gpu = self.param.gpus[1] if len(self.param.gpus) > 1 else self.param.gpus[0]
        # Simulate diffamp path: data is unshifted (DC at corners) — no undo needed
        self.vit = PtychoViTInferenceOp(
            self,
            engine_path=self.engine_path,
            gpu=vit_gpu,
            data_is_shifted=False,
            name="vit_inference",
        )
        self.vit_save = SaveViTResult(
            self,
            positions_provider=lambda: self.point_proc.positions_um,
            pixel_size_m=float(self.pty.recon.x_pixel_m),
            x_range_um=float(self.pty.recon.x_range_um),
            y_range_um=float(self.pty.recon.y_range_um),
            name="vit_save",
        )

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
    def __init__(self, *args, config_path=None, config_overrides=None, engine_path=None, **kwargs):
        super().__init__(*args,**kwargs)

        self.config_path = config_path
        self.param = parse_config(self.config_path, config_overrides=config_overrides)
        self.gpu = self.param.gpus[0]
        self.engine_path = engine_path or "/models/ptycho_vit_amp_phase_b64.engine"

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
        # Per-frame scan positions (microns), filled by PointProcessorOp as
        # PandA data arrives. Read by SaveViTResult and published to tiled
        # so the dashboard mosaic stitcher uses real positions.
        self.point_proc.positions_um = np.full((nz, 2), np.nan, dtype=np.float64)

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

        self.pty.num_points_min = self.min_points



    def compose(self):

        self.param.live_recon_flag = True

        # Streaming-pipeline frame batch size. Decoupled from the ViT engine
        # batch dim — PtychoViTInferenceOp chunks each batch into engine-sized
        # sub-batches internally so this value can stay tuned for throughput.
        self.batchsize = 64
        self.min_points = 256

        # Which branches to wire. "iterative" runs only PtychoRecon, "vit" runs
        # only PtychoViTInferenceOp, "both" runs them in parallel (default).
        recon_mode = str(getattr(self.param, "recon_mode", "both")).lower()
        if recon_mode not in ("iterative", "vit", "both"):
            raise ValueError(
                f"recon_mode must be one of iterative/vit/both, got: {recon_mode!r}"
            )
        self.recon_mode = recon_mode

        self.flip_image = True  # According to detector settings

        # Derive scan-size parameters from config.
        # x_num / y_num: number of scan positions along each axis.
        # nz: total expected frames, rounded down to batchsize multiple.
        # num_points_max: minimum points before reconstruction begins.
        x_num = int(self.param.x_num)
        y_num = int(self.param.y_num)
        nz = (x_num * y_num) - (x_num * y_num) % self.batchsize
        num_points_max = max(x_num * 2, self.min_points)

        self.eiger_zmq_rx = EigerZmqRxOp(self, os.environ['SERVER_STREAM_SOURCE'], name="eiger_zmq_rx")
        self.eiger_decompress = EigerDecompressOp(self, name="eiger_decompress")
        self.pos_rx = PositionRxOp(
            self,
            endpoint=os.environ['PANDA_STREAM_SOURCE'],
            ch1=getattr(self.param, 'pos_x_channel', '/INENC2.VAL.Value'),
            ch2=getattr(self.param, 'pos_y_channel', '/INENC3.VAL.Value'),
            upsample_factor=10,
            name="pos_rx",
        )

        self.image_batch = ImageBatchOp(self, name="image_batch")
        self.image_proc = ImagePreprocessorOp(self, name="image_proc")
        self.image_send = ImageSendOp(self, name="image_send")
        self.point_proc = PointProcessorOp(
            self,
            x_direction=self.param.x_direction,
            y_direction=self.param.y_direction,
            name="point_proc",
        )

        self.pty = PtychoRecon(self, param=self.param, name='pty')

        self.o = SaveResult(self, name='out')
        self.live_result = SaveLiveResult(self, name='live_result')

        self.config_ops(self.param)

        # Initialize operators from config (replaces InitRecon scan header flush).
        # ImageBatchOp: set detector crop ROI from config.
        det_roiy0 = int(self.param.det_roiy0)
        det_roix0 = int(self.param.det_roix0)
        self.image_batch.roi = np.array([
            [det_roiy0 + self.param.batch_y0, det_roiy0 + self.param.batch_y0 + self.param.ny],
            [det_roix0 + self.param.batch_x0, det_roix0 + self.param.batch_x0 + self.param.nx],
        ])

        # PointProcessorOp: set scan geometry from config.
        self.point_proc.x_range_um = np.abs(self.param.x_range)
        self.point_proc.y_range_um = np.abs(self.param.y_range)
        self.point_proc.x_ratio = float(self.param.x_ratio)
        self.point_proc.y_ratio = float(self.param.y_ratio)
        self.point_proc.min_points = num_points_max
        self.point_proc.angle = float(getattr(self.param, 'angle', 0.0))
        self.point_proc.simulate_positions = bool(getattr(self.param, 'simulate_positions', False))
        if self.point_proc.simulate_positions:
            self.point_proc.pos0_simul = np.tile(
                np.linspace(0, self.param.x_range, x_num + 1)[:-1], [y_num, 1]
            ).reshape((x_num * y_num,)) * self.point_proc.x_direction
            self.point_proc.pos1_simul = np.tile(
                np.linspace(0, self.param.y_range, y_num + 1)[:-1], [x_num, 1]
            ).T.reshape((x_num * y_num,)) * self.point_proc.y_direction

        # PtychoRecon: reset engine for this scan.
        self.pty.recon.reset_for_scan(
            scan_num=str(self.param.scan_num),
            x_range_um=np.abs(self.param.x_range),
            y_range_um=np.abs(self.param.y_range),
            num_points_max=num_points_max,
        )
        self.pty.num_points_min = num_points_max
        self.pty.points_total = nz
        if self.pty.num_points_min < self.pty.recon.gpu_batch_size:
            self.pty.num_points_min = self.pty.recon.gpu_batch_size
        self.pty.probe_initialized = False

        # Override the positions_um buffer that config_ops sized to
        # `pty.recon.num_points` (the recon engine's max-points cap, default
        # 8192). We want one slot per scan frame so ViT batches beyond the
        # iterative cap still publish positions for the dashboard.
        self.point_proc.positions_um = np.full(
            (x_num * y_num, 2), np.nan, dtype=np.float64
        )

        # Each pipeline run gets a fresh container in Tiled keyed by its own uid.
        # Metadata captures the raw scan being reconstructed plus the scan-grid
        # geometry that downstream consumers (synaps-dash) need to stitch
        # per-frame ViT predictions into a global mosaic. Done after
        # reset_for_scan so x_pixel_m is populated from the engine.
        self.run_uid = uuid.uuid4().hex
        run_metadata = {
            "scan_num": str(self.param.scan_num),
            "raw_uid": str(getattr(self.param, "raw_uid", "") or ""),
            "scan_id": str(
                getattr(self.param, "scan_id", self.param.scan_num)
            ),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "recon_mode": recon_mode,
            "x_pixel_m": float(self.pty.recon.x_pixel_m),
            "y_pixel_m": float(self.pty.recon.y_pixel_m),
            "x_num": int(self.param.x_num),
            "y_num": int(self.param.y_num),
            "x_range_um": float(np.abs(self.param.x_range)),
            "y_range_um": float(np.abs(self.param.y_range)),
            "x_direction": float(self.param.x_direction),
            "y_direction": float(self.param.y_direction),
        }
        _writer.start_run(self.run_uid, metadata=run_metadata)

        # --- PtychoViT inference (parallel to iterative recon) ---
        # Prefer a second GPU for PyCUDA/TRT when available, but fall back to
        # the recon GPU on single-GPU nodes instead of hard-failing.
        vit_gpu = self.param.gpus[1] if len(self.param.gpus) > 1 else self.param.gpus[0]
        # Live mode: ImagePreprocessorOp applies fftshift — undo it for model
        self.vit = PtychoViTInferenceOp(
            self,
            engine_path=self.engine_path,
            gpu=vit_gpu,
            data_is_shifted=True,
            name="vit_inference",
        )
        # SaveViTResult publishes positions_um alongside each batch and
        # accumulates the running phase mosaic at <run>/vit/mosaic. The
        # dashboard renders that array directly — no client-side stitching.
        self.vit_save = SaveViTResult(
            self,
            positions_provider=lambda: self.point_proc.positions_um,
            pixel_size_m=float(self.pty.recon.x_pixel_m),
            x_range_um=float(self.pty.recon.x_range_um),
            y_range_um=float(self.pty.recon.y_range_um),
            name="vit_save",
        )

        self.add_flow(self.eiger_zmq_rx, self.eiger_decompress, {("image_index_encoding", "image_index_encoding")})
        self.add_flow(self.eiger_decompress, self.image_batch, {("decompressed_image", "image"), ("image_index", "image_index")})
        self.add_flow(self.image_batch, self.image_proc, {("image_batch", "image_batch"), ("image_indices", "image_indices_in")})
        self.add_flow(self.image_proc, self.image_send, {("diff_amp", "diff_amp"), ("image_indices", "image_indices")})

        self.add_flow(self.pos_rx, self.point_proc, {("pointRx_out", "pointOp_in")})
        self.add_flow(self.image_send, self.point_proc, {("image_indices_out", "pointOp_in")})

        if self.recon_mode in ("iterative", "both"):
            self.add_flow(self.image_send, self.pty, {("frame_ready_num", "frame_ready_num")})
            self.add_flow(self.point_proc, self.pty, {("pos_ready_num", "pos_ready_num")})
            self.add_flow(self.pty, self.live_result, {("save_live_result", "results")})
            self.add_flow(self.pty, self.o, {("output", "output")})

        if self.recon_mode in ("vit", "both"):
            # ViT: branch off ImagePreprocessorOp's diff_amp (fan-out, parallel to image_send)
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
    
    
