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

# Set at the very end of PtychoApp.compose() — after every operator has been
# instantiated, every flow added, and config_ops() called (which is what
# initialises ImageBatchOp.roi etc). The subprocess sentinel watcher reads
# this to flip state.pipeline_ready=True in the parent API process, so
# external publishers can block until the pipeline can actually consume
# frames. Without this, ZMQ PUB silently drops frames sent before the SUB is
# bound (~6 s after /run on a cold start).
_pipeline_ready = threading.Event()

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
from ptychoml import compute_sample_pixel_size


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
from .orientation import reduce_d4_sequence
from ptychoml.preprocess import D4_NAMES
from .liverecon_utils import parse_scan_header
from .vit_inference import (
    PtychoViTInferenceOp,
    SaveViTResult,
    MosaicWriterOp,
    PositionsWriterOp,
    BatchWriterOp,
)
from .engine_probe import find_onnx_for_engine, inner_crop_from_onnx
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
        # Refinement iterations after data collection completes (the run ends
        # it_ends_after iterations after the last point-count advance). With
        # fast replay ingest the stream finishes in tens of seconds, so this
        # is most of the engine's full-data iteration budget.
        self.it_ends_after = int(getattr(param, "it_ends_after", 30))
        _cre = getattr(param, "clear_region_enabled", True)
        self.clear_region_enabled = bool(_cre) if not isinstance(_cre, str) else _cre.lower() != "false"
        # n_iterations sizes the in-RAM probe/object history ring buffers in
        # StreamingRecon (it % n_iterations) — it is NOT the run length. The
        # run normally ends via finished_collecting (all points received +
        # it_ends_after refinement iterations); max_iterations is only a
        # safety cap. Capping at n_iterations used to kill streaming runs
        # mid-scan (300 iterations ~= 4k of 40k frames).
        self.n_iterations = int(getattr(param, "n_iterations", 500))
        self.max_iterations = int(getattr(param, "max_iterations", 1_000_000))
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
            self.max_iterations = self.it
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
            # clear_region zeroes the object under the NEW points' windows —
            # those 256 px windows overlap previously reconstructed rows, so
            # on dense fly scans every streaming advance wipes a stripe of
            # converged object. The sparsity heuristic (canvas/points > 16)
            # was meant for sparse scans; config clear_region_enabled=False
            # disables it outright (the replay does).
            if self.clear_region_enabled and np.ceil(self.recon.x_range_um*1e-6/self.recon.x_pixel_m)*np.ceil(self.recon.y_range_um*1e-6/self.recon.x_pixel_m)/self.points_total > 16:
                self.recon.clear_region(self.recon.num_points_recon, ready_num)
            # ML_grad recomputes product_d from the current probe/object every
            # batch (no running dual state), so newly activated points need no
            # product initialisation here — matches holoscan-framework.
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


        # Debug dump of the engine's exact inputs, for offline inspection /
        # offline ptycho recon. Set HOLOPTYCHO_DUMP_ENGINE_INPUTS=<dir> on the
        # server; fires once when 2500+ points are in.
        _dump_dir = os.environ.get("HOLOPTYCHO_DUMP_ENGINE_INPUTS")
        if (
            _dump_dir
            and not getattr(self, "_engine_inputs_dumped", False)
            and self.recon.num_points_recon >= 2500
        ):
            n = int(self.recon.num_points_recon)
            np.save(os.path.join(_dump_dir, "diff_d.npy"), self.recon.diff_d[:n].get())
            np.save(os.path.join(_dump_dir, "point_info_d.npy"), self.recon.point_info_d[:n].get())
            self._logger.info("Dumped engine inputs (%d points) to %s", n, _dump_dir)
            self._engine_inputs_dumped = True
        
        # Terminate when we've either (a) finished collecting data and run
        # `it_ends_after` more iterations, or (b) hit the safety iteration
        # cap. Both gated on `num_points_min < inf` so we only fire once.
        # NOTE: the cap is max_iterations, NOT n_iterations — n_iterations is
        # the history ring-buffer depth and is far smaller than a full
        # streaming scan's iteration count.
        finished_collecting = (
            self.it - self.it_last_update >= self.it_ends_after
        )
        hit_iteration_cap = self.it >= self.max_iterations
        if (finished_collecting or hit_iteration_cap) and self.num_points_min < np.inf:
            reason = "iteration cap" if hit_iteration_cap else "data collection complete"
            self._logger.info(
                "Iterative recon finishing (%s, it=%d, max_iterations=%d)",
                reason, self.it, self.max_iterations,
            )
            self.num_points_min = np.inf
            op_output.emit((self.recon,self.timestamp_iter,self.num_points_recv_iter),"output")
        sys.stdout.flush()
        sys.stderr.flush()
        
# Module-level writer — initialized once when the pipeline starts.
# Requires TILED_BASE_URL and TILED_API_KEY; raises RuntimeError otherwise.
_writer = get_writer()

# Probe amplitudes from the iterative engine come out normalized; ptycho-vit
# was trained on probes at 256× this scale (the model's fixed 256x256 input
# size). Multiply on the way out so live snapshots and the final probe land
# in Tiled at the magnitude downstream consumers expect.
_PROBE_SCALE = 256.0

@create_op(inputs="results")
def SaveLiveResult(results):
    try:
        _writer.write_live(
            iteration=results[2],
            probe=results[0] * _PROBE_SCALE,
            obj=results[1],
        )
    except Exception:
        pass

@create_op(inputs="output")
def SaveResult(output):
    print('Live recon done! Saving results..')
    engine = output[0]
    _writer.write_final(
        probe=np.asarray(engine.prb_mode) * _PROBE_SCALE,
        obj=np.asarray(engine.obj_mode),
        timestamps=np.array(output[1]),
        num_points=np.array(output[2]),
    )
    # Iterative branch finished — mark the run complete in metadata. For
    # vit-only runs SaveResult is never wired and the metadata flip happens
    # at clean subprocess exit (_pipeline_subprocess.main).
    _writer.mark_run_complete()
    print('Saving results done.')


class FrameWriterOp(Operator):
    """Persist detector-frame intensity to Tiled.

    Subscribes to ImagePreprocessorOp's ``intensity`` tap — chunks of
    ``(B, H, W)`` uint16 in detector-frame orientation (after bad-pixel
    inpaint, before rot90/fftshift/floor/sqrt). Each chunk is filtered on
    ``stride`` (keep frames where ``frame_idx % stride == 0``), mapped to
    compact-dp rows (``row = frame_idx // stride``), and patched into
    ``<run>/diffraction/dp`` via ``TiledWriter.write_diffraction_chunk``.

    For ``stride=1`` (default for iterative/both, opt-in for ViT-only): every
    frame is written — full fine-tuning capture. For ``stride>1`` (default
    1000 for ViT-only): only every Nth frame is written, providing
    spot-check visibility on the dashboard without the WAN cost of writing
    every frame. The stride is stamped into run metadata as ``dp_stride``
    so the dashboard can label its detector tile.
    """

    # Push a dp_frames_written metadata update at most this often. Each update
    # is a small HTTPS PUT and we don't need realtime — the dashboard polls
    # the slider state every 2 s anyway. Throttling keeps the metadata
    # endpoint from drowning under per-batch updates on fast scans.
    _PROGRESS_UPDATE_INTERVAL_S = 1.0

    def __init__(self, *args, stride: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self._stride = max(1, int(stride))
        self._first_batch = True
        self._max_row_written = 0
        self._last_progress_push = 0.0

    def setup(self, spec: OperatorSpec):
        spec.input("intensity").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)
        spec.input("image_indices").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)

    def compute(self, op_input, op_output, context):
        frames = op_input.receive("intensity")
        indices = op_input.receive("image_indices")
        idx_arr = np.asarray(indices)
        if self._first_batch:
            # Fail fast if the first batch isn't aligned to scan-frame zero —
            # that means upstream dropped frames (e.g. ZMQ PUB slow-joiner
            # race), and silently continuing would leave dp[0..N-1] filled
            # with the zero initial state. A model trained on those slots
            # would learn that "black detector" maps to whatever object/probe
            # was at those positions — real harm. Better to abort the run so
            # the operator can fix the publisher and retry.
            first = int(idx_arr[0])
            if first != 0:
                raise RuntimeError(
                    f"FrameWriterOp: first batch starts at scan frame {first}, "
                    "expected 0. Frames were dropped before the pipeline could "
                    "receive them. Likely cause: publisher started pushing "
                    "before holoptycho's ZMQ SUB was live (ZMQ PUB silently "
                    "drops to a not-yet-subscribed peer). Restart the "
                    "publisher after /run returns 200 — the API now blocks "
                    "until the pipeline is ready."
                )
            self._first_batch = False

        if self._stride <= 1:
            kept_frames = np.asarray(frames)
            kept_rows = idx_arr
        else:
            keep_mask = (idx_arr % self._stride) == 0
            if not keep_mask.any():
                return
            kept_frames = np.asarray(frames)[keep_mask]
            kept_rows = idx_arr[keep_mask] // self._stride

        _writer.write_diffraction_chunk(kept_rows, kept_frames)

        # Track high-water row index. Dashboard slider clamps against this
        # via the dp_frames_written metadata so users can't scroll past
        # actually-written rows.
        new_max = int(kept_rows[-1]) + 1
        if new_max > self._max_row_written:
            self._max_row_written = new_max
            now = time.time()
            if now - self._last_progress_push >= self._PROGRESS_UPDATE_INTERVAL_S:
                _writer.update_dp_progress(self._max_row_written)
                self._last_progress_push = now


class InferenceFrameWriterOp(Operator):
    """Persist ViT inference output to Tiled, mirroring FrameWriterOp's stride.

    Subscribes to ``vit_inference``'s ``vit_result`` (a tuple of ``(pred,
    indices)`` with ``pred`` shape ``(B, n_channels, H, W)`` and ``indices``
    shape ``(B,)``). Keeps the same frames FrameWriterOp keeps (``idx %
    stride == 0``), maps each to its compact row, and patches
    ``<run>/diffraction/inference[row]`` with the model's prediction.

    Result: for every frame written to ``<run>/diffraction/dp``, the
    corresponding model output is stored at the matching row of
    ``<run>/diffraction/inference``. The dashboard pairs the two tiles via
    the shared row index.
    """

    def __init__(self, *args, stride: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self._stride = max(1, int(stride))

    def setup(self, spec: OperatorSpec):
        spec.input("results").connector(
            IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32
        )

    def compute(self, op_input, op_output, context):
        try:
            results = op_input.receive("results")
            pred, indices = results
            idx_arr = np.asarray(indices)
            if self._stride <= 1:
                kept_pred = np.asarray(pred)
                kept_rows = idx_arr
            else:
                keep_mask = (idx_arr % self._stride) == 0
                if not keep_mask.any():
                    return
                kept_pred = np.asarray(pred)[keep_mask]
                kept_rows = idx_arr[keep_mask] // self._stride
            _writer.write_inference_chunk(kept_rows, kept_pred)
        except Exception:
            logging.getLogger(__name__).exception(
                "InferenceFrameWriterOp.compute failed"
            )


class PtychoApp(Application):
    def __init__(self, *args, config_path=None, config_overrides=None, engine_path=None, **kwargs):
        super().__init__(*args,**kwargs)

        self.config_path = config_path
        self.param = parse_config(self.config_path, config_overrides=config_overrides)
        self.gpu = self.param.gpus[0]
        self.engine_path = engine_path or "/models/ptycho_vit_amp_phase_b64.engine"

        # Multi-threaded scheduler so downstream tiled-write operators
        # (MosaicWriterOp, BatchWriterOp) run concurrently with the upstream
        # ViT branch instead of serially under the default GreedyScheduler.
        self.scheduler(MultiThreadScheduler(
            self,
            worker_thread_number=11,
            check_recession_period_ms=1.0,
            stop_on_deadlock=True,
            stop_on_deadlock_timeout=500,
            name="multithread_scheduler",
        ))

    def config_ops(self,param):
        # vit-only mode (recon_mode='vit') skips creating the PtychoRecon
        # engine (no CuPy context on the ViT GPU), so derive the geometry the
        # engine would have provided from config instead.
        has_recon = hasattr(self, 'pty')
        if has_recon:
            nx_prb = self.pty.recon.nx_prb
            ny_prb = self.pty.recon.ny_prb
            nz = self.pty.recon.num_points
        else:
            nx_prb = int(param.nx)
            ny_prb = int(param.ny)
            nz = int(getattr(param, "live_num_points_max", 0)) or 8192

        self.image_batch.roi = None
        self.image_batch.batchsize = self.batchsize
        self.image_batch.detector_orientation = self.detector_orientation
        self.image_batch.nx_prb = nx_prb
        self.image_batch.ny_prb = ny_prb
        self.image_batch.images_to_add = np.zeros((self.batchsize, nx_prb, ny_prb), dtype = np.uint32)
        self.image_batch.indices_to_add = np.zeros(self.batchsize, dtype=np.int32)

        self.image_proc.detmap_threshold = 0
        self.image_proc.badpixels = np.array([])

        # Iterative-engine GPU targets only exist when the engine does; in
        # vit-only mode they stay None and ImageSendOp/PointProcessorOp skip
        # the device copies.
        if has_recon:
            self.image_send.diff_d_target = self.pty.recon.diff_d
        self.image_send.max_points = nz

        self.point_proc.point_info = np.zeros((nz,4),dtype = np.int32)
        if has_recon:
            self.point_proc.point_info_target = self.pty.recon.point_info_d
            # Engine GPU buffers hold num_points_l points (live_num_points_max,
            # default 8192) — usually far fewer than the scan's nz. The feed
            # guards must use THIS capacity: writing past it is an
            # out-of-bounds GPU write (cudaErrorIllegalAddress).
            self.point_proc.target_max_points = int(self.pty.recon.num_points_l)
            self.image_send.target_max_points = int(self.pty.recon.num_points_l)
        # Per-frame scan positions (microns), filled by PointProcessorOp as
        # PandA data arrives. Read by SaveViTResult and published to tiled
        # so the dashboard mosaic stitcher uses real positions.
        self.point_proc.positions_um = np.full((nz, 2), np.nan, dtype=np.float64)

        self.point_proc.min_points = self.min_points
        self.point_proc.max_points = nz
        if has_recon:
            # NOTE: point_proc.x/y_direction deliberately keep the SHARED
            # config convention set at construction (they feed positions_um /
            # the ViT mosaic). The engine's convention may differ via
            # x/y_direction_iterative, carried by point_proc.x/y_sign_rel —
            # recon.x/y_direction already reflect the iterative values, so the
            # old copy-down here would leak them onto the ViT stream.
            self.point_proc.x_range_um = self.pty.recon.x_range_um
            self.point_proc.y_range_um = self.pty.recon.y_range_um
            self.point_proc.x_pixel_m = self.pty.recon.x_pixel_m
            self.point_proc.y_pixel_m = self.pty.recon.y_pixel_m
            self.point_proc.obj_pad = self.pty.recon.obj_pad
            # Engine object-array dims: lets PointProcessorOp center the scan
            # in the (over-allocated) object array and clamp out-of-bounds
            # crop windows instead of handing the gather/scatter kernels
            # negative indices (cudaErrorIllegalAddress).
            self.point_proc.obj_nx_limit = int(self.pty.recon.nx_obj)
            self.point_proc.obj_ny_limit = int(self.pty.recon.ny_obj)
        else:
            # x_direction/y_direction are already set on point_proc at
            # construction from config; here we only need the pixel size so
            # process_point_info can convert encoder counts to microns. Match
            # the engine's far-field formula via ptychoml.
            _lnm = float(getattr(param, "lambda_nm", None) or
                         (1.2398 / float(param.xray_energy_kev)))
            _zm = float(getattr(param, "z_m", 0.0))
            _ccdpx = float(getattr(param, "ccd_pixel_um", 55.0))
            if _zm and _ccdpx and nx_prb and ny_prb:
                _wl_m, _ccd_m = _lnm * 1e-9, _ccdpx * 1e-6
                self.point_proc.x_pixel_m = compute_sample_pixel_size(_wl_m, _zm, _ccd_m, nx_prb)
                self.point_proc.y_pixel_m = compute_sample_pixel_size(_wl_m, _zm, _ccd_m, ny_prb)
            self.point_proc.obj_pad = int(getattr(param, "obj_pad", 4))
        self.point_proc.nx_prb = nx_prb
        self.point_proc.ny_prb = ny_prb

        self.point_proc.angle_correction_flag = param.angle_correction_flag

        if has_recon:
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

        # Local->global coordinate correction (a ptychoml D4 name) applied to the
        # whole raw frame before cropping. Source-dependent: live Eiger ZMQ is
        # raw-local (default 'fliplr' — a flip over the vertical axis); replay
        # sets 'identity' because Tiled data is already corrected. The old
        # `flip_image=True` applied 'fliplr' to BOTH sources: that happened to be
        # correct for live but double-flipped replay back to local coords — do
        # NOT map the legacy flag to a D4.
        self.detector_orientation = str(
            getattr(self.param, "detector_orientation", "fliplr")
        )
        if self.detector_orientation not in D4_NAMES:
            raise ValueError(
                f"detector_orientation must be one of {D4_NAMES}; got "
                f"{self.detector_orientation!r}"
            )
        if getattr(self.param, "flip_image", None) is not None and not getattr(
            self.param, "detector_orientation", None
        ):
            print(
                "WARNING: 'flip_image' is deprecated and ignored; use "
                "'detector_orientation' (D4 name). Live default is 'fliplr'.",
                file=sys.stderr,
            )

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
        # auto_center_dp (lossless centered crop) is wired into ImageBatchOp
        # after its ROI is set below.
        # Geometry + normalization for the two output branches. See
        # ImagePreprocessorOp docstrings for what each does. ``fftshift_dp``
        # defaults to None so ptychoml's auto-detector picks the right DC
        # convention per batch; override via the scan JSON only when a
        # specific dataset misbehaves.
        self.image_proc.tap_orient = str(getattr(self.param, "tap_orient", "antitranspose"))
        # dp_orient: D4 aligning the (global) diffraction frame to the model /
        # scan convention. The frame is already global after
        # detector_orientation, so the default is 'identity' — fully
        # deterministic, no runtime sweep. 'auto' opts in to the ViT
        # orientation-autodetect sweep (vit/both modes), which overwrites
        # dp_orient at runtime with the best-scoring D4.
        _dp_orient_cfg = str(getattr(self.param, "dp_orient", None) or "identity")
        self.orient_autodetect = _dp_orient_cfg == "auto"
        if self.orient_autodetect:
            _dp_orient_cfg = "identity"  # seed only; the sweep overwrites it
        elif _dp_orient_cfg not in D4_NAMES:
            raise ValueError(
                f"dp_orient must be 'auto' or one of {D4_NAMES}; got {_dp_orient_cfg!r}"
            )
        self.image_proc.dp_orient = _dp_orient_cfg
        _fftshift_dp = getattr(self.param, "fftshift_dp", None)
        self.image_proc.fftshift_dp = (
            bool(_fftshift_dp) if _fftshift_dp is not None else None
        )
        # Per-scan ViT normalization: the max intensity across all DPs in
        # this scan, with hot pixels excluded. In live mode we don't have
        # the full DP stack to compute it from, so it must come from the
        # scan JSON (operator pre-computes it offline from a prior scan, or
        # ptychoml.compute_intensity_normalization can produce it from a
        # representative subset before the live run starts). Without it the
        # amplitude scale drifts from what the model was trained against and
        # predictions are systematically off-scale.
        self.image_proc.normalization = float(getattr(self.param, "vit_normalization", 1.0e5))
        self.image_proc.scale = float(getattr(self.param, "vit_scale", 1.0e4))
        # Photon-count threshold for hot-pixel zeroing (None disables). Matches
        # hxn_to_vit.py's default when enabled.
        _hot_pix = getattr(self.param, "hot_pixel_count_threshold", None)
        self.image_proc.hot_pixel_count_threshold = (
            float(_hot_pix) if _hot_pix is not None else None
        )
        self.image_send = ImageSendOp(self, name="image_send")
        self.point_proc = PointProcessorOp(
            self,
            x_direction=self.param.x_direction,
            y_direction=self.param.y_direction,
            swap_xy=bool(getattr(self.param, "position_swap_xy", False)),
            name="point_proc",
        )

        # --- Iterative-only orientation / direction overrides (demo knobs) ---
        # dp_orient_iterative: absolute D4 (or comma-separated sequence reduced
        # to one element) for the ITERATIVE engine's diffraction input.
        # Unset = disabled — the engine follows the shared dp_orient, including
        # any runtime change by the orientation auto-detect. Do NOT default it
        # to the config dp_orient: that would counter-rotate the engine input
        # the moment the auto-detect updates dp_orient.
        _dp_it = getattr(self.param, "dp_orient_iterative", None)
        if _dp_it:
            _dp_it = reduce_d4_sequence(str(_dp_it))
            if int(self.param.nx) % 2 or int(self.param.ny) % 2:
                raise ValueError(
                    "dp_orient_iterative requires even nx/ny (the relative D4 "
                    f"is applied post-fftshift); got nx={self.param.nx} "
                    f"ny={self.param.ny}"
                )
            self.image_send.dp_orient_iterative = _dp_it
        # x/y_direction_iterative: ±1 sign convention for the engine's
        # point_info stream. Unset = follow the shared x/y_direction
        # (value-fallback is safe here — these are static config values).
        _xd = float(self.param.x_direction)
        _yd = float(self.param.y_direction)
        _xd_it = float(getattr(self.param, "x_direction_iterative", _xd) or _xd)
        _yd_it = float(getattr(self.param, "y_direction_iterative", _yd) or _yd)
        for _name, _v in (("x_direction", _xd), ("y_direction", _yd),
                          ("x_direction_iterative", _xd_it),
                          ("y_direction_iterative", _yd_it)):
            if abs(_v) != 1.0:
                raise ValueError(f"{_name} must be +1.0 or -1.0, got {_v}")
        # For unit signs the relative factor iter/shared equals iter*shared.
        self.point_proc.x_sign_rel = _xd_it * _xd
        self.point_proc.y_sign_rel = _yd_it * _yd
        if recon_mode == "vit" and (
            _dp_it or _xd_it != _xd or _yd_it != _yd
        ):
            print(
                "WARNING: dp_orient_iterative / x|y_direction_iterative are "
                "set but recon_mode='vit' — the iterative engine is not "
                "wired, so these flags have no effect.",
                file=sys.stderr,
            )

        # vit-only mode skips the iterative engine and its result writers so
        # no CuPy context is opened on the ViT GPU (PyCUDA + CuPy on the same
        # device from different threads can crash — SIGABRT).
        if recon_mode in ("iterative", "both"):
            self.pty = PtychoRecon(self, param=self.param, name='pty')
            self.o = SaveResult(self, name='out')
            self.live_result = SaveLiveResult(self, name='live_result')

        self.config_ops(self.param)

        # Initialize operators from config (replaces InitRecon scan header flush).
        # ImageBatchOp crop: a SINGLE global ROI offset in the coordinate-corrected
        # frame — batch_x0/batch_y0 mark the top-left of the nx x ny model window.
        # (The old det_roi base offset is gone; the acquisition crop that made
        # replay frames pre-cropped is handled by detector_orientation='identity'
        # on the replay side, not by config math.) Auto-centering, if enabled,
        # overrides this with a beam-centered box.
        if int(getattr(self.param, "det_roix0", 0)) or int(getattr(self.param, "det_roiy0", 0)):
            print(
                "WARNING: 'det_roix0'/'det_roiy0' are deprecated and ignored; the "
                "crop is a single global ROI (batch_x0/batch_y0) after coordinate "
                "correction.",
                file=sys.stderr,
            )
        batch_x0 = int(self.param.batch_x0)
        batch_y0 = int(self.param.batch_y0)
        self.image_batch.roi = np.array([
            [batch_y0, batch_y0 + self.param.ny],
            [batch_x0, batch_x0 + self.param.nx],
        ])
        # Opt-in lossless DP auto-centering (default off). When enabled,
        # ImageBatchOp buffers the first batch at a ±headroom window, segments to
        # find the beam, and crops every batch to an nx x ny box centered on it
        # (no np.roll, no zero-fill). Headroom defaults to nx//4. Enable via
        # auto_center_dp=true in the scan JSON only when the detector ROI is too
        # far off-centre to fix with batch_x0/batch_y0.
        self.image_batch.auto_center = bool(getattr(self.param, "auto_center_dp", False))
        self.image_batch.headroom = int(
            getattr(self.param, "auto_center_headroom", int(self.param.nx) // 4)
        )

        # PointProcessorOp: set scan geometry from config.
        self.point_proc.x_range_um = np.abs(self.param.x_range)
        self.point_proc.y_range_um = np.abs(self.param.y_range)
        self.point_proc.x_ratio = float(self.param.x_ratio)
        self.point_proc.y_ratio = float(self.param.y_ratio)
        self.point_proc.min_points = num_points_max
        self.point_proc.angle = float(getattr(self.param, 'angle', 0.0))
        # Number of raw encoder samples per detector frame; PointProcessorOp
        # averages each group of this many raw samples down to one position.
        # Default 1 = positions are 1:1 with frames (replays of pre-averaged
        # tiled data). The real HXN beamline currently emits 10 raw samples
        # per frame, so its prod config sets panda_upsample=10. Mismatch
        # leaves positions_um mostly NaN and silently stalls the mosaic.
        self.point_proc.upsample = int(getattr(self.param, 'panda_upsample', 1))
        self.point_proc.simulate_positions = bool(getattr(self.param, 'simulate_positions', False))
        if self.point_proc.simulate_positions:
            self.point_proc.pos0_simul = np.tile(
                np.linspace(0, self.param.x_range, x_num + 1)[:-1], [y_num, 1]
            ).reshape((x_num * y_num,)) * self.point_proc.x_direction
            self.point_proc.pos1_simul = np.tile(
                np.linspace(0, self.param.y_range, y_num + 1)[:-1], [x_num, 1]
            ).T.reshape((x_num * y_num,)) * self.point_proc.y_direction

        # PtychoRecon: reset engine for this scan (iterative/both only).
        if recon_mode in ("iterative", "both"):
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

        # Sample-plane geometry for the metadata and the ViT mosaic. Read from
        # the engine when it exists, otherwise computed directly from config
        # (vit-only, no CuPy context). Matches the engine's far-field formula.
        _lambda_nm = float(getattr(self.param, "lambda_nm", None) or
                           (1.2398 / float(self.param.xray_energy_kev)))
        if hasattr(self, 'pty'):
            _x_pixel_m = float(self.pty.recon.x_pixel_m)
            _y_pixel_m = float(self.pty.recon.y_pixel_m)
        else:
            _z_m = float(getattr(self.param, "z_m", 0.0))
            _ccd_px = float(getattr(self.param, "ccd_pixel_um", 55.0))
            _nx, _ny = int(self.param.nx), int(self.param.ny)
            if _z_m and _ccd_px and _nx and _ny:
                _wl_m, _ccd_m = _lambda_nm * 1e-9, _ccd_px * 1e-6
                _x_pixel_m = compute_sample_pixel_size(_wl_m, _z_m, _ccd_m, _nx)
                _y_pixel_m = compute_sample_pixel_size(_wl_m, _z_m, _ccd_m, _ny)
            else:
                _x_pixel_m = _y_pixel_m = 0.0

        # Each pipeline run gets a fresh container in Tiled keyed by its own uid.
        # Metadata captures the raw scan being reconstructed plus the scan-grid
        # geometry that downstream consumers (synaps-dash) need to stitch
        # per-frame ViT predictions into a global mosaic.
        self.run_uid = uuid.uuid4().hex
        run_metadata = {
            "scan_num": str(self.param.scan_num),
            "raw_uid": str(getattr(self.param, "raw_uid", "") or ""),
            "scan_id": str(
                getattr(self.param, "scan_id", self.param.scan_num)
            ),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "recon_mode": recon_mode,
            "x_pixel_m": _x_pixel_m,
            "y_pixel_m": _y_pixel_m,
            "x_num": int(self.param.x_num),
            "y_num": int(self.param.y_num),
            "x_range_um": float(np.abs(self.param.x_range)),
            "y_range_um": float(np.abs(self.param.y_range)),
            "x_direction": float(self.param.x_direction),
            "y_direction": float(self.param.y_direction),
            "xray_energy_kev": float(getattr(self.param, "xray_energy_kev", 0.0)),
            "wavelength_m": _lambda_nm * 1e-9,
            "distance_m": float(getattr(self.param, "z_m", 0.0)),
            # True iff this run's iterative branch will populate final/probe
            # and final/object — the supervised targets ptycho-vit's training
            # loader requires. Lets downstream tooling list fine-tuning
            # candidates via a Tiled query without inspecting subcontainers.
            "fine_tunable": recon_mode in ("iterative", "both"),
            # Flipped to True when the holoscan pipeline finishes processing
            # this scan (SaveResult for iterative/both, or clean subprocess
            # exit for vit-only). Used to filter mid-flight runs out of batch
            # processing without inspecting state.
            "complete": False,
        }
        # Iterative-only overrides — recorded only when set so downstream
        # consumers can tell the engine's conventions differed from the shared
        # x/y_direction / dp_orient above.
        if _dp_it:
            run_metadata["dp_orient_iterative"] = _dp_it
        if _xd_it != _xd:
            run_metadata["x_direction_iterative"] = _xd_it
        if _yd_it != _yd:
            run_metadata["y_direction_iterative"] = _yd_it
        # Record which ViT engine produced this run's vit/ outputs (issue #34),
        # so a run's provenance doesn't require log-archaeology. Forwarded from
        # the server via env (runner.py); absent for standalone/non-server runs.
        # Only meaningful when the ViT branch actually ran.
        if recon_mode in ("vit", "both"):
            _model_name = os.environ.get("HOLOPTYCHO_MODEL_NAME")
            _model_version = os.environ.get("HOLOPTYCHO_MODEL_VERSION")
            _engine_path = os.environ.get("HOLOPTYCHO_ENGINE_PATH")
            if _model_name:
                run_metadata["model_name"] = str(_model_name)
            if _model_version:
                run_metadata["model_version"] = str(_model_version)
            if _engine_path:
                run_metadata["engine_path"] = str(_engine_path)
        _writer.start_run(self.run_uid, metadata=run_metadata)

        # Detector-frame downsampling. ViT-only runs default to keeping 1
        # in every 1000 frames (40 frames for a 40K-point scan) — enough
        # for an operator to spot-check that preprocessing looks right on
        # the dashboard, without paying the WAN cost of writing every
        # frame. Iterative/both runs default to stride=1 (every frame)
        # because final/probe + final/object only mean ptycho-vit can
        # fine-tune on this run if every frame is present. Explicit
        # `frame_write_stride` in the config overrides the default.
        stride_cfg = getattr(self.param, "frame_write_stride", None)
        if stride_cfg is None:
            frame_write_stride = 1000 if recon_mode == "vit" else 1
        else:
            frame_write_stride = max(1, int(stride_cfg))
        nz_total = int(x_num * y_num)
        n_keep = (nz_total - 1) // frame_write_stride + 1
        _writer.start_diffraction_buffer(
            n_keep=n_keep,
            frame_shape=(int(self.param.nx), int(self.param.ny)),
            dtype=np.uint8,
            stride=frame_write_stride,
        )
        # Sibling inference buffer: for each kept dp row, store the ViT model's
        # (amp, phase) prediction so the dashboard can pair them. Only wired
        # when the ViT branch runs ("vit" or "both").
        if recon_mode in ("vit", "both"):
            _writer.start_inference_buffer(
                n_keep=n_keep,
                frame_shape=(int(self.param.nx), int(self.param.ny)),
                dtype=np.float32,
                n_channels=2,
                stride=frame_write_stride,
            )

        # --- PtychoViT inference (parallel to iterative recon) ---
        # Prefer a second GPU for PyCUDA/TRT when available, but fall back to
        # the recon GPU on single-GPU nodes instead of hard-failing.
        vit_gpu = self.param.gpus[1] if len(self.param.gpus) > 1 else self.param.gpus[0]
        # DC-convention handling is auto-detected end-to-end by ptychoml
        # (ImagePreprocessorOp's preprocess_diffraction centers the beam, and
        # the session's own detect_dc_at_corner check verifies it), so no
        # manual fftshift flag is needed here — leave fftshift at its default
        # None.
        #
        # Orientation auto-detect wiring: the op runs ptychoml.autodetect_
        # orientation on the first batch with enough finite positions, then
        # sets image_proc.dp_orient. preprocess_kwargs mirrors the live
        # preprocessing so the sweep scores under the same normalization/scale/
        # hot-pixel/fftshift settings (dp_orient excluded — it's the sweep var).
        _preprocess_kwargs = {
            "normalization": float(self.image_proc.normalization),
            "scale": float(self.image_proc.scale),
            "hot_pixel_count_threshold": self.image_proc.hot_pixel_count_threshold,
            "fftshift": self.image_proc.fftshift_dp,
        }
        self.vit = PtychoViTInferenceOp(
            self,
            engine_path=self.engine_path,
            gpu=vit_gpu,
            image_proc=self.image_proc,
            positions_provider=lambda: self.point_proc.positions_um,
            preprocess_kwargs=_preprocess_kwargs,
            orient_autodetect=self.orient_autodetect,
            name="vit_inference",
        )
        # SaveViTResult publishes positions_um alongside each batch and
        # accumulates the running phase mosaic at <run>/vit/mosaic. The
        # dashboard renders that array directly — no client-side stitching.
        # Per-batch pred + indices export is gated by the config field
        # vit_batch_writes (default off) — see SaveViTResult docstring.
        enable_batch_writes = bool(getattr(self.param, "vit_batch_writes", False))
        # Canvas safety margin. 1.2 fits ~83% of the canvas with painted
        # scan area on a typical 5 µm scan; bump for HXN scans with
        # settling-row overshoot (e.g. 404611 commanded 2 µm → observed
        # 6 µm needs ≥3.0). Off-canvas frames trigger a warning in
        # SaveViTResult and are dropped.
        mosaic_overshoot = float(getattr(self.param, "mosaic_overshoot_factor", 1.2))
        # Min fractional overlap for a mosaic pixel to count as "covered"
        # when normalising canvas / counts; below this the pixel is filled
        # with the valid-region median instead of a thin-coverage value.
        _min_overlap = float(getattr(self.param, "mosaic_min_overlap", 0.5))
        # D4 transform applied to each ViT output patch before stitching
        # (via ptychoml.apply_d4). Default 'identity'; the orientation
        # auto-detector will choose this once wired up.
        _patch_flip = str(getattr(self.param, "patch_flip", "identity"))
        # inner_crop: how many pixels to trim from each ViT output patch before
        # stitching (drops the FFT-leakage border outside the probe support).
        # Precedence: explicit config > derived from the ONNX probe geometry >
        # None (SaveViTResult auto-derives from the patch size).
        _inner_crop_cfg = int(getattr(self.param, "inner_crop", 0))
        if _inner_crop_cfg > 0:
            _inner_crop = _inner_crop_cfg
        else:
            _inner_crop = None
            _onnx_path = find_onnx_for_engine(self.engine_path)
            if _onnx_path is not None:
                _inner_crop = inner_crop_from_onnx(_onnx_path)
                if _inner_crop is not None:
                    logging.getLogger("holoptycho.PtychoApp").info(
                        "inner_crop=%d derived from probe in %s",
                        _inner_crop, _onnx_path.name,
                    )
        self.vit_save = SaveViTResult(
            self,
            positions_provider=lambda: self.point_proc.positions_um,
            pixel_size_m=_x_pixel_m,
            # positions_um col 0 = slow axis (INENC3, y_range) → canvas width;
            # col 1 = fast axis (INENC2, x_range) → canvas height. Matches the
            # PointProcessorOp default column convention (swap_xy=False).
            x_range_um=float(np.abs(self.param.y_range)),
            y_range_um=float(np.abs(self.param.x_range)),
            overshoot_factor=mosaic_overshoot,
            min_overlap_count=_min_overlap,
            patch_flip=_patch_flip,
            inner_crop=_inner_crop,
            enable_batch_writes=enable_batch_writes,
            name="vit_save",
        )
        self.mosaic_writer = MosaicWriterOp(self, name="mosaic_writer")
        self.positions_writer = PositionsWriterOp(self, name="positions_writer")
        if enable_batch_writes:
            self.batch_writer = BatchWriterOp(self, name="batch_writer")
        self.frame_writer = FrameWriterOp(self, name="frame_writer", stride=frame_write_stride)
        if recon_mode in ("vit", "both"):
            self.inference_writer = InferenceFrameWriterOp(
                self, name="inference_writer", stride=frame_write_stride,
            )

        self.add_flow(self.eiger_zmq_rx, self.eiger_decompress, {("image_index_encoding", "image_index_encoding")})
        self.add_flow(self.eiger_decompress, self.image_batch, {("decompressed_image", "image"), ("image_index", "image_index")})
        self.add_flow(self.image_batch, self.image_proc, {("image_batch", "image_batch"), ("image_indices", "image_indices_in")})
        self.add_flow(self.image_proc, self.image_send, {("diff_amp", "diff_amp"), ("image_indices", "image_indices"), ("dp_orient_used", "dp_orient_used")})
        # Tap detector-frame intensity (pre-rot/shift) into the Tiled
        # diffraction buffer. Always wired so the dashboard tile and
        # downstream ptycho-vit fine-tuning have the data on any run.
        self.add_flow(self.image_proc, self.frame_writer, {
            ("intensity", "intensity"),
            ("image_indices", "image_indices"),
        })

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
            # Persist model output per kept dp row — sibling of FrameWriterOp.
            self.add_flow(self.vit, self.inference_writer, {("vit_result", "results")})
            # Async tiled mosaic write: capacity=1 + QueuePolicy.POP on the
            # writer's input drops superseded snapshots while a write is in
            # flight, so the ViT branch keeps stitching at full cadence.
            self.add_flow(self.vit_save, self.mosaic_writer, {("mosaic_snapshot", "snapshot")})
            # Async tiled positions write: same drop-policy semantics.
            self.add_flow(self.vit_save, self.positions_writer, {("positions_snapshot", "snapshot")})
            if enable_batch_writes:
                # Per-batch pred + indices via a bounded FIFO (no drop,
                # every batch is unique data). Gated by config.
                self.add_flow(self.vit_save, self.batch_writer, {("vit_batch", "batch")})

        # Graph is built and every op's __init__ has run — for EigerZmqRxOp
        # that means socket.connect() has fired, so the SUB is bound to the
        # publisher endpoint. Setting this flag triggers the subprocess
        # sentinel watcher, which lets the parent API release a /run caller
        # that's been blocking on readiness. Frames sent now will land in the
        # SUB's HWM buffer until the scheduler starts pulling them in
        # app.run() (the very next call).
        _pipeline_ready.set()


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

    app = PtychoApp(config_path=config_path)

    # Scheduler is configured in PtychoApp.__init__.
    app.run()
    
    
