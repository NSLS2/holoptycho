import logging

import numpy as np
import cupy as cp

from holoscan.core import Operator, OperatorSpec
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler, EventBasedScheduler
from holoscan.logger import LogLevel, set_log_level
from holoscan.decorator import create_op, Input

from pipeline_source import EigerRxBase, parse_args

class ImageBatchOp(Operator):
    def __init__(self, *args, batchsize=250, **kwargs):
        self.logger = logging.getLogger("ImageBatchOp")
        logging.basicConfig(level=logging.INFO)
        self.counter = 0
        self.batchsize = batchsize
        self.images_to_add = np.zeros((self.batchsize, 256, 256))
        self.indices_to_add = np.zeros(self.batchsize, dtype=np.int32)
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("image")
        spec.input("image_index")
        spec.output("image_batch")
        spec.output("image_indices")
        
    def compute(self, op_input, op_output, context):
        image = op_input.receive("image")
        image_index = op_input.receive("image_index")
        
        self.images_to_add[self.counter, :, :] = image
        self.indices_to_add[self.counter] = image_index
        
        if self.counter < (self.batchsize - 1):
            self.counter += 1
        else:
            op_output.emit(self.images_to_add.copy(), "image_batch")
            op_output.emit(self.indices_to_add.copy(), "image_indices")
            self.counter = 0

class PointBatchOp(Operator):
    def __init__(self, *args, batchsize=200, **kwargs):
        self.logger = logging.getLogger("PointBatchOp")
        logging.basicConfig(level=logging.INFO)
        self.counter = 0
        self.batchsize = batchsize
        self.point_buffer = None
        self.index_buffer = None

        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("point")
        spec.input("point_index")
        spec.output("point_batch")
        spec.output("point_indices")
        
    def compute(self, op_input, op_output, context):
        new_points = op_input.receive("point")
        new_indices = op_input.receive("point_index")
        if self.point_buffer is None:
            self.point_buffer = new_points
            self.index_buffer = new_indices
        else:
            self.point_buffer = np.concatenate((self.point_buffer, new_points), axis=1)
            self.index_buffer = np.concatenate((self.index_buffer, new_indices), axis=0)
        
        if self.point_buffer.shape[1] < self.batchsize:
            return
        else:
            op_output.emit(self.point_buffer[:, :self.batchsize], "point_batch")
            op_output.emit(self.index_buffer[:self.batchsize], "point_indices")
            self.point_buffer = self.point_buffer[:, self.batchsize:]
            self.index_buffer = self.index_buffer[self.batchsize:]


class ImagePreprocessorOp(Operator):
    def __init__(self, *args,
                #  roi=[[1, 257], [1,257]],
                 detmap_threshold=0,
                 badpixels=[[207], [211]],
                 **kwargs):
        self.logger = logging.getLogger("ImagePreprocessorOp")
        logging.basicConfig(level=logging.INFO)
        # self.roi = np.array(roi)
        self.detmap_threshold = detmap_threshold
        self.badpixels = np.array(badpixels)
        super().__init__(*args, **kwargs)
        
    def setup(self, spec: OperatorSpec):
        spec.input("image_batch")
        spec.input("image_indices_in")
        spec.output("diff_amp")
        spec.output("image_indices")
        
    def compute(self, op_input, op_output, context):
        images = op_input.receive("image_batch")
        indices = op_input.receive("image_indices_in")
        
        processed_images = cp.asarray(images)
        
        for bd in self.badpixels.T:
            x = int(bd[0])
            y = int(bd[1])
            processed_images[:, x, y] = cp.median(processed_images[:, x-1:x+2, y-1:y+2], axis=(2, 1))
        
        # processed_images = processed_images[:, self.roi[0,0]:self.roi[0,1], self.roi[1,0]:self.roi[1,1]]
        processed_images = cp.rot90(processed_images, axes=(2,1))
        processed_images = cp.fft.fftshift(processed_images, axes=(2,1))
        if self.detmap_threshold > 0:
            processed_images[processed_images<self.detmap_threshold] = 0
        diff_amp = cp.sqrt(processed_images)

        op_output.emit(diff_amp, "diff_amp")
        op_output.emit(cp.asarray(indices), "image_indices")

class PointPreprocessorOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("PointPreprocessorOp")
        logging.basicConfig(level=logging.INFO)
        
    def setup(self, spec: OperatorSpec):
        spec.input("point_batch")
        spec.input("point_indices_in")
        spec.output("processed_points")
        spec.output("point_indices")
        
    def compute(self, op_input, op_output, context):
        points = op_input.receive("point_batch")
        indices = op_input.receive("point_indices_in")
        # Add any position processing here if needed in the future
        processed_points = cp.asarray(points)#.copy()

        op_output.emit(processed_points, "processed_points")
        op_output.emit(cp.asarray(indices), "point_indices")

class DataGatherOp(Operator):
    def __init__(self, *args, 
                 num_parallel_streams=2,
                 num_batches_per_emit=2,
                 num_batches_overlap=0,
                 **kwargs):
        self.logger = logging.getLogger("DataGatherOp")
        logging.basicConfig(level=logging.INFO)
        self.num_parallel_streams = num_parallel_streams
        self.num_batches_per_emit = num_batches_per_emit
        self.num_batches_overlap = num_batches_overlap
        self.current_port = 1
        
        # Store for previous mini-batches
        self.stored_diff_amps = []
        self.stored_points = []
        self.stored_image_indices = []  # Keep this for internal tracking
        
        super().__init__(*args, **kwargs)
        
    def setup(self, spec: OperatorSpec):
        spec.input("diff_amp")
        spec.input("image_indices")
        spec.input("points")
        spec.input("point_indices")
        for i in range(1, self.num_parallel_streams + 1):
            spec.output(f"batch{i}")
        
    def compute(self, op_input, op_output, context):
        # Get current mini-batch of batchsize
        diff_amp = op_input.receive("diff_amp")
        image_indices = op_input.receive("image_indices")
        points = op_input.receive("points")
        point_indices = op_input.receive("point_indices")
        
        # Check if indices match exactly
        indices_match = cp.array_equal(image_indices, point_indices)
        if not indices_match:
            # Find differences for detailed logging
            only_in_image = cp.setdiff1d(image_indices, point_indices)
            only_in_point = cp.setdiff1d(point_indices, image_indices)
            self.logger.warning(f"Indices mismatch detected! "
                               f"Image indices shape: {image_indices.shape}, "
                               f"Point indices shape: {point_indices.shape}, "
                               f"Indices only in image ({only_in_image.shape}): {only_in_image}, "
                               f"Indices only in point ({only_in_point.shape}): {only_in_point}")
        
        # Ensure indices match by finding common indices
        common_indices = cp.intersect1d(image_indices, point_indices)
        
        if len(common_indices) > 0:
            # Create arrays to store positions in the original arrays
            img_positions = cp.zeros(len(common_indices), dtype=cp.int32)
            pnt_positions = cp.zeros(len(common_indices), dtype=cp.int32)
            
            # Find positions of common indices in the original arrays
            for i, idx in enumerate(common_indices):
                img_positions[i] = cp.where(image_indices == idx)[0][0]
                pnt_positions[i] = cp.where(point_indices == idx)[0][0]
            
            # Use these positions to filter the data
            filtered_diff_amp = diff_amp[img_positions]
            filtered_points = points[:, pnt_positions]
            
            # Store the filtered mini-batch
            self.stored_diff_amps.append(filtered_diff_amp)
            self.stored_points.append(filtered_points)
            self.stored_image_indices.append(common_indices)  # Keep for internal tracking
            
            # If we have enough mini-batches to emit
            if len(self.stored_diff_amps) >= self.num_batches_per_emit:
                # Create combined batch from stored mini-batches - without indices
                batch = {
                    "diff_amp": cp.concatenate(self.stored_diff_amps[:self.num_batches_per_emit]),
                    "points": cp.concatenate(self.stored_points[:self.num_batches_per_emit], axis=1)
                }
                
                # Emit through current port
                op_output.emit(batch, f"batch{self.current_port}")
                
                # Remove oldest mini-batches, keeping overlap
                keep_count = self.num_batches_overlap
                self.stored_diff_amps = self.stored_diff_amps[self.num_batches_per_emit - keep_count:]
                self.stored_points = self.stored_points[self.num_batches_per_emit - keep_count:]
                self.stored_image_indices = self.stored_image_indices[self.num_batches_per_emit - keep_count:]
                
                # Increment port number, wrapping back to 1 after reaching num_parallel_streams
                self.current_port = (self.current_port % self.num_parallel_streams) + 1

@create_op(inputs=Input("batch", arg_map={"diff_amp": "images", "points": "points"}))
def sink_func(images, points):
    if images.shape[0] == 250:
        print(f"SinkOp received images: shape={images.shape}, {images[0,0,0]=}")
        print(f"SinkOp received images: shape={points.shape}, {points[0,0]=}")
    else:
        print(f"SinkOp received images: shape={images.shape}, {images[0,0,0]=}, {images[250,0,0]=}")
        print(f"SinkOp received points: shape={points.shape}, {points[0,0]=}, {points[0,250]=}")

class PreprocAppBase(EigerRxBase):

    def compose(self):
        eiger_zmq_rx, pos_rx = super().compose()
        
        # Create operators
        batchsize = self.kwargs('img_batch_op')["batchsize"]
        img_batch_op = ImageBatchOp(self, batchsize=batchsize, name="img_batch_op")
        point_batch_op = PointBatchOp(self, batchsize=batchsize, name="point_batch_op")
        
        img_proc_op = ImagePreprocessorOp(self, **self.kwargs('img_proc_op'), name="img_proc_op")
        point_proc_op = PointPreprocessorOp(self, **self.kwargs('point_proc_op'), name="point_proc_op")
        print(self.kwargs('gather_op'))
        gather_op = DataGatherOp(self, **self.kwargs('gather_op'), name="gather_op")
        
        # Connect source operators to batch operators
        self.add_flow(eiger_zmq_rx, img_batch_op, {("image", "image"), ("image_index", "image_index")})
        self.add_flow(pos_rx, point_batch_op, {("point", "point"), ("point_index", "point_index")})
        
        # Connect batch operators to preprocessing operators with updated port names
        self.add_flow(img_batch_op, img_proc_op, {("image_batch", "image_batch"), ("image_indices", "image_indices_in")})
        self.add_flow(point_batch_op, point_proc_op, {("point_batch", "point_batch"), ("point_indices", "point_indices_in")})
        
        # Connect preprocessing operators to gather op
        self.add_flow(img_proc_op, gather_op, {("diff_amp", "diff_amp"), ("image_indices", "image_indices")})
        self.add_flow(point_proc_op, gather_op, {("processed_points", "points"), ("point_indices", "point_indices")})
        
        return eiger_zmq_rx, pos_rx, gather_op

class PreprocApp(PreprocAppBase):
    def compose(self):
        _, _, gather_op = super().compose()
        
        # Create N sinks and connect them to corresponding gather_op ports
        sinks = []
        for i in range(1, self.kwargs('gather_op')["num_parallel_streams"] + 1):
            sink = sink_func(self, name=f"sink{i}")
            sinks.append(sink)
            self.add_flow(gather_op, sink, {(f"batch{i}", "batch")})

if __name__ == "__main__":
    config = parse_args()
    
    app = PreprocApp()
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
                worker_thread_number=8,
                check_recession_period_ms=0.001,
                stop_on_deadlock=True,
                stop_on_deadlock_timeout=500,
                name="multithread_scheduler",
            )
    app.scheduler(scheduler)
    
    app.run()
    
    
