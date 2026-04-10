import numpy as np


def _format_cuda_version(raw: int) -> str:
    major = raw // 1000
    minor = (raw % 1000) // 10
    return f"{major}.{minor}"


def _try_get_cuda_driver_version() -> int | None:
    import ctypes

    for lib in ("libcuda.so.1", "libcuda.so"):
        try:
            libcuda = ctypes.CDLL(lib)
        except OSError:
            continue
        ver = ctypes.c_int()
        try:
            rc = libcuda.cuDriverGetVersion(ctypes.byref(ver))
        except Exception:
            return None
        if int(rc) == 0:
            return int(ver.value)
    return None


def _try_get_cuda_runtime_version() -> int | None:
    import ctypes

    for lib in ("libcudart.so", "libcudart.so.12", "libcudart.so.13"):
        try:
            libcudart = ctypes.CDLL(lib)
        except OSError:
            continue
        ver = ctypes.c_int()
        try:
            rc = libcudart.cudaRuntimeGetVersion(ctypes.byref(ver))
        except Exception:
            return None
        if int(rc) == 0:
            return int(ver.value)
    return None


def _trt_init_hint() -> str:
    import os

    parts = []
    drv = _try_get_cuda_driver_version()
    rt = _try_get_cuda_runtime_version()
    if drv is not None:
        parts.append(f"CUDA driver version: {_format_cuda_version(drv)} ({drv})")
    if rt is not None:
        parts.append(f"CUDA runtime version: {_format_cuda_version(rt)} ({rt})")
    if drv is not None and rt is not None and rt > drv:
        parts.append(
            "Detected CUDA runtime > driver. This commonly means a CUDA 13.x TensorRT wheel was installed on a CUDA 12.x driver."
        )
        parts.append(
            "Fix: install the TensorRT wheel matching your driver (e.g. `tensorrt-cu12` for CUDA 12.x drivers), or upgrade the NVIDIA driver."
        )
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None:
        parts.append(f"CUDA_VISIBLE_DEVICES={vis!r}")
    return "\n".join(parts)


def load_engine(engine_path: str):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    try:
        runtime = trt.Runtime(logger)
    except TypeError as exc:
        hint = _trt_init_hint()
        msg = "TensorRT Runtime() failed to initialize CUDA."
        if hint:
            msg += "\n" + hint
        raise RuntimeError(msg) from exc
    with open(engine_path, "rb") as f:
        engine_bytes = f.read()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine from {engine_path}.")
    return engine


def save_engine(engine, engine_path: str):
    import os

    serialized = engine.serialize()
    try:
        engine_bytes = bytes(serialized)
    except TypeError:
        engine_bytes = serialized

    out_dir = os.path.dirname(os.path.abspath(engine_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_path = engine_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(engine_bytes)
    os.replace(tmp_path, engine_path)


def build_engine_from_onnx(
    onnx_model_path: str,
    *,
    fp16: bool = False,
    tf32: bool = True,
    max_workspace_size_bytes: int = 1 << 30,
):
    import tensorrt as trt

    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    logger = trt.Logger(trt.Logger.ERROR)
    try:
        builder = trt.Builder(logger)
    except TypeError as exc:
        hint = _trt_init_hint()
        msg = "TensorRT Builder() failed to initialize CUDA."
        if hint:
            msg += "\n" + hint
        raise RuntimeError(msg) from exc
    config = builder.create_builder_config()
    workspace_bytes = int(max_workspace_size_bytes)
    if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        except Exception:
            pass
    if hasattr(config, "max_workspace_size"):
        try:
            config.max_workspace_size = workspace_bytes
        except Exception:
            pass

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if tf32:
        config.set_flag(trt.BuilderFlag.TF32)

    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(onnx_model_path)
    if not success:
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(
            "Failed to parse ONNX model with TensorRT:\n" + "\n".join(errors)
        )

    engine = None
    if hasattr(builder, "build_engine"):
        engine = builder.build_engine(network, config)
    if engine is None and hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT failed to build a serialized network.")
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("TensorRT failed to build an engine from the ONNX model.")
    return engine


def allocate_io_buffers(engine):
    """Allocate I/O buffers for TensorRT engine."""
    import pycuda.driver as cuda
    import tensorrt as trt

    inputs = []
    outputs = []
    stream = cuda.Stream()

    # TensorRT 8.x: bindings API
    if hasattr(engine, "num_bindings"):
        bindings = []
        for binding_idx in range(engine.num_bindings):
            binding_name = engine.get_binding_name(binding_idx)
            dtype = trt.nptype(engine.get_binding_dtype(binding_idx))
            shape = tuple(engine.get_binding_shape(binding_idx))
            if any(dim < 0 for dim in shape):
                raise ValueError(
                    f"Dynamic shapes are not supported by this helper; binding {binding_name} has shape {shape}."
                )
            n_elements = int(trt.volume(shape))
            host_mem = cuda.pagelocked_empty(n_elements, dtype=dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            io = {
                "name": binding_name,
                "index": binding_idx,
                "shape": shape,
                "dtype": dtype,
                "host": host_mem,
                "device": device_mem,
            }
            if engine.binding_is_input(binding_idx):
                inputs.append(io)
            else:
                outputs.append(io)
        return inputs, outputs, bindings, stream

    # TensorRT 9+/10+: I/O tensors API
    if hasattr(engine, "num_io_tensors"):
        bindings = {}
        for tensor_idx in range(engine.num_io_tensors):
            name = engine.get_tensor_name(tensor_idx)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            shape = tuple(engine.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise ValueError(
                    f"Dynamic shapes are not supported by this helper; tensor {name} has shape {shape}."
                )
            n_elements = int(trt.volume(shape))
            host_mem = cuda.pagelocked_empty(n_elements, dtype=dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings[name] = int(device_mem)
            io = {
                "name": name,
                "index": tensor_idx,
                "shape": shape,
                "dtype": dtype,
                "host": host_mem,
                "device": device_mem,
            }
            mode = engine.get_tensor_mode(name)
            if int(mode) == int(trt.TensorIOMode.INPUT):
                inputs.append(io)
            else:
                outputs.append(io)
        return inputs, outputs, bindings, stream

    raise RuntimeError(
        "Unsupported TensorRT engine API: no bindings or I/O tensor accessors found."
    )


def infer(engine_context, inputs, outputs, bindings, stream, cuda_context=None):
    """Run inference with TensorRT engine."""
    import pycuda.driver as cuda

    # Ensure CUDA context is active for this thread
    if cuda_context is not None:
        cuda_context.push()

    try:
        # Use synchronous memcpy
        for io in inputs:
            cuda.memcpy_htod(int(io["device"]), io["host"])

        if isinstance(bindings, dict):
            if not hasattr(engine_context, "set_tensor_address"):
                raise RuntimeError(
                    "TensorRT context does not support set_tensor_address(); cannot run I/O tensor inference."
                )
            for name, addr in bindings.items():
                engine_context.set_tensor_address(name, addr)

            if hasattr(engine_context, "execute_async_v3"):
                try:
                    engine_context.execute_async_v3(stream_handle=stream.handle)
                except TypeError:
                    engine_context.execute_async_v3(stream.handle)
            else:
                raise RuntimeError(
                    "TensorRT context does not expose execute_async_v3(); cannot run I/O tensor inference."
                )
        else:
            engine_context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Wait for inference to complete before copying results
        stream.synchronize()

        for io in outputs:
            cuda.memcpy_dtoh(io["host"], int(io["device"]))

        return [io["host"] for io in outputs]
    finally:
        if cuda_context is not None:
            cuda_context.pop()


def reshape_output_flat(
    flat: np.ndarray,
    *,
    batch_size: int,
    height: int,
    width: int,
):
    expected_single = batch_size * height * width
    expected_dual = batch_size * 2 * height * width

    if flat.size == expected_single:
        return flat.reshape(batch_size, height, width)
    if flat.size == expected_dual:
        return flat.reshape(batch_size, 2, height, width)
    raise ValueError(
        f"Unexpected output size: got {flat.size}, expected {expected_single} or {expected_dual} elements."
    )
