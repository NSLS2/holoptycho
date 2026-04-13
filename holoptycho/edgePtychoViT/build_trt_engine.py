import argparse
import os

from .helper_trt import build_engine_from_onnx, save_engine


def _describe_engine(engine) -> str:
    import tensorrt as trt

    lines = []
    if hasattr(engine, "num_bindings"):
        lines.append(f"num_bindings={engine.num_bindings}")
        for binding_idx in range(engine.num_bindings):
            name = engine.get_binding_name(binding_idx)
            shape = tuple(engine.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding_idx)).__name__
            direction = "INPUT" if engine.binding_is_input(binding_idx) else "OUTPUT"
            lines.append(f"  [{binding_idx}] {direction} {name} shape={shape} dtype={dtype}")
        return "\n".join(lines)

    if hasattr(engine, "num_io_tensors"):
        lines.append(f"num_io_tensors={engine.num_io_tensors}")
        for tensor_idx in range(engine.num_io_tensors):
            name = engine.get_tensor_name(tensor_idx)
            shape = tuple(engine.get_tensor_shape(name))
            dtype = trt.nptype(engine.get_tensor_dtype(name)).__name__
            mode = engine.get_tensor_mode(name)
            direction = "INPUT" if int(mode) == int(trt.TensorIOMode.INPUT) else "OUTPUT"
            lines.append(f"  [{tensor_idx}] {direction} {name} shape={shape} dtype={dtype}")
        return "\n".join(lines)

    return "Unable to describe engine (unknown TensorRT API)."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a TensorRT .engine (plan) from an ONNX model."
    )
    parser.add_argument("--onnx", required=True, help="Path to ONNX model.")
    parser.add_argument(
        "--engine", required=True, help="Output TensorRT engine path (.engine)."
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU index to use.")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 builder flag.")
    parser.add_argument(
        "--no-tf32",
        action="store_true",
        help="Disable TF32 builder flag (enabled by default).",
    )
    parser.add_argument(
        "--workspace-gb",
        type=float,
        default=1.0,
        help="TensorRT workspace size in GiB (default: 1.0).",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import pycuda.autoinit  # noqa: F401

    max_workspace = int(args.workspace_gb * (1 << 30))
    engine = build_engine_from_onnx(
        args.onnx,
        fp16=bool(args.fp16),
        tf32=not bool(args.no_tf32),
        max_workspace_size_bytes=max_workspace,
    )
    save_engine(engine, args.engine)
    print(f"Wrote engine to {os.path.abspath(args.engine)}")
    print(_describe_engine(engine))


if __name__ == "__main__":
    main()

