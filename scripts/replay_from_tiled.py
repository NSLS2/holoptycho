"""Replay a ptychography scan from tiled over ZMQ.

Reads diffraction frames and motor positions for a given scan from a Tiled
catalog, then publishes them over two ZMQ sockets mimicking the exact wire
formats of the Eiger detector and PandA box.

This is the recommended way to test holoptycho end-to-end without a live
beamline.  It can be run locally or from a remote machine via SSH tunnel.

Usage
-----
    pixi run -e replay python scripts/replay_from_tiled.py \\
        --scan-num 320045 \\
        --tiled-url https://tiled.nsls2.bnl.gov \\
        --tiled-api-key <key> \\
        --eiger-endpoint tcp://0.0.0.0:5555 \\
        --panda-endpoint tcp://0.0.0.0:5556 \\
        --rate 200

SSH tunnel (run on your local machine to expose the ZMQ ports):
    ssh -L 5555:localhost:5555 -L 5556:localhost:5556 <slurm-login-node>

Then point holoptycho at:
    SERVER_STREAM_SOURCE=tcp://localhost:5555
    PANDA_STREAM_SOURCE=tcp://localhost:5556

Environment variables (alternative to CLI flags):
    TILED_BASE_URL   — Tiled server URL
    TILED_API_KEY    — Tiled API key
    TILED_SCAN_NUM   — scan number to replay
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import numpy as np
import zmq
from tiled.client import from_uri


# ---------------------------------------------------------------------------
# Eiger wire format helpers
# ---------------------------------------------------------------------------

def _compress_bslz4(array: np.ndarray) -> bytes:
    """Compress a 2-D detector frame with bslz4 (bit-shuffle + LZ4)."""
    from dectris.compression import compress
    return compress(array.tobytes(), "bslz4", elem_size=array.dtype.itemsize)


def _eiger_encoding_msg(shape: tuple, dtype: np.dtype) -> bytes:
    """Build the dimage_d-1.0 encoding JSON frame."""
    dtype_map = {np.dtype("uint32"): "uint32", np.dtype("uint16"): "uint16"}
    encoding_map = {
        np.dtype("uint32"): "bs32-lz4<",
        np.dtype("uint16"): "bs16-lz4<",
    }
    return json.dumps({
        "htype": "dimage_d-1.0",
        "encoding": encoding_map.get(dtype, "bs32-lz4<"),
        "shape": [shape[1], shape[0]],  # Eiger reports [cols, rows]
        "type": dtype_map.get(dtype, "uint32"),
    }).encode()


def publish_eiger(
    frames: np.ndarray,
    endpoint: str,
    server_public_key: str,
    server_secret_key: str,
    client_public_key: str,
    rate_hz: float,
):
    """Publish Eiger frames over an encrypted CurveZMQ PUB socket.

    Parameters
    ----------
    frames:
        Array of shape [N, H, W] — one detector frame per scan point.
    endpoint:
        ZMQ bind address, e.g. ``tcp://0.0.0.0:5555``.
    server_public_key, server_secret_key:
        CurveZMQ keypair for this publisher (the "server").
    client_public_key:
        Public key of the holoptycho subscriber (the "client").
    rate_hz:
        Target frame rate in Hz.
    """
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.curve_publickey = server_public_key.encode("ascii")
    socket.curve_secretkey = server_secret_key.encode("ascii")
    socket.curve_server = True
    socket.bind(endpoint)

    # Brief pause to let subscribers connect
    time.sleep(0.5)

    interval = 1.0 / rate_hz
    n_frames, h, w = frames.shape
    dtype = frames.dtype

    print(f"[eiger] publishing {n_frames} frames at {rate_hz} Hz on {endpoint}")

    for frame_id, frame in enumerate(frames):
        header = json.dumps({"frame": frame_id, "series": 1}).encode()
        encoding = _eiger_encoding_msg((h, w), dtype)
        compressed = _compress_bslz4(frame)

        socket.send(header, zmq.SNDMORE)
        socket.send(encoding, zmq.SNDMORE)
        socket.send(compressed)

        time.sleep(interval)

    print("[eiger] done")
    socket.close()
    context.term()


def publish_panda(
    positions_x: list,
    positions_y: list,
    endpoint: str,
    ch1: str,
    ch2: str,
    rate_hz: float,
    points_per_message: int = 41,
):
    """Publish PandA position data over a plain ZMQ PUB socket.

    Parameters
    ----------
    positions_x, positions_y:
        Lists of motor encoder values (one per scan point).
    endpoint:
        ZMQ bind address, e.g. ``tcp://0.0.0.0:5556``.
    ch1, ch2:
        Channel names matching holoptycho's PositionRxOp configuration,
        e.g. ``/INENC2.VAL.Value`` and ``/INENC3.VAL.Value``.
    rate_hz:
        Target message rate in Hz.
    points_per_message:
        Number of encoder samples bundled per ZMQ message (matches PandA
        default of 41).
    """
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(endpoint)

    time.sleep(0.5)

    interval = 1.0 / rate_hz
    n_points = len(positions_x)

    # Send start message
    socket.send_json({
        "msg_type": "start",
        "arm_time": time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        "hw_time_offset_ns": None,
    })

    print(f"[panda]  publishing {n_points} positions at {rate_hz} Hz on {endpoint}")

    frame_number = 0
    for i in range(0, n_points, points_per_message):
        chunk_x = positions_x[i:i + points_per_message]
        chunk_y = positions_y[i:i + points_per_message]
        size = len(chunk_x)
        socket.send_json({
            "msg_type": "data",
            "frame_number": frame_number,
            "datasets": {
                ch1: {
                    "dtype": "float64",
                    "size": size,
                    "starting_sample_number": i,
                    "data": chunk_x,
                },
                ch2: {
                    "dtype": "float64",
                    "size": size,
                    "starting_sample_number": i,
                    "data": chunk_y,
                },
            },
        })
        frame_number += 1
        time.sleep(interval)

    socket.send_json({"msg_type": "stop", "emitted_frames": frame_number})
    print("[panda]  done")
    socket.close()
    context.term()


# ---------------------------------------------------------------------------
# Tiled data loading
# ---------------------------------------------------------------------------

def load_scan_from_tiled(
    tiled_url: str,
    api_key: str,
    scan_num: int | str,
) -> tuple[np.ndarray, list, list]:
    """Load diffraction frames and motor positions for a scan from tiled.

    Returns
    -------
    frames : np.ndarray, shape [N, H, W]
    positions_x : list of float
    positions_y : list of float
    """
    client = from_uri(tiled_url, api_key=api_key)

    # Navigate to the scan — adjust the path to match your catalog structure.
    # This placeholder traversal assumes the raw scan data lives at:
    #   <root>/raw/<scan_num>/frames  (array [N, H, W])
    #   <root>/raw/<scan_num>/positions_x  (array [N])
    #   <root>/raw/<scan_num>/positions_y  (array [N])
    # Update these paths to match your actual tiled catalog layout.
    scan_key = str(scan_num)
    try:
        scan_node = client["raw"][scan_key]
    except KeyError:
        print(
            f"ERROR: scan {scan_key!r} not found in tiled catalog at {tiled_url}.\n"
            "Update the path in load_scan_from_tiled() to match your catalog structure.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading frames for scan {scan_key} from tiled...")
    frames = scan_node["frames"].read()
    positions_x = scan_node["positions_x"].read().tolist()
    positions_y = scan_node["positions_y"].read().tolist()

    print(f"Loaded {len(frames)} frames, shape={frames.shape}, dtype={frames.dtype}")
    return frames, positions_x, positions_y


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a ptychography scan from tiled over ZMQ (Eiger + PandA wire formats).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scan-num",
        default=os.environ.get("TILED_SCAN_NUM"),
        required=not os.environ.get("TILED_SCAN_NUM"),
        help="Scan number to replay (or set TILED_SCAN_NUM env var)",
    )
    parser.add_argument(
        "--tiled-url",
        default=os.environ.get("TILED_BASE_URL", ""),
        help="Tiled server URL (or set TILED_BASE_URL env var)",
    )
    parser.add_argument(
        "--tiled-api-key",
        default=os.environ.get("TILED_API_KEY", ""),
        help="Tiled API key (or set TILED_API_KEY env var)",
    )
    parser.add_argument(
        "--eiger-endpoint",
        default="tcp://0.0.0.0:5555",
        help="ZMQ bind address for Eiger frames (default: tcp://0.0.0.0:5555)",
    )
    parser.add_argument(
        "--panda-endpoint",
        default="tcp://0.0.0.0:5556",
        help="ZMQ bind address for PandA positions (default: tcp://0.0.0.0:5556)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=200.0,
        help="Target frame/message rate in Hz (default: 200)",
    )
    parser.add_argument(
        "--eiger-server-public-key",
        default=os.environ.get("SERVER_PUBLIC_KEY", ""),
        help="CurveZMQ server public key (or set SERVER_PUBLIC_KEY env var)",
    )
    parser.add_argument(
        "--eiger-server-secret-key",
        default=os.environ.get("SERVER_SECRET_KEY", ""),
        help="CurveZMQ server secret key (or set SERVER_SECRET_KEY env var)",
    )
    parser.add_argument(
        "--eiger-client-public-key",
        default=os.environ.get("CLIENT_PUBLIC_KEY", ""),
        help="CurveZMQ client public key (or set CLIENT_PUBLIC_KEY env var)",
    )
    parser.add_argument(
        "--panda-ch1",
        default="/INENC2.VAL.Value",
        help="PandA x-axis channel name (default: /INENC2.VAL.Value)",
    )
    parser.add_argument(
        "--panda-ch2",
        default="/INENC3.VAL.Value",
        help="PandA y-axis channel name (default: /INENC3.VAL.Value)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.tiled_url:
        print("ERROR: --tiled-url or TILED_BASE_URL is required", file=sys.stderr)
        sys.exit(1)
    if not args.tiled_api_key:
        print("ERROR: --tiled-api-key or TILED_API_KEY is required", file=sys.stderr)
        sys.exit(1)

    # Validate CurveZMQ keys
    missing_keys = []
    if not args.eiger_server_public_key:
        missing_keys.append("--eiger-server-public-key / SERVER_PUBLIC_KEY")
    if not args.eiger_server_secret_key:
        missing_keys.append("--eiger-server-secret-key / SERVER_SECRET_KEY")
    if not args.eiger_client_public_key:
        missing_keys.append("--eiger-client-public-key / CLIENT_PUBLIC_KEY")
    if missing_keys:
        print(
            "ERROR: CurveZMQ keys required for Eiger socket:\n  " + "\n  ".join(missing_keys),
            file=sys.stderr,
        )
        sys.exit(1)

    # Load data from tiled
    frames, positions_x, positions_y = load_scan_from_tiled(
        tiled_url=args.tiled_url,
        api_key=args.tiled_api_key,
        scan_num=args.scan_num,
    )

    # Run Eiger and PandA publishers concurrently
    eiger_thread = threading.Thread(
        target=publish_eiger,
        args=(
            frames,
            args.eiger_endpoint,
            args.eiger_server_public_key,
            args.eiger_server_secret_key,
            args.eiger_client_public_key,
            args.rate,
        ),
        name="eiger-publisher",
    )
    panda_thread = threading.Thread(
        target=publish_panda,
        args=(
            positions_x,
            positions_y,
            args.panda_endpoint,
            args.panda_ch1,
            args.panda_ch2,
            args.rate,
        ),
        name="panda-publisher",
    )

    eiger_thread.start()
    panda_thread.start()

    eiger_thread.join()
    panda_thread.join()

    print("Replay complete.")


if __name__ == "__main__":
    main()
