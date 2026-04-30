"""Replay a ptychography run from tiled over ZMQ.

Reads diffraction frames and motor positions for a given run UID from a Tiled
catalog, then publishes them over two ZMQ sockets mimicking the exact wire
formats of the Eiger detector and PandA box.

This is the recommended way to test holoptycho end-to-end without a live
beamline.  It can be run locally or from a remote machine via SSH tunnel.

Usage
-----
    tiled login https://tiled.nsls2.bnl.gov
    pixi run -e replay python scripts/replay_from_tiled.py \\
        --uid 67e77251-cbe4-444c-8a8c-36491b0b9100 \\
        --tiled-url https://tiled.nsls2.bnl.gov/hxn/migration \\
        --eiger-endpoint tcp://0.0.0.0:5555 \\
        --panda-endpoint tcp://0.0.0.0:5556 \\
        --rate 200

    To have the replay script configure and start holoptycho before publishing,
    add --hp-start. It will build the run config from the same run metadata
    and POST it to the holoptycho API before replay begins.

SSH tunnel (run on your local machine to expose the ZMQ ports):
    ssh -L 5555:localhost:5555 -L 5556:localhost:5556 <slurm-login-node>

Then point holoptycho at:
    SERVER_STREAM_SOURCE=tcp://localhost:5555
    PANDA_STREAM_SOURCE=tcp://localhost:5556

Environment variables (alternative to CLI flags):
    TILED_BASE_URL   — Tiled server URL
    TILED_API_KEY    — Tiled API key
    TILED_RUN_UID    — run UID to replay
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import zmq
from config_from_tiled import (
    add_reconstruction_arguments,
    build_full_config,
    lookup_run,
    open_tiled_node,
)


# ---------------------------------------------------------------------------
# Eiger wire format helpers
# ---------------------------------------------------------------------------

def _compress_bslz4(array: np.ndarray) -> bytes:
    """Compress a 2-D detector frame with bslz4 (bit-shuffle + LZ4)."""
    try:
        from dectris.compression import compress
    except ImportError:
        import bitshuffle

        flat = np.ascontiguousarray(array).ravel()
        block_elems = (flat.size // 8) * 8
        if block_elems <= 0:
            raise RuntimeError(
                "Frame is too small for bslz4 fallback compression; "
                "need at least 8 elements."
            )
        payload = bytes(bitshuffle.compress_lz4(flat, block_size=block_elems))
        header = struct.pack(
            ">QI",
            flat.nbytes,
            block_elems * flat.dtype.itemsize,
        )
        return header + payload

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
    """Publish Eiger frames over a ZMQ PUB socket.

    Parameters
    ----------
    frames:
        Array of shape [N, H, W] — one detector frame per scan point.
    endpoint:
        ZMQ bind address, e.g. ``tcp://0.0.0.0:5555``.
    server_public_key, server_secret_key, client_public_key:
        If all three are provided, enable CurveZMQ for the Eiger publisher.
        If all are empty, publish over plain ZMQ.
    rate_hz:
        Target frame rate in Hz.
    """
    context = zmq.Context()
    socket = context.socket(zmq.PUB)

    auth_values = {
        "SERVER_PUBLIC_KEY": server_public_key,
        "SERVER_SECRET_KEY": server_secret_key,
        "CLIENT_PUBLIC_KEY": client_public_key,
    }
    configured = {name: value for name, value in auth_values.items() if value}

    if configured and len(configured) != len(auth_values):
        missing = [name for name, value in auth_values.items() if not value]
        raise RuntimeError(
            "Incomplete Eiger ZMQ auth configuration; set all of "
            f"{', '.join(auth_values)} or leave them all unset. Missing: {', '.join(missing)}"
        )

    if len(configured) == len(auth_values):
        socket.curve_publickey = server_public_key.encode("ascii")
        socket.curve_secretkey = server_secret_key.encode("ascii")
        socket.curve_server = True

    socket.bind(endpoint)

    # Brief pause to let subscribers connect
    time.sleep(0.5)

    interval = 1.0 / rate_hz
    n_frames, h, w = frames.shape
    dtype = frames.dtype

    print(f"[eiger] publishing {n_frames} frames at {rate_hz} Hz on {endpoint}", flush=True)

    for frame_id, frame in enumerate(frames):
        header = json.dumps({"frame": frame_id, "series": 1}).encode()
        encoding = _eiger_encoding_msg((h, w), dtype)
        compressed = _compress_bslz4(frame)

        socket.send(header, zmq.SNDMORE)
        socket.send(encoding, zmq.SNDMORE)
        socket.send(compressed)

        time.sleep(interval)

    print("[eiger] done", flush=True)
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

    print(f"[panda]  publishing {n_points} positions at {rate_hz} Hz on {endpoint}", flush=True)

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
    print("[panda]  done", flush=True)
    socket.close()
    context.term()


# ---------------------------------------------------------------------------
# Tiled data loading
# ---------------------------------------------------------------------------

def load_scan_from_tiled(
    tiled_url: str,
    run_uid: str,
    api_key: str = "",
) -> tuple[np.ndarray, list, list]:
    """Load diffraction frames and motor positions for a run from tiled.

    Authentication is taken from the tiled credential cache (run ``tiled login``
    before calling this script).  Pass ``api_key`` only if you want to override.

    Returns
    -------
    frames : np.ndarray, shape [N, H, W]
    positions_x : list of float
    positions_y : list of float
    """
    if api_key:
        print(
            "WARNING: --tiled-api-key is currently ignored when --tiled-url points at a catalog path; "
            "use 'tiled login' cached credentials instead.",
            file=sys.stderr,
        )

    client = open_tiled_node(tiled_url)
    run = lookup_run(client, run_uid, tiled_url)
    scan_num = run.metadata.get("start", {}).get("scan_id", run_uid)

    print(f"Loading frames for scan {scan_num} ({run_uid}) from tiled...", flush=True)
    frames = run["primary"]["eiger2_image"].read()
    if frames.ndim == 4 and frames.shape[0] == 1:
        frames = frames[0]
    positions_x = run["primary"]["inenc2_val"].read().tolist()
    positions_y = run["primary"]["inenc3_val"].read().tolist()

    print(f"Loaded {len(frames)} frames, shape={frames.shape}, dtype={frames.dtype}", flush=True)
    return frames, positions_x, positions_y


def _json_request(url: str, method: str = "GET", payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8")) if resp.readable() else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("detail", body)
        except json.JSONDecodeError:
            detail = body
        raise RuntimeError(f"holoptycho API error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to reach holoptycho API at {url}: {exc.reason}") from exc


def start_holoptycho_pipeline(args) -> None:
    """Start or restart the holoptycho pipeline with config from the same run."""
    hp_url = args.hp_url.rstrip("/")
    try:
        _json_request(f"{hp_url}/logs/clear", method="POST")
        print("[holoptycho] logs cleared", flush=True)
    except RuntimeError as exc:
        print(f"[holoptycho] WARNING: failed to clear logs: {exc}", flush=True)
    config_tiled_url = args.hp_config_tiled_url or args.tiled_url
    config = build_full_config(args.uid, tiled_url=config_tiled_url, args=args)
    status = _json_request(f"{hp_url}/status")
    endpoint = "/restart" if status.get("status") in ("starting", "running", "finished", "error") else "/run"
    result = _json_request(f"{hp_url}{endpoint}", method="POST", payload={"config": config})
    print(f"[holoptycho] {result.get('detail', 'pipeline request submitted')}", flush=True)
    if args.hp_startup_wait > 0:
        time.sleep(args.hp_startup_wait)


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
        "--uid",
        default=os.environ.get("TILED_RUN_UID"),
        required=not os.environ.get("TILED_RUN_UID"),
        help="Run UID to replay (or set TILED_RUN_UID env var)",
    )
    parser.add_argument(
        "--tiled-url",
        default=os.environ.get("TILED_BASE_URL", ""),
        help="Tiled catalog URL containing run entries, e.g. https://tiled.nsls2.bnl.gov/hxn/migration",
    )
    parser.add_argument(
        "--tiled-api-key",
        default=os.environ.get("TILED_API_KEY", ""),
        help="Tiled API key (optional — uses cached credentials from 'tiled login' if omitted)",
    )
    parser.add_argument(
        "--hp-start",
        action="store_true",
        help="Start or restart holoptycho via its API using config from the same run before replaying",
    )
    parser.add_argument(
        "--hp-url",
        default=os.environ.get("HOLOPTYCHO_URL", "http://localhost:8000"),
        help="holoptycho API base URL for --hp-start (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--hp-config-tiled-url",
        default=os.environ.get("HP_CONFIG_TILED_URL", os.environ.get("TILED_CONFIG_BASE_URL", "")),
        help="Tiled URL used to build the hp config for --hp-start; defaults to --tiled-url",
    )
    parser.add_argument(
        "--hp-startup-wait",
        type=float,
        default=2.0,
        help="Seconds to wait after --hp-start before publishing ZMQ (default: 2.0)",
    )
    add_reconstruction_arguments(parser)
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
        help="CurveZMQ server public key for encrypted Eiger PUB (optional)",
    )
    parser.add_argument(
        "--eiger-server-secret-key",
        default=os.environ.get("SERVER_SECRET_KEY", ""),
        help="CurveZMQ server secret key for encrypted Eiger PUB (optional)",
    )
    parser.add_argument(
        "--eiger-client-public-key",
        default=os.environ.get("CLIENT_PUBLIC_KEY", ""),
        help="CurveZMQ client public key for encrypted Eiger PUB (optional)",
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

    if args.hp_start:
        start_holoptycho_pipeline(args)

    # Load data from tiled
    frames, positions_x, positions_y = load_scan_from_tiled(
        tiled_url=args.tiled_url,
        api_key=args.tiled_api_key,
        run_uid=args.uid,
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

    print("Replay complete.", flush=True)


if __name__ == "__main__":
    main()
