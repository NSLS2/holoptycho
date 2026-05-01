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
    get_stream,
    lookup_run,
    open_tiled_node,
)


# ---------------------------------------------------------------------------
# Eiger wire format helpers
# ---------------------------------------------------------------------------

def _encode_frame(array: np.ndarray, *, no_compress: bool) -> bytes:
    """Encode a 2-D detector frame for the dimage_d-1.0 wire format.

    With ``no_compress=True`` the payload is the raw frame bytes (paired with
    the ``"raw"`` encoding header); the receiver reshapes the bytes directly
    without invoking the dectris decompressor. With ``no_compress=False`` the
    frame is bslz4-compressed to match the live Eiger wire format.
    """
    if no_compress:
        return np.ascontiguousarray(array).tobytes()

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


def _eiger_encoding_msg(shape: tuple, dtype: np.dtype, *, no_compress: bool) -> bytes:
    """Build the dimage_d-1.0 encoding JSON frame."""
    dtype_map = {np.dtype("uint32"): "uint32", np.dtype("uint16"): "uint16"}
    if no_compress:
        encoding = "raw"
    else:
        encoding_map = {
            np.dtype("uint32"): "bs32-lz4<",
            np.dtype("uint16"): "bs16-lz4<",
        }
        encoding = encoding_map.get(dtype, "bs32-lz4<")
    return json.dumps({
        "htype": "dimage_d-1.0",
        "encoding": encoding,
        "shape": [shape[1], shape[0]],  # Eiger reports [cols, rows]
        "type": dtype_map.get(dtype, "uint32"),
    }).encode()


def publish_eiger(
    chunks_iter,
    n_frames: int,
    frame_shape: tuple,
    frame_dtype: np.dtype,
    endpoint: str,
    server_public_key: str,
    server_secret_key: str,
    client_public_key: str,
    rate_hz: float,
    no_compress: bool = False,
):
    """Publish Eiger frames over a ZMQ PUB socket.

    Frames are pulled from ``chunks_iter`` lazily — each iteration yields a
    ``(M, H, W)`` ndarray fetched from tiled. Switching to chunked iteration
    avoids holding the whole 5 GB scan in memory and lets the publisher
    start streaming before the entire scan has been transferred from tiled.

    Parameters
    ----------
    chunks_iter:
        Iterator yielding ``(M, H, W)`` ndarrays of detector frames.
    n_frames:
        Total number of frames the iterator will yield (for logging).
    frame_shape:
        ``(H, W)`` of every frame — used to build the encoding header.
    frame_dtype:
        dtype of every frame — used to build the encoding header.
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
    encoding = _eiger_encoding_msg(frame_shape, frame_dtype, no_compress=no_compress)

    mode = "raw" if no_compress else "bslz4"
    print(f"[eiger] publishing {n_frames} frames at {rate_hz} Hz on {endpoint} ({mode})", flush=True)

    frame_id = 0
    for chunk in chunks_iter:
        for frame in chunk:
            header = json.dumps({"frame": frame_id, "series": 1}).encode()
            payload = _encode_frame(frame, no_compress=no_compress)
            socket.send(header, zmq.SNDMORE)
            socket.send(encoding, zmq.SNDMORE)
            socket.send(payload)
            time.sleep(interval)
            frame_id += 1

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

def _fetch_frame_chunk(
    frames_node, frame_axis: int, start: int, stop: int
) -> np.ndarray:
    """Fetch a (stop - start, H, W) chunk via tiled server-side slicing.

    Squeezes out the spurious unit axis used by eiger2 ``(1, N, H, W)`` and
    eiger1 ``(N, 1, H, W)`` layouts so the caller always sees ``(M, H, W)``.
    """
    if frame_axis == 0:
        chunk = np.asarray(frames_node[start:stop])
    else:
        chunk = np.asarray(frames_node[:, start:stop])
    if chunk.ndim == 4 and chunk.shape[0] == 1:
        chunk = chunk[0]
    elif chunk.ndim == 4 and chunk.shape[1] == 1:
        chunk = chunk[:, 0]
    return chunk


def setup_scan_from_tiled(
    tiled_url: str,
    run_uid: str,
    api_key: str = "",
    max_frames: int | None = None,
    skip_frames: int = 0,
    autodetect_frames: int = 50,
) -> dict:
    """Open tiled handles for a run and load just the data needed up front.

    Authentication is taken from the tiled credential cache (run ``tiled login``
    before calling this script).  Pass ``api_key`` only if you want to override.

    Eagerly loads only:
      * the encoder arrays (small — a few hundred KB even for 10k frames), and
      * the first ``autodetect_frames`` detector frames so the caller can run
        ``_auto_batch_offsets`` before publishing starts.

    The remaining frames are streamed chunk-by-chunk during publishing via
    ``_fetch_frame_chunk(frames_node, frame_axis, start, stop)``.

    Returns
    -------
    dict with keys: frames_node, frame_axis, start, end, n_frames, frame_shape,
    frame_dtype, head_frames, positions_x, positions_y.
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

    stream = get_stream(run, "primary")

    # Detector name varies by scan: eiger2 used the explicit "_image" suffix,
    # eiger1 puts the image array under the bare detector name.
    detector_candidates = ("eiger2_image", "eiger1_image", "eiger2", "eiger1")
    detector_key = next((k for k in detector_candidates if k in list(stream)), None)
    if detector_key is None:
        raise KeyError(
            f"No known eiger detector found in stream. Tried: {detector_candidates}. "
            f"Available keys: {list(stream)[:30]}"
        )

    x_candidates = ("inenc2_val", "zpssx")
    y_candidates = ("inenc3_val", "zpssy")
    x_key = next((k for k in x_candidates if k in list(stream)), None)
    y_key = next((k for k in y_candidates if k in list(stream)), None)
    if x_key is None or y_key is None:
        raise KeyError(
            f"No known motor encoders found. Tried x={x_candidates} y={y_candidates}. "
            f"Available keys: {list(stream)[:30]}"
        )

    slice_msg = f", first {max_frames}" if max_frames else ""
    print(
        f"Setting up streamed replay for scan {scan_num} ({run_uid}) "
        f"[detector={detector_key}, x={x_key}, y={y_key}{slice_msg}]...",
        flush=True,
    )
    frames_node = stream[detector_key]
    shape = tuple(frames_node.shape)
    if len(shape) == 4 and shape[0] == 1:
        frame_axis, n_total = 1, shape[1]
        frame_h, frame_w = shape[2], shape[3]
    elif len(shape) == 4:
        frame_axis, n_total = 0, shape[0]
        frame_h, frame_w = shape[2], shape[3]
    else:
        frame_axis, n_total = 0, shape[0]
        frame_h, frame_w = shape[1], shape[2]
    frame_dtype = frames_node.dtype

    if skip_frames < 0:
        raise ValueError(f"skip_frames must be >= 0, got {skip_frames}")
    if skip_frames >= n_total:
        raise ValueError(f"skip_frames={skip_frames} >= n_total={n_total}; nothing to publish")
    end_frame = min(skip_frames + max_frames, n_total) if max_frames is not None else n_total
    n_publish = end_frame - skip_frames

    # Pull the first chunk eagerly so auto-batch detection can run before
    # the holoptycho pipeline starts. Cap to ``autodetect_frames`` or the
    # size of the publish window, whichever is smaller.
    head_n = min(autodetect_frames, n_publish)
    print(f"  ... fetching head ({head_n} frames) for autodetect", flush=True)
    head_frames = _fetch_frame_chunk(frames_node, frame_axis, skip_frames, skip_frames + head_n)

    positions_x = np.asarray(stream[x_key].read()).ravel().tolist()
    positions_y = np.asarray(stream[y_key].read()).ravel().tolist()
    upsample = max(1, len(positions_x) // n_total)
    pos_start = skip_frames * upsample
    pos_end = end_frame * upsample
    positions_x = positions_x[pos_start:pos_end]
    positions_y = positions_y[pos_start:pos_end]

    if skip_frames or max_frames is not None:
        print(f"Skipping first {skip_frames} frames; will publish {n_publish} frames "
              f"(scan-orig indices {skip_frames}..{end_frame - 1}).", flush=True)
    print(f"Will stream {n_publish} frames, shape=({frame_h}, {frame_w}), dtype={frame_dtype}", flush=True)
    return {
        "frames_node": frames_node,
        "frame_axis": frame_axis,
        "start": skip_frames,
        "end": end_frame,
        "n_frames": n_publish,
        "frame_shape": (frame_h, frame_w),
        "frame_dtype": frame_dtype,
        "head_frames": head_frames,
        "positions_x": positions_x,
        "positions_y": positions_y,
    }


def iter_frame_chunks(
    streams: dict, head_frames: np.ndarray, chunk_size: int
):
    """Yield chunks of detector frames; tiled fetches run in a worker thread.

    Tiled HTTPS read tops out around 1-2 MB/s while ZMQ publish at 200 Hz
    drains a 32 MB chunk in ~1.3 s. Without overlap the publisher idles
    ~17 s between bursts waiting on each fetch. A worker thread pushes
    chunks into a ``Queue(maxsize=1)``: the worker is always trying to
    stage chunk N+1 while the consumer is iterating chunk N, and ``put``
    blocks on the bounded queue when the consumer hasn't drained the
    previous chunk yet.

    ``head_frames`` is the eager auto-detect read from
    :func:`setup_scan_from_tiled`; we yield it as-is to avoid re-reading
    the same range from tiled.
    """
    import queue
    import threading

    head_end = streams["start"] + len(head_frames)
    end = streams["end"]
    frames_node = streams["frames_node"]
    frame_axis = streams["frame_axis"]

    q: "queue.Queue" = queue.Queue(maxsize=1)
    sentinel = object()

    def _worker():
        try:
            q.put(head_frames)
            for s in range(head_end, end, chunk_size):
                stop = min(s + chunk_size, end)
                q.put(_fetch_frame_chunk(frames_node, frame_axis, s, stop))
        except Exception as e:
            q.put(("__error__", e))
        else:
            q.put(sentinel)

    threading.Thread(target=_worker, daemon=True, name="tiled-fetcher").start()

    while True:
        item = q.get()
        if item is sentinel:
            return
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "__error__":
            raise item[1]
        yield item


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
    # Retry-with-backoff for the brief window where a prior runner thread is
    # still finalizing — /status flips to "stopped" before the thread is fully
    # joined, so /run or /restart can race and return 400.
    for attempt in range(6):
        try:
            result = _json_request(f"{hp_url}{endpoint}", method="POST", payload={"config": config})
            break
        except RuntimeError as exc:
            if "still shutting down" in str(exc) and attempt < 5:
                time.sleep(2.0)
                continue
            raise
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
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Truncate the scan to the first N frames (handy for quick tests on big scans)",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Drop the first N Eiger frames (and aligned encoder samples) before "
             "publishing. Useful when a scan has a settling/ramp-up region whose "
             "encoder readings overshoot the commanded scan range and crash the "
             "iterative recon.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Number of frames per tiled fetch during streaming (default: 256). "
             "Smaller = lower latency before publishing starts and lower peak "
             "memory; larger = fewer round-trips. The publisher fetches the "
             "next chunk between rate-limited frame sends, so as long as a "
             "chunk's fetch (~hundreds of ms) is shorter than the publish "
             "interval (chunk_size / rate_hz) the stream stays smooth.",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Skip bslz4 compression and publish raw frame bytes with a "
             "'raw' encoding header. Avoids the ~30s/batch Python "
             "bitshuffle bottleneck (dectris-compression 0.3.1 dropped "
             "the C compress entrypoint, so the script falls back to a "
             "pure-Python implementation otherwise). The receiver inside "
             "holoptycho recognises 'raw' and reshapes the bytes directly.",
    )
    return parser.parse_args()


def _auto_batch_offsets(frames: np.ndarray, nx: int, ny: int) -> tuple[int, int]:
    """Auto-detect detector ROI offsets from the diffraction pattern center.

    Averages the first ~50 frames, masks out uint16-saturated pixels (which
    can be hot pixels or detector artifacts that drag the center of mass off
    course), then computes the intensity-weighted center of mass and derives
    ``(batch_x0, batch_y0)`` so that an ``nx × ny`` crop is centered on it.

    Verified on scan 404611: target was (135, 70), detected (137, 68) — 2px
    rounding noise after sat masking.
    """
    sample = frames[:min(50, len(frames))].astype(np.float64)
    mean_frame = sample.mean(axis=0)
    sat_mask = (sample == np.iinfo(frames.dtype).max).any(axis=0)
    masked = np.where(sat_mask, 0.0, mean_frame)
    total = masked.sum()
    if total <= 0:
        # Frames are all zero or all saturated — bail to (0, 0) and let the
        # user override.
        return 0, 0
    ys, xs = np.indices(masked.shape)
    cy = float((ys * masked).sum() / total)
    cx = float((xs * masked).sum() / total)
    h, w = mean_frame.shape
    bx0 = max(0, min(w - nx, round(cx - nx / 2)))
    by0 = max(0, min(h - ny, round(cy - ny / 2)))
    return int(bx0), int(by0)


def main():
    args = parse_args()

    if not args.tiled_url:
        print("ERROR: --tiled-url or TILED_BASE_URL is required", file=sys.stderr)
        sys.exit(1)

    # Open tiled handles + eagerly load only the head frames + encoder
    # arrays. The remaining frames are pulled chunk-by-chunk during publish.
    streams = setup_scan_from_tiled(
        tiled_url=args.tiled_url,
        api_key=args.tiled_api_key,
        run_uid=args.uid,
        max_frames=args.max_frames,
        skip_frames=args.skip_frames,
    )

    if args.batch_x0 is None or args.batch_y0 is None:
        auto_x0, auto_y0 = _auto_batch_offsets(streams["head_frames"], args.nx, args.ny)
        if args.batch_x0 is None:
            args.batch_x0 = auto_x0
        if args.batch_y0 is None:
            args.batch_y0 = auto_y0
        h, w = streams["frame_shape"]
        print(
            f"[holoptycho] auto-detected diffraction center: "
            f"batch_x0={args.batch_x0}, batch_y0={args.batch_y0} "
            f"(box {args.nx}x{args.ny} on {h}x{w} detector)",
            flush=True,
        )

    if args.hp_start:
        start_holoptycho_pipeline(args)

    # Run Eiger and PandA publishers concurrently. The Eiger thread pulls
    # chunks from tiled lazily — first chunk is the already-fetched head,
    # subsequent chunks are server-side-sliced via _fetch_frame_chunk.
    chunks_iter = iter_frame_chunks(streams, streams["head_frames"], args.chunk_size)
    eiger_thread = threading.Thread(
        target=publish_eiger,
        args=(
            chunks_iter,
            streams["n_frames"],
            streams["frame_shape"],
            streams["frame_dtype"],
            args.eiger_endpoint,
            args.eiger_server_public_key,
            args.eiger_server_secret_key,
            args.eiger_client_public_key,
            args.rate,
            args.no_compress,
        ),
        name="eiger-publisher",
    )
    panda_thread = threading.Thread(
        target=publish_panda,
        args=(
            streams["positions_x"],
            streams["positions_y"],
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
