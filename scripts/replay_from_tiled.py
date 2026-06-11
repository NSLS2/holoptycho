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

    By default the replay script configures and starts holoptycho via its API
    using config built from the same run metadata. Pass --no-hp-start when
    holoptycho is already running with a config that matches the scan.

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
    load_config_from_tiled,
    lookup_run,
    lookup_uid_by_scan_id,
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
    prebound_socket=None,
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
        Logged target frame rate in Hz, but no longer enforced per-frame.
        Each fetched chunk is dumped to ZMQ as fast as the SNDHWM and the
        SUB-side pipeline allow; the receiver's HWM is what absorbs the
        burst. Tightly pacing per-frame here just made the publisher block
        on tiled fetches because the chunk drain rate had to be slower
        than the chunk fetch rate, which reintroduced the SUB-side dead
        zones we already chased down.
    """
    if prebound_socket is not None:
        # Caller bound the socket before starting the pipeline so the SUB
        # could connect during compose(). Use it directly — no bind/sleep needed.
        socket = prebound_socket
        context = None
        _own_socket = False
    else:
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        # Bump send HWM well above the default 1000 so the publisher can dump
        # a fetched chunk into ZMQ in one burst without waiting for the
        # subscriber to drain. At ~128 KB per Eiger frame, 20000 = ~2.6 GB
        # peak in the kernel/zmq buffers — comfortable for a dev box and
        # enough to hold a full HXN scan (10000 frames) with margin. Above
        # SNDHWM, frames get dropped at the PUB side, which is at least
        # predictable (and visible to the subscriber via missing frame_id
        # gaps) — far better than the silent stalls we had with rate-pacing
        # racing tiled fetch latency.
        socket.setsockopt(zmq.SNDHWM, 20000)

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
        _own_socket = True

    encoding = _eiger_encoding_msg(frame_shape, frame_dtype, no_compress=no_compress)

    mode = "raw" if no_compress else "bslz4"
    print(
        f"[eiger] publishing {n_frames} frames (target {rate_hz} Hz; "
        f"dumping at ZMQ-bound rate) on {endpoint} ({mode})",
        flush=True,
    )

    frame_id = 0
    diag_window_start = time.perf_counter()
    diag_frames_in_window = 0
    diag_pre_chunk_block_ms = 0.0
    chunk_iter = iter(chunks_iter)
    while True:
        # Time how long the next-chunk fetch blocks the publisher. With
        # the prefetch threadpool keeping n_workers chunks in flight, this
        # should usually be ~0. If chunk_wait jumps in steady state, tiled
        # is the bottleneck and the SUB-side ZMQ buffer (HWM=20000 on the
        # receiver) is what's keeping the pipeline fed during the gap.
        t_pre = time.perf_counter()
        try:
            chunk = next(chunk_iter)
        except StopIteration:
            break
        diag_pre_chunk_block_ms += (time.perf_counter() - t_pre) * 1000.0
        for frame in chunk:
            header = json.dumps({"frame": frame_id, "series": 1}).encode()
            payload = _encode_frame(frame, no_compress=no_compress)
            socket.send(header, zmq.SNDMORE)
            socket.send(encoding, zmq.SNDMORE)
            socket.send(payload)
            frame_id += 1
            diag_frames_in_window += 1
            now = time.perf_counter()
            if now - diag_window_start >= 1.0:
                print(
                    f"[eiger] 1s: sent={diag_frames_in_window} "
                    f"chunk_wait={diag_pre_chunk_block_ms:.0f} ms "
                    f"total={frame_id}/{n_frames}",
                    flush=True,
                )
                diag_window_start = now
                diag_frames_in_window = 0
                diag_pre_chunk_block_ms = 0.0

    print("[eiger] done", flush=True)
    if _own_socket:
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
        Logged target rate, no longer enforced. PandA positions are
        already small and infrequent (one message per ``points_per_message``
        scan points); the pipeline correlates them with Eiger frames by
        ``frame_number``, not wall-clock timing.
    points_per_message:
        Number of encoder samples bundled per ZMQ message (matches PandA
        default of 41).
    """
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    # PandA messages are tiny (~10 positions per message) so the HWM bump
    # mostly costs nothing; matches the Eiger publisher and the SUB-side HWM.
    socket.setsockopt(zmq.SNDHWM, 20000)
    socket.bind(endpoint)

    time.sleep(0.5)

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
    n_msgs_planned = (n_points + points_per_message - 1) // points_per_message
    last_progress = time.perf_counter()
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
        now = time.perf_counter()
        if frame_number % 50 == 0 or now - last_progress >= 1.0:
            print(
                f"[panda]  sent {frame_number}/{n_msgs_planned} msgs "
                f"({frame_number * points_per_message}/{n_points} pts)",
                flush=True,
            )
            last_progress = now

    socket.send_json({"msg_type": "stop", "emitted_frames": frame_number})
    print(f"[panda]  done — {frame_number}/{n_msgs_planned} msgs sent", flush=True)
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

    # Real HXN PandA emits 10 encoder samples per detector frame, and the
    # pipeline averages them back to one position per frame. Tiled storage
    # is inconsistent across scans — some hold the raw 10× samples, some
    # hold the already-averaged 1× values. We detect the ratio here and
    # forward it as `panda_upsample` in the hp config so the pipeline's
    # averaging stride matches what we publish (no redundant replicate+
    # average dance).
    positions_x = np.asarray(stream[x_key].read()).ravel()
    positions_y = np.asarray(stream[y_key].read()).ravel()
    upsample = max(1, len(positions_x) // n_total)
    pos_start = skip_frames * upsample
    pos_end = end_frame * upsample
    positions_x = positions_x[pos_start:pos_end]
    positions_y = positions_y[pos_start:pos_end]

    # Unit conversion. The pipeline's PointProcessorOp expects raw encoder
    # counts on the wire (real HXN PandA's native format) and applies
    # ``x_ratio`` (= -x_scale_factor µm/count, typically ~1e-5) to get
    # position in µm. When tiled has the raw inenc2_val/inenc3_val keys
    # (magnitude ~1e4–1e5), publish as-is. When only zpssx/zpssy is
    # available (motor user_readback in **micrometers**, magnitude single-
    # digit µm), the pipeline's ``* x_ratio`` mangles those values down
    # by ~10⁴, clustering all frames at canvas center — the "tiny dot"
    # mosaic. Convert back to count-space so the pipeline's math lands at
    # the right µm value: count = um / (x_ratio * x_direction). Detect
    # by magnitude: HXN encoder counts are easily > 1e3, motor readback
    # in µm is < 1e3 for any plausible nano-stage scan.
    if np.abs(positions_x).max() < 1e3 or np.abs(positions_y).max() < 1e3:
        cfg = load_config_from_tiled(run_uid, tiled_url=tiled_url)
        x_ratio = float(cfg["x_ratio"])
        y_ratio = float(cfg["y_ratio"])
        x_direction = float(cfg["x_direction"])
        y_direction = float(cfg["y_direction"])
        x_scale = x_ratio * x_direction
        y_scale = y_ratio * y_direction
        print(
            f"  ... tiled positions look like motor readback in µm "
            f"(max |x|={np.abs(positions_x).max():.3f}); converting to encoder "
            f"counts via um / (x_ratio * x_direction)",
            flush=True,
        )
        positions_x = positions_x / x_scale
        positions_y = positions_y / y_scale

    positions_x = positions_x.tolist()
    positions_y = positions_y.tolist()

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
        "panda_upsample": upsample,
    }


def iter_frame_chunks(
    streams: dict, head_frames: np.ndarray, chunk_size: int,
    n_workers: int = 4,
):
    """Yield chunks of detector frames; tiled fetches run in a worker pool.

    Tiled HTTPS read tops out around 1-2 MB/s per connection. A single
    fetcher gates the replay at ~15-22 s per 128 MB chunk
    (chunk_size=1024 × 256x256 uint16). Multiple parallel fetchers provide
    throughput in excess of one connection's share, up to the tiled server
    limit. The default of 4 workers stays well within the httpx connection
    pool (typically ~10 connections) to avoid ``httpx.PoolTimeout`` when
    all pool slots are held by long-running chunk fetches.

    Increasing n_workers beyond the pool limit causes requests to pile up
    waiting for a free connection; when they wait longer than the httpx
    pool_timeout (5 s by default) they raise ``PoolTimeout`` and abort the
    replay. Keep n_workers <= 8 unless you configure a larger httpx pool.

    Order is preserved with a sliding window: at most ``n_workers`` fetches
    are in flight, and ``yield`` waits on them in submission order. Memory
    peak is ``n_workers * chunk_size * frame_bytes`` (~128 MB at the default
    chunk_size=1024 with 4 workers).

    ``head_frames`` is the eager auto-detect read from
    :func:`setup_scan_from_tiled`; we yield it as-is to avoid re-reading
    the same range from tiled.
    """
    import collections
    import concurrent.futures

    head_end = streams["start"] + len(head_frames)
    end = streams["end"]
    frames_node = streams["frames_node"]
    frame_axis = streams["frame_axis"]

    yield head_frames

    ranges = [
        (s, min(s + chunk_size, end))
        for s in range(head_end, end, chunk_size)
    ]
    if not ranges:
        return

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=n_workers, thread_name_prefix="tiled-fetcher"
    ) as ex:
        pending: collections.deque = collections.deque()
        idx = 0
        # Prime the pool with up to n_workers pending fetches.
        while idx < len(ranges) and len(pending) < n_workers:
            s, e = ranges[idx]
            f = ex.submit(_fetch_frame_chunk, frames_node, frame_axis, s, e)
            f._range = (s, e)
            pending.append(f)
            idx += 1
        # Yield in submission order; submit a new fetch for each one we drain.
        while pending:
            fut = pending.popleft()
            # Retry on transient httpx.PoolTimeout (pool slot contention).
            # Back off briefly and re-submit the same range rather than
            # aborting the whole replay.
            retries_left = 5
            while True:
                try:
                    chunk = fut.result()
                    break
                except Exception as exc:
                    if retries_left > 0 and "PoolTimeout" in type(exc).__name__:
                        retries_left -= 1
                        import time
                        wait = 2 ** (5 - retries_left)  # 2, 4, 8, 16, 32 s
                        print(
                            f"[tiled] PoolTimeout fetching chunk — retrying in {wait}s "
                            f"({retries_left} retries left)",
                            flush=True,
                        )
                        time.sleep(wait)
                        # Reuse the same range by re-submitting the failed future.
                        # Find which range this was by peeking at chunk counters.
                        # Simpler: keep (start, end) alongside each future.
                        # We store them as fut._range below at submission.
                        s2, e2 = fut._range
                        fut = ex.submit(_fetch_frame_chunk, frames_node, frame_axis, s2, e2)
                        fut._range = (s2, e2)
                    else:
                        raise
            if idx < len(ranges):
                s, e = ranges[idx]
                f = ex.submit(_fetch_frame_chunk, frames_node, frame_axis, s, e)
                f._range = (s, e)
                pending.append(f)
                idx += 1
            yield chunk


def _json_request(
    url: str,
    method: str = "GET",
    payload: dict | None = None,
    timeout: float = 90,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        # 90s default — generous enough for /run, which now blocks on the
        # subprocess composing its operator graph (ZMQ SUB binds, TiledWriter
        # init, etc — typically ~6s but can be longer on cold model loads).
        # Server-side budget is 60s.
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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


def _wait_for_pipeline_ready(hp_url: str, timeout: float = 60.0) -> None:
    """Poll /status until pipeline_ready is True or the timeout expires.

    Used by --no-hp-start to ensure the ZMQ SUB is live before publishing.
    /run and /restart already block server-side until ready, so this is only
    needed in the --no-hp-start path where the caller started the pipeline
    out-of-band (e.g. 'hp start') and hands control back to the replay script.
    """
    deadline = time.monotonic() + timeout
    interval = 0.5
    while True:
        try:
            status = _json_request(f"{hp_url}/status", timeout=5)
        except RuntimeError as exc:
            print(f"[holoptycho] warning: could not reach API while waiting for readiness: {exc}", flush=True)
            status = {}
        pipeline_status = status.get("status", "unknown")
        if status.get("pipeline_ready"):
            print(f"[holoptycho] pipeline ready (status={pipeline_status!r})", flush=True)
            return
        if pipeline_status in ("stopped", "finished"):
            raise RuntimeError(
                f"holoptycho pipeline is in status {pipeline_status!r} — start it with "
                "'hp start' or 'hp restart' before running replay."
            )
        if pipeline_status == "error":
            raise RuntimeError(
                f"holoptycho pipeline reported an error: {status.get('error')!r}. "
                "Check 'hp logs' for details."
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"Timed out waiting for pipeline to become ready after {timeout:.0f} s "
                f"(last status={pipeline_status!r}). "
                "Is holoptycho running? Start it with 'hp start' first."
            )
        print(
            f"[holoptycho] waiting for pipeline ready (status={pipeline_status!r}, "
            f"{remaining:.0f}s remaining) …",
            flush=True,
        )
        time.sleep(interval)


def start_holoptycho_pipeline(args, panda_upsample: int = 1) -> None:
    """Start or restart the holoptycho pipeline with config from the same run."""
    hp_url = args.hp_url.rstrip("/")
    try:
        _json_request(f"{hp_url}/logs/clear", method="POST")
        print("[holoptycho] logs cleared", flush=True)
    except RuntimeError as exc:
        print(f"[holoptycho] WARNING: failed to clear logs: {exc}", flush=True)
    config_tiled_url = args.hp_config_tiled_url or args.tiled_url
    config = build_full_config(args.uid, tiled_url=config_tiled_url, args=args)
    config["panda_upsample"] = str(int(panda_upsample))
    # Tiled frames are already coordinate-corrected (the acquisition/file-writer
    # applied the local->global rotation when saving), so the pipeline must NOT
    # re-apply it on replay — force the no-op. Live runs use 'rot180' instead.
    config["detector_orientation"] = "identity"
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
    # /run and /restart now block server-side until the pipeline reports
    # ready (operators composed, ZMQ SUBs bound). On 200 the response is a
    # status snapshot rather than {"detail": ...}, so look for both shapes.
    label = result.get("detail") or (
        f"pipeline ready (status={result.get('status')!r})"
        if "status" in result
        else "pipeline request submitted"
    )
    print(f"[holoptycho] {label}", flush=True)


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
        help="Run UID to replay (or set TILED_RUN_UID env var; alternatively, pass --scan-id)",
    )
    parser.add_argument(
        "--scan-id",
        type=int,
        default=None,
        help="Bluesky scan_id (integer). Resolved to a UID via tiled search of hxn/migration; "
             "if multiple runs share the scan_id the newest by start.time is used.",
    )
    parser.add_argument(
        "--tiled-url",
        default=os.environ.get("TILED_BASE_URL", "https://tiled.nsls2.bnl.gov/hxn/migration"),
        help="Tiled catalog URL containing run entries (default: HXN migration catalog)",
    )
    parser.add_argument(
        "--tiled-api-key",
        default=os.environ.get("TILED_API_KEY", ""),
        help="Tiled API key (optional — uses cached credentials from 'tiled login' if omitted)",
    )
    parser.add_argument(
        "--hp-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start or restart holoptycho via its API using config from the same run before replaying. "
             "Default: on. Pass --no-hp-start when holoptycho is already running with a config "
             "that exactly matches the scan being replayed.",
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
        help="Informational target frame/message rate in Hz. Logged in the "
             "startup banner but not enforced — the publisher dumps each "
             "fetched chunk into ZMQ as fast as the SNDHWM and the receiver "
             "drain allow.",
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
        default=1024,
        help="Number of frames per tiled fetch during streaming (default: 1024). "
             "Smaller = lower latency before publishing starts and lower peak "
             "memory; larger = fewer round-trips. The publisher fetches the "
             "next chunk between rate-limited frame sends, so as long as a "
             "chunk's fetch (~hundreds of ms) is shorter than the publish "
             "interval (chunk_size / rate_hz) the stream stays smooth.",
    )
    parser.add_argument(
        "--tiled-workers",
        type=int,
        default=4,
        help="Number of parallel tiled HTTP fetches (default: 4). Each worker "
             "uses one connection from the shared httpx pool. Raising this above "
             "the pool limit (~10) causes httpx.PoolTimeout crashes; lower it "
             "if you see pool timeouts, raise it on fast networks if chunks are "
             "not arriving fast enough.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Bslz4-compress the published Eiger frames (matching the live "
             "wire format). Off by default because dectris-compression 0.3.1 "
             "dropped its C compress entrypoint and the pure-Python "
             "bitshuffle fallback is ~30s/batch — basically unusable. Enable "
             "only when explicitly verifying the decompression code path. "
             "Without this flag the script publishes raw bytes with a "
             "'raw' encoding header instead. The receiver inside "
             "holoptycho recognises 'raw' and reshapes the bytes directly.",
    )
    return parser.parse_args()


def _auto_batch_offsets(frames: np.ndarray, nx: int, ny: int) -> tuple[int, int]:
    """Auto-detect detector ROI offsets from the diffraction pattern center.

    Averages up to 50 frames, masks dtype-saturated pixels (which would
    otherwise drag the COM off course), then computes the intensity-weighted
    center of mass and returns (bx0, by0) such that an nx × ny crop is
    centered on it. Returns (0, 0) if the masked frame has zero total
    intensity.

    Verified on scan 404611: target was (135, 70), detected (137, 68) — 2px
    rounding noise after sat masking.

    Inlined verbatim from ptychoml.preprocess.auto_detect_roi_offsets so the
    replay env doesn't need the private ptychoml repo.
    """
    n_sample = 50
    saturation_threshold = np.iinfo(frames.dtype).max - 1
    sample = frames[:min(n_sample, len(frames))].astype(np.float64)
    mean_frame = sample.mean(axis=0)
    sat_mask = (sample > saturation_threshold).any(axis=0)
    masked = np.where(sat_mask, 0.0, mean_frame)
    total = masked.sum()
    if total <= 0:
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

    if args.scan_id is not None and args.uid:
        print("ERROR: pass either --uid or --scan-id, not both", file=sys.stderr)
        sys.exit(1)
    if args.scan_id is not None:
        args.uid = lookup_uid_by_scan_id(args.tiled_url, args.scan_id)
        print(f"[replay] scan_id={args.scan_id} → uid={args.uid}", flush=True)
    if not args.uid:
        print(
            "ERROR: must provide --uid, --scan-id, or set TILED_RUN_UID",
            file=sys.stderr,
        )
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

    # Pre-bind the Eiger PUB socket in both paths.  Binding before the
    # pipeline starts (or before waiting for it to be ready) ensures
    # EigerZmqRxOp.connect() always finds a live endpoint, so the ZMQ
    # SUBSCRIBE handshake completes well before the first frame is published.
    # In the --no-hp-start path the pipeline may have been running for
    # an arbitrary time; if the PUB was bound only after _wait_for_pipeline_ready
    # returned, ZMQ's reconnect backoff could exceed the 0.5 s sleep in
    # publish_eiger and the slow-joiner race would drop the first N frames
    # (causing FrameWriterOp to abort on the frame-0 check).
    eiger_ctx = zmq.Context()
    eiger_sock = eiger_ctx.socket(zmq.PUB)
    eiger_sock.setsockopt(zmq.SNDHWM, 20000)
    eiger_sock.bind(args.eiger_endpoint)
    print(f"[eiger] pre-bound PUB on {args.eiger_endpoint}", flush=True)

    if args.hp_start:
        start_holoptycho_pipeline(args, panda_upsample=streams["panda_upsample"])
    else:
        # --no-hp-start: pipeline was started out-of-band.  Wait for the ZMQ
        # SUB to be live; the pre-bound PUB above gives it the full wait
        # window to complete the handshake.
        _wait_for_pipeline_ready(args.hp_url)
        # pipeline_ready may have been True before we even bound the PUB
        # (e.g. user ran 'hp start' then ran replay separately). Give ZMQ
        # time to complete the reconnect cycle after the PUB appeared.
        # With ZMQ_RECONNECT_IVL_MAX=1000ms in datasource.py the SUB retries
        # at most every 1 s, so 2 s is sufficient margin.
        print("[eiger] waiting 2s for ZMQ SUB to reconnect to pre-bound PUB ...", flush=True)
        time.sleep(2.0)

    # Run Eiger and PandA publishers concurrently. The Eiger thread pulls
    # chunks from tiled lazily — first chunk is the already-fetched head,
    # subsequent chunks are server-side-sliced via _fetch_frame_chunk.
    chunks_iter = iter_frame_chunks(streams, streams["head_frames"], args.chunk_size, args.tiled_workers)
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
            not args.compress,
            eiger_sock,  # pre-bound socket, or None for --no-hp-start
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

    if eiger_ctx is not None:
        eiger_ctx.term()

    print("Replay complete.", flush=True)


if __name__ == "__main__":
    main()
