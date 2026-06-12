#!/usr/bin/env python3
"""
Verify connectivity to the Eiger and PandA ZMQ streams.

Reads the same environment variables as the holoptycho pipeline:
  SERVER_STREAM_SOURCE   - Eiger ZMQ endpoint  (default tcp://localhost:5555)
  PANDA_STREAM_SOURCE    - PandA ZMQ endpoint  (default tcp://localhost:5556)
  SERVER_PUBLIC_KEY      - Eiger CurveZMQ server public key (optional)
  CLIENT_PUBLIC_KEY      - CurveZMQ client public key       (optional)
  CLIENT_SECRET_KEY      - CurveZMQ client secret key       (optional)

All three CurveZMQ keys must be set together or not at all.

Usage:
    pixi run python scripts/check_zmq.py
    pixi run python scripts/check_zmq.py --timeout 10

    # Also dump received diffraction patterns + positions to files for
    # manual inspection (decoded the same way the pipeline decodes them):
    pixi run python scripts/check_zmq.py --save
    pixi run python scripts/check_zmq.py --save /tmp/zmqdump --num-frames 20
"""

import argparse
import base64
import json
import os
import socket
import sys

import numpy as np
import zmq


# ---------------------------------------------------------------------------
# Eiger frame decode — mirrors holoptycho/datasource.py::decode_json_message
# ---------------------------------------------------------------------------

_SUPPORTED_ENCODINGS = {
    "bs32-lz4<": "bslz4",
    "lz4<": "lz4",
    "bs16-lz4<": "bslz4",
    "raw": "raw",
}
_SUPPORTED_TYPES = {"uint32": "uint32", "uint16": "uint16"}


def _decode_frame(data_msg: bytes, encoding_msg: dict) -> np.ndarray:
    """Decode one Eiger ``dimage_d-1.0`` payload into a 2-D ndarray.

    Replicates ``decode_json_message`` from ``holoptycho/datasource.py`` —
    including the ``reshape(shape[1], shape[0])`` that inverts the Eiger
    ``[cols, rows]`` header convention — so the orientation written to disk is
    exactly what the pipeline sees.
    """
    enc = encoding_msg.get("encoding")
    shape = encoding_msg.get("shape")
    typ = encoding_msg.get("type")

    enc_str = _SUPPORTED_ENCODINGS.get(enc)
    if not enc_str:
        raise RuntimeError(f"Encoding {enc!r} is not supported")
    type_str = _SUPPORTED_TYPES.get(typ)
    if not type_str:
        raise RuntimeError(f"Type {typ!r} is not supported")

    elem_type = getattr(np, type_str)
    elem_size = elem_type(0).nbytes
    if enc_str == "raw":
        image = np.frombuffer(bytearray(data_msg), dtype=elem_type)
    else:
        from dectris.compression import decompress

        decompressed = decompress(data_msg, enc_str, elem_size=elem_size)
        image = np.frombuffer(bytearray(decompressed), dtype=elem_type)
    # Eiger reports shape as [cols, rows]; reshape back to (rows, cols).
    return image.reshape(shape[1], shape[0])


def _tcp_reachable(host: str, port: int, timeout: float = 3.0) -> bool:
    """Return True if a TCP connection to host:port succeeds within timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _parse_endpoint(endpoint: str) -> tuple[str, int]:
    """Parse 'tcp://host:port' → (host, port)."""
    # endpoint looks like tcp://hostname:5559
    _, _, hostport = endpoint.partition("://")
    host, _, port_str = hostport.rpartition(":")
    return host, int(port_str)


def _parse_zmq_key(value: str, name: str) -> bytes:
    """Parse a CurveZMQ key string into bytes suitable for zmq.setsockopt.

    Accepts:
      - Z85-encoded 40-character ASCII strings (the native ZMQ format)
      - Base64-encoded strings that decode to 32 raw key bytes
      - Raw 32-byte binary strings (stored as latin-1)
      - Raw 40-byte binary strings (stored as latin-1, passed directly to ZMQ)

    Raises ValueError with a clear message if the value is none of the above.
    """
    # Z85: 40 printable ASCII chars
    try:
        encoded = value.encode("ascii")
        if len(encoded) == 40:
            return encoded
    except UnicodeEncodeError:
        pass  # Not ASCII — try other formats

    # Base64: decode to 32 raw bytes, then re-encode as Z85
    try:
        ascii_val = value.encode("ascii")
        raw = base64.b64decode(ascii_val)
        if len(raw) == 32:
            return zmq.z85.encode(raw)
    except (UnicodeEncodeError, Exception):
        pass

    # Raw binary stored as a string (latin-1 preserves byte values 0-255)
    try:
        raw = value.encode("latin-1")
        hex_preview = raw.hex()
        if len(raw) == 32:
            print(
                f"  {name}: treating as raw 32-byte binary key (hex: {hex_preview[:16]}...)"
            )
            return raw
        if len(raw) == 40:
            print(
                f"  {name}: treating as raw 40-byte binary key (hex: {hex_preview[:16]}...)"
            )
            return raw
        raise ValueError(f"{name}: raw bytes length {len(raw)}, expected 32 or 40")
    except Exception as exc:
        pass

    raise ValueError(
        f"{name}: cannot parse key. Got {len(value)}-char string with non-ASCII chars. "
        f"Expected Z85 (40 ASCII chars) or base64 (→ 32 bytes). "
        f"Hex dump: {value.encode('latin-1', errors='replace').hex()}"
    )


def _apply_curve(sock: zmq.Socket) -> bool:
    """Apply CurveZMQ keys from env if SERVER_PUBLIC_KEY is set. Returns True if applied.

    CLIENT_PUBLIC_KEY / CLIENT_SECRET_KEY are optional — if absent a throwaway
    keypair is generated automatically. The Eiger server uses the client public
    key only for encryption, not for allowlisting, so an ephemeral pair works.
    """
    server_key = os.environ.get("SERVER_PUBLIC_KEY", "")
    if not server_key:
        return False
    client_pub = os.environ.get("CLIENT_PUBLIC_KEY", "")
    client_sec = os.environ.get("CLIENT_SECRET_KEY", "")
    if not client_pub or not client_sec:
        client_pub, client_sec = zmq.curve_keypair()
        print("  CurveZMQ: using ephemeral client keypair")
    else:
        client_pub = _parse_zmq_key(client_pub, "CLIENT_PUBLIC_KEY")
        client_sec = _parse_zmq_key(client_sec, "CLIENT_SECRET_KEY")
        print("  CurveZMQ: using provided client keypair")
    sock.setsockopt(
        zmq.CURVE_SERVERKEY, _parse_zmq_key(server_key, "SERVER_PUBLIC_KEY")
    )
    sock.setsockopt(zmq.CURVE_PUBLICKEY, client_pub)
    sock.setsockopt(zmq.CURVE_SECRETKEY, client_sec)
    return True


def check_eiger(
    ctx: zmq.Context,
    endpoint: str,
    timeout_ms: int,
    save_frames: int = 0,
    save_dir: str | None = None,
) -> str:
    """Returns 'ok', 'timeout', or 'error'.

    When ``save_frames`` > 0, collect and decode that many image frames and
    write them to ``<save_dir>/eiger_dps.npy`` as a ``(N, H, W)`` array.
    """
    print(f"Eiger  {endpoint}")
    host, port = _parse_endpoint(endpoint)
    if not _tcp_reachable(host, port):
        print(
            f"  ERROR — TCP connection to {host}:{port} failed (host unreachable or port closed)"
        )
        return "error"
    print(f"  TCP {host}:{port} reachable")
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.RCVHWM, 0 if save_frames else 10)
    curve = _apply_curve(sock)
    if curve:
        print("  CurveZMQ: enabled")
    else:
        print("  CurveZMQ: disabled (no keys set)")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.connect(endpoint)

    collected: list[np.ndarray] = []
    try:
        # The Eiger streams each image as three consecutive ZMQ messages: a
        # JSON header (contains "frame"), a JSON encoding descriptor, and the
        # binary payload. The live detector sends them as three separate
        # messages; the replay sends them as one multipart message. Sequential
        # recv() handles both (recv() returns one frame at a time either way).
        # Non-image traffic (global/series headers) is skipped by requiring a
        # "frame" key.
        max_attempts = (save_frames * 50 + 200) if save_frames else 40
        for attempt in range(max_attempts):
            try:
                raw = sock.recv()
            except zmq.Again:
                if collected:
                    break  # got some frames already; stop waiting
                print(f"  TIMEOUT after {timeout_ms / 1000:.1f}s — no data received")
                print("  Is the detector armed / a scan running?")
                return "timeout"

            try:
                # json.loads on bytes sniffs the encoding and raises
                # UnicodeDecodeError (not JSONDecodeError) on a binary payload,
                # so catch broadly and skip anything that isn't a JSON object.
                header = json.loads(raw)
            except Exception:
                continue
            if not isinstance(header, dict) or "frame" not in header:
                continue

            frame_id = header["frame"]
            if not save_frames:
                print(f"  OK — received frame {frame_id}")
                return "ok"

            # Saving path: the next two messages are the encoding descriptor
            # and the binary payload.
            try:
                encoding = json.loads(sock.recv())
                data = sock.recv()
                image = _decode_frame(data, encoding)
            except zmq.Again:
                if collected:
                    break
                print(f"  TIMEOUT after {timeout_ms / 1000:.1f}s — no data received")
                return "timeout"
            except Exception as exc:  # noqa: BLE001 — report and keep going
                print(f"  WARN — failed to decode frame {frame_id}: {exc}")
                continue
            collected.append(image)
            print(
                f"  collected frame {frame_id}: shape={image.shape} "
                f"dtype={image.dtype}"
            )
            if len(collected) >= save_frames:
                break

        if save_frames:
            if not collected:
                print(f"  TIMEOUT — no frames decoded within {timeout_ms / 1000:.1f}s")
                return "timeout"
            arr = np.stack(collected)
            path = os.path.join(save_dir or ".", "eiger_dps.npy")
            np.save(path, arr)
            print(
                f"  OK — saved {len(collected)} dps to {os.path.abspath(path)} "
                f"(shape={arr.shape}, dtype={arr.dtype})"
            )
            return "ok"

        print("  TIMEOUT — received messages but none contained a 'frame' key")
        print("  Is the detector armed / a scan running?")
        return "timeout"
    except Exception as exc:
        print(f"  ERROR — {exc}")
        return "error"
    finally:
        sock.close()


def check_panda(
    ctx: zmq.Context,
    endpoint: str,
    timeout_ms: int,
    save_msgs: int = 0,
    save_dir: str | None = None,
) -> str:
    """Returns 'ok', 'timeout', or 'error'.

    When ``save_msgs`` > 0, collect that many PandA ``data`` messages, gather
    their per-channel position samples, and write them to
    ``<save_dir>/panda_positions.npy`` as an ``(N, n_channels)`` array (columns
    ordered by sorted channel name, printed to stdout).
    """
    print(f"PandA  {endpoint}")
    host, port = _parse_endpoint(endpoint)
    if not _tcp_reachable(host, port):
        print(
            f"  ERROR — TCP connection to {host}:{port} failed (host unreachable or port closed)"
        )
        return "error"
    print(f"  TCP {host}:{port} reachable")
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.RCVHWM, 0 if save_msgs else 10)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.connect(endpoint)

    channel_data: dict[str, list] = {}
    n_data_msgs = 0
    try:
        max_attempts = (save_msgs * 5 + 200) if save_msgs else 20
        for attempt in range(max_attempts):
            try:
                raw = sock.recv()
            except zmq.Again:
                if n_data_msgs:
                    break  # got some data messages already; stop waiting
                print(f"  TIMEOUT after {timeout_ms / 1000:.1f}s — no data received")
                print("  Is PandA streaming / a scan running?")
                return "timeout"

            try:
                msg = json.loads(raw)
            except Exception:  # binary or non-JSON message — skip
                continue
            if not isinstance(msg, dict):
                continue

            msg_type = msg.get("msg_type", "<unknown>")
            if not save_msgs:
                print(f"  OK — received msg_type='{msg_type}'")
                return "ok"

            # Saving path: accumulate position samples from 'data' messages.
            if msg_type != "data":
                print(f"  (skipping msg_type='{msg_type}')")
                continue
            for ch, dset in msg.get("datasets", {}).items():
                channel_data.setdefault(ch, []).extend(dset.get("data", []))
            n_data_msgs += 1
            print(
                f"  collected data msg {n_data_msgs} "
                f"(frame_number={msg.get('frame_number')})"
            )
            if n_data_msgs >= save_msgs:
                break

        if save_msgs:
            if not channel_data:
                print(f"  TIMEOUT — no 'data' messages within {timeout_ms / 1000:.1f}s")
                return "timeout"
            names = sorted(channel_data)
            cols = [np.asarray(channel_data[n], dtype=float) for n in names]
            m = min(len(c) for c in cols)
            arr = np.stack([c[:m] for c in cols], axis=1)
            path = os.path.join(save_dir or ".", "panda_positions.npy")
            np.save(path, arr)
            print(
                f"  OK — saved {arr.shape[0]} positions to {os.path.abspath(path)} "
                f"(shape={arr.shape})"
            )
            print(f"       columns: {names}")
            return "ok"

        print("  TIMEOUT — received messages but none were valid JSON")
        return "timeout"
    except Exception as exc:
        print(f"  ERROR — {exc}")
        return "error"
    finally:
        sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for a message on each stream (default: 5)",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const=".",
        default=None,
        metavar="DIR",
        help="Dump received diffraction patterns to <DIR>/eiger_dps.npy and "
        "positions to <DIR>/panda_positions.npy for manual inspection. "
        "DIR defaults to the current directory if given with no value. "
        "Implies a longer wait, so consider raising --timeout.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="Number of Eiger frames to save when --save is set (default: 10)",
    )
    parser.add_argument(
        "--num-position-msgs",
        type=int,
        default=10,
        help="Number of PandA 'data' messages to gather when --save is set "
        "(default: 10)",
    )
    args = parser.parse_args()

    eiger_ep = os.environ.get(
        "SERVER_STREAM_SOURCE", "tcp://xf03idc-eiger2-ioc.nsls2.bnl.local:5559"
    )
    panda_ep = os.environ.get(
        "PANDA_STREAM_SOURCE", "tcp://xf03idc-eiger2-ioc.nsls2.bnl.local:6666"
    )
    timeout_ms = int(args.timeout * 1000)

    save_dir = args.save
    save_frames = args.num_frames if save_dir is not None else 0
    save_msgs = args.num_position_msgs if save_dir is not None else 0
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving enabled — writing files to {os.path.abspath(save_dir)}")

    ctx = zmq.Context()
    results = {}
    print()
    results["eiger"] = check_eiger(ctx, eiger_ep, timeout_ms, save_frames, save_dir)
    print()
    results["panda"] = check_panda(ctx, panda_ep, timeout_ms, save_msgs, save_dir)
    print()
    ctx.term()

    all_ok = all(r == "ok" for r in results.values())
    for name, result in results.items():
        status = {"ok": "OK", "timeout": "TIMEOUT", "error": "ERROR"}.get(
            result, result
        )
        print(f"  {status:7s}  {name}")
    print()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
