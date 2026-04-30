from __future__ import annotations

"""Runs the Holoscan application (PtychoApp) in a subprocess and keeps
AppState in sync with its lifecycle.

Why subprocess (not a thread or asyncio task in the API process)?

Holoscan's GXF runtime, CUDA contexts, TensorRT engines, and CuPy memory
pools don't fully release between back-to-back ``Application`` instances in
the same Python process — confirmed by the Holoscan docs and by empirical
testing (``cudaErrorIllegalAddress`` on the second ``run_async``). The SDK
expects one application per process. Spawning a fresh subprocess per
``/run`` gives us a guaranteed-clean CUDA context every time, and lets
``/stop`` use OS signals (SIGUSR1 for graceful, SIGTERM/SIGKILL as escalation)
which the OS guarantees to deliver — no SDK cooperation needed.

Stop semantics (three-stage):

1. SIGUSR1 — child sets ``_finish_event``, ``PtychoRecon.compute()`` trips
   the iteration-cap branch on its next tick, ``SaveResult`` writes
   ``final/`` to Tiled, ``fragment.stop_execution()`` releases the run loop,
   the subprocess exits 0 cleanly. Preserves ``write_final``.
2. SIGTERM — fallback if soft signal didn't drain within the grace window.
3. SIGKILL — last resort if the subprocess is wedged in C++ code.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time

from pathlib import Path

from .state import AppState, CONFIG_DIR
from . import db

logger = logging.getLogger("holoptycho.runner")

# Module-level handle to the running pipeline subprocess.
_proc: subprocess.Popen | None = None
_stop_requested: bool = False
_state_lock = threading.Lock()

# Sentinel file the subprocess touches after fragment.stop_execution() returns.
# We classify rc<0 + sentinel-present as "finished" instead of "error" because
# Holoscan's TensorRT/CUDA destructors can SIGABRT during operator teardown
# even after the work fully completed.
_WORK_COMPLETE_SENTINEL = Path("/tmp/holoptycho_work_complete")

# Soft stop window. Long enough for PtychoRecon to trip the iteration-cap
# branch and for SaveResult to write_final to Tiled.
_SOFT_STOP_TIMEOUT = 10.0
# SIGTERM window. The Python signal handler runs at the next bytecode
# boundary, but if compute() is in C code (CUDA, TensorRT) we wait for it.
_SIGTERM_TIMEOUT = 10.0
# SIGKILL is unconditional after this. The OS reclaims the CUDA context.
_SIGKILL_TIMEOUT = 5.0
# Race window in start() for any lingering subprocess from a prior /stop.
_STARTUP_RACE_TIMEOUT = 30.0

_REQUIRED_ENV_VARS = ("SERVER_STREAM_SOURCE", "PANDA_STREAM_SOURCE")

_REQUIRED_CONFIG_FIELDS = (
    "scan_num",
    "nx", "ny",
    "x_range", "y_range",
    "x_num", "y_num",
    "det_roix0", "det_roiy0",
    "x_ratio", "y_ratio",
    "xray_energy_kev",
    "ccd_pixel_um",
    "distance",
)


def _monitor(state: AppState, proc: subprocess.Popen):
    """Wait on subprocess exit and translate returncode into state.status."""
    rc = proc.wait()
    with _state_lock:
        stopped = _stop_requested
    work_complete = _WORK_COMPLETE_SENTINEL.exists()
    if rc == 0:
        if stopped:
            state.update(status="stopped")
            logger.info("Pipeline subprocess stopped on request (rc=0)")
        else:
            state.update(status="finished")
            logger.info("Pipeline subprocess finished normally (rc=0)")
    elif stopped and rc < 0:
        # Negative rc on POSIX = killed by signal; expected on hard stop.
        state.update(status="stopped")
        logger.info("Pipeline subprocess killed on stop request (signal=%d)", -rc)
    elif work_complete:
        # Holoscan teardown abort after work was fully completed (write_final
        # landed, stop_execution returned). Treat as a normal finish.
        state.update(status="finished")
        logger.warning(
            "Pipeline subprocess crashed during teardown (rc=%d) but work "
            "completed; classifying as finished", rc,
        )
    else:
        state.update(status="error", error=f"pipeline subprocess exited rc={rc}")
        logger.error("Pipeline subprocess exited with error (rc=%d)", rc)
    with _state_lock:
        _clear_pipeline_state()


def _clear_pipeline_state():
    """Reset module-level handles. Caller must hold _state_lock."""
    global _proc, _stop_requested
    _proc = None
    _stop_requested = False


def start(state: AppState, config: dict | None = None) -> None:
    """Start the Holoscan application as a subprocess.

    Returns once the subprocess has been spawned and the config has been
    delivered on its stdin. Raises ``RuntimeError`` if a previous subprocess
    is still alive, if no config is available, or if required ZMQ env vars
    are missing.
    """
    global _proc, _stop_requested

    with state._lock:
        if state.status in ("starting", "running"):
            raise RuntimeError(
                f"App is already {state.status}. Stop it first."
            )

    # Race-guard: if a previous subprocess hasn't exited yet (e.g. /stop was
    # called but cleanup is in flight), wait briefly for it before rejecting.
    with _state_lock:
        prior = _proc
    if prior is not None and prior.poll() is None:
        try:
            prior.wait(timeout=_STARTUP_RACE_TIMEOUT)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Previous pipeline is still shutting down after "
                f"{_STARTUP_RACE_TIMEOUT:.0f} s. Try again in a moment."
            )

    # Validate required ZMQ env vars before doing anything else.
    missing = [v for v in _REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        raise RuntimeError(
            f"Required environment variable(s) not set: {', '.join(missing)}. "
            "Set SERVER_STREAM_SOURCE and PANDA_STREAM_SOURCE to the ZMQ "
            "endpoints of the Eiger detector and PandA box respectively."
        )

    # Resolve config: use provided config, fall back to last persisted.
    if config is not None:
        db.set_last_config(config)
        state.update(last_config=config)
    else:
        config = state.last_config or db.get_last_config()
        if config is None:
            raise RuntimeError(
                "No config provided and no previous config found. "
                "Pass a config JSON to 'hp start'."
            )
        state.update(last_config=config)

    config_path = db.write_config_ini(config, CONFIG_DIR)
    logger.info("Config written to %s", config_path)

    # Validate required config fields.
    missing_fields = [f for f in _REQUIRED_CONFIG_FIELDS if f not in config]
    if missing_fields:
        raise RuntimeError(
            f"Config is missing required field(s): {', '.join(missing_fields)}."
        )

    # Build the subprocess env. Inherit everything (so SERVER_STREAM_SOURCE,
    # PANDA_STREAM_SOURCE, AZURE_*, TILED_*, etc. all flow through), and add
    # the path-knob env vars the entry-point script reads.
    child_env = dict(os.environ)
    child_env["HOLOPTYCHO_LOG_FILE"] = state.log_file
    child_env["HOLOPTYCHO_CONFIG_PATH"] = str(config_path)
    child_env["HOLOPTYCHO_COMPLETE_SENTINEL"] = str(_WORK_COMPLETE_SENTINEL)
    if state.current_engine_path:
        child_env["HOLOPTYCHO_ENGINE_PATH"] = state.current_engine_path

    # Clear stale sentinel from any prior run.
    _WORK_COMPLETE_SENTINEL.unlink(missing_ok=True)

    # Reset stop bookkeeping before spawning.
    with _state_lock:
        _stop_requested = False

    state.update(start_time=time.time(), error=None)

    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "holoptycho.server._pipeline_subprocess"],
            stdin=subprocess.PIPE,
            env=child_env,
            # No need to capture stdout/stderr — the child writes to the
            # shared log file via its own RotatingFileHandler.
        )
    except Exception as exc:
        message = f"Failed to spawn pipeline subprocess: {exc}"
        state.update(status="error", error=message, start_time=None)
        logger.exception("Popen failed")
        raise RuntimeError(message) from exc

    try:
        proc.stdin.write(json.dumps(config).encode())
        proc.stdin.close()
    except Exception as exc:
        # If config delivery failed, kill the orphan and bail.
        proc.kill()
        proc.wait(timeout=5)
        message = f"Failed to deliver config to pipeline subprocess: {exc}"
        state.update(status="error", error=message, start_time=None)
        raise RuntimeError(message) from exc

    with _state_lock:
        _proc = proc

    state.update(status="running")
    logger.info("Pipeline subprocess started (pid=%d)", proc.pid)

    monitor = threading.Thread(
        target=_monitor,
        args=(state, proc),
        daemon=True,
        name="holoscan-monitor",
    )
    monitor.start()


def stop(state: AppState) -> None:
    """Stop the running pipeline subprocess: SIGUSR1 → SIGTERM → SIGKILL.

    Blocks until the subprocess has exited and ``state.status`` has reached
    a terminal value. Raises ``RuntimeError`` if the pipeline is not running.
    """
    global _stop_requested

    with state._lock:
        if state.status not in ("starting", "running"):
            raise RuntimeError(f"App is not running (status={state.status!r})")

    with _state_lock:
        proc = _proc
        _stop_requested = True

    if proc is None:
        # Status was "running" but we have no proc handle; defensively flip.
        state.update(status="stopped")
        return

    # Stage 1: SIGUSR1 — graceful, preserves write_final.
    logger.info("Stop requested — sending SIGUSR1 (graceful) to pid=%d", proc.pid)
    try:
        proc.send_signal(signal.SIGUSR1)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=_SOFT_STOP_TIMEOUT)
        _await_status_terminal(state)
        return
    except subprocess.TimeoutExpired:
        pass

    # Stage 2: SIGTERM.
    logger.warning(
        "SIGUSR1 grace expired (%.1f s) — sending SIGTERM to pid=%d",
        _SOFT_STOP_TIMEOUT, proc.pid,
    )
    try:
        proc.terminate()
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=_SIGTERM_TIMEOUT)
        _await_status_terminal(state)
        return
    except subprocess.TimeoutExpired:
        pass

    # Stage 3: SIGKILL.
    logger.error(
        "SIGTERM grace expired (%.1f s) — sending SIGKILL to pid=%d",
        _SIGTERM_TIMEOUT, proc.pid,
    )
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=_SIGKILL_TIMEOUT)
    except subprocess.TimeoutExpired:
        # OS-level SIGKILL not delivered — extremely unusual. Best we can
        # do is raise; the next /run will hit the race-guard and fail until
        # the kernel reaps the zombie.
        raise RuntimeError(
            f"Pipeline subprocess (pid={proc.pid}) did not exit after SIGKILL "
            f"within {_SIGKILL_TIMEOUT:.0f} s. Kernel-level reap pending."
        )

    _await_status_terminal(state)


def _await_status_terminal(state: AppState, timeout: float = 2.0) -> None:
    """Wait briefly for the monitor thread to publish a terminal status.

    The monitor thread runs after proc.wait() returns, so there's a small
    window where stop() returns before status has flipped. Spin briefly so
    callers see a consistent state on return.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        with state._lock:
            if state.status in ("stopped", "finished", "error"):
                return
        time.sleep(0.05)
