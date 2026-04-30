from __future__ import annotations

"""Runs the Holoscan application (PtychoApp) via Application.run_async() and
keeps AppState in sync with its lifecycle.

Stop semantics (two-stage):

1. Soft — ``_finish_event`` is set; ``PtychoRecon.compute()`` trips the
   iteration-cap branch on its next tick, ``SaveResult`` writes ``final/``
   to Tiled, the graph deadlocks, ``run_async()`` resolves naturally.
   Preserves ``write_final``.
2. Hard — after a soft timeout, ``Application.stop_execution()`` flips every
   operator's async condition to ``EVENT_NEVER``. Operators currently inside
   ``compute()`` finish their current call (no preemption — Holoscan caveat),
   then the scheduler exits.
"""

import logging
import os
import threading
import time
from concurrent.futures import Future, TimeoutError as FutureTimeoutError

from .state import AppState, CONFIG_DIR
from . import db

logger = logging.getLogger("holoptycho.runner")

# Module-level handles to the running pipeline.
_app: object | None = None              # holoscan.core.Application instance
_app_future: Future | None = None       # returned by app.run_async()
_stop_requested: bool = False           # set by stop() so monitor labels result "stopped"
_state_lock = threading.Lock()          # guards _app, _app_future, _stop_requested

# Soft stop window. Long enough for PtychoRecon to trip the iteration-cap
# branch and for SaveResult to write_final to Tiled (typical: <2 s including
# network).
_SOFT_STOP_TIMEOUT = 5.0
# Hard stop window. stop_execution() returns immediately, but run_async()
# only returns once any in-flight compute() finishes naturally. Sized for the
# longest expected iter_once() (~12 s for a full DM batch).
_HARD_STOP_TIMEOUT = 15.0
# Race window after a stop request, used by start() to wait for the prior
# pipeline to finish before rejecting a new /run.
_STARTUP_RACE_TIMEOUT = 30.0

_REQUIRED_ENV_VARS = ("SERVER_STREAM_SOURCE", "PANDA_STREAM_SOURCE")

# Config fields that must be present — no sensible default exists for these.
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


def _monitor(state: AppState, app, future: Future):
    """Translate Future resolution into state.status and clean up the executor.

    Runs as a daemon thread; exits when the Future resolves.
    """
    try:
        future.result()
    except Exception as exc:
        state.update(status="error", error=str(exc))
        logger.exception("Holoscan app raised an exception")
    else:
        with _state_lock:
            stopped = _stop_requested
        if stopped:
            state.update(status="stopped")
            logger.info("Holoscan app stopped on request")
        else:
            state.update(status="finished")
            logger.info("Holoscan app finished normally")
    finally:
        try:
            app.shutdown_async_executor(wait=True)
        except Exception:
            logger.exception("Failed to shut down async executor")
        with _state_lock:
            _clear_pipeline_state()


def _clear_pipeline_state():
    """Reset module-level pipeline handles. Caller must hold _state_lock."""
    global _app, _app_future, _stop_requested
    _app = None
    _app_future = None
    _stop_requested = False


def start(state: AppState, config: dict | None = None) -> None:
    """Start the Holoscan application via app.run_async().

    Parameters
    ----------
    state:
        Shared application state.
    config:
        Config dict for this run. If ``None``, the last persisted config is
        used; raises ``RuntimeError`` if neither is available.

    Raises ``RuntimeError`` if a previous pipeline is still alive after the
    startup race window, if no config is available, or if required ZMQ
    environment variables are not set.
    """
    global _app, _app_future, _stop_requested

    with state._lock:
        if state.status in ("starting", "running"):
            raise RuntimeError(
                f"App is already {state.status}. Stop it first."
            )

    # Race-guard: if a previous Future hasn't resolved yet (e.g. stop() was
    # called but the runtime hasn't fully exited), wait briefly before
    # rejecting. This makes back-to-back stop+start work without an external
    # retry loop.
    with _state_lock:
        prior_future = _app_future
    if prior_future is not None and not prior_future.done():
        try:
            prior_future.result(timeout=_STARTUP_RACE_TIMEOUT)
        except FutureTimeoutError:
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

    # Resolve config: use provided config, fall back to last persisted config.
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

    # Validate required config fields before importing heavy deps.
    missing_fields = [f for f in _REQUIRED_CONFIG_FIELDS if f not in config]
    if missing_fields:
        raise RuntimeError(
            f"Config is missing required field(s): {', '.join(missing_fields)}."
        )

    # Import heavy GPU deps here so the FastAPI server can start without them.
    try:
        from holoptycho.ptycho_holo import PtychoApp
    except ImportError as exc:
        message = f"Failed to import Holoscan app: {exc}"
        state.update(status="error", error=message, start_time=None)
        logger.exception("Holoscan app import failed")
        raise RuntimeError(message) from exc

    resolved_engine = state.current_engine_path
    try:
        app = PtychoApp(
            config_path=config_path,
            config_overrides=config,
            engine_path=resolved_engine,
        )
    except Exception as exc:
        message = f"Failed to initialize Holoscan app: {exc}"
        state.update(status="error", error=message, start_time=None)
        logger.exception(
            "Holoscan app initialization failed (config_path=%s, engine_path=%s)",
            config_path,
            resolved_engine,
        )
        raise RuntimeError(message) from exc

    # Reset the soft-finish flag so the new run starts fresh.
    from holoptycho import ptycho_holo
    ptycho_holo._finish_event.clear()

    # run_async() submits app.run() to an internal ThreadPoolExecutor and
    # returns the Future. The Future is alive immediately, so we transition
    # straight to "running" without a separate "starting" phase. Order matters:
    # populate the module-level handles BEFORE flipping status to "running" so
    # a concurrent /stop never sees "running" with no Future to wait on.
    state.update(start_time=time.time(), error=None)
    future = app.run_async()

    with _state_lock:
        _app = app
        _app_future = future
        _stop_requested = False

    state.update(status="running")

    monitor = threading.Thread(
        target=_monitor,
        args=(state, app, future),
        daemon=True,
        name="holoscan-monitor",
    )
    monitor.start()
    logger.info("Holoscan app started (run_async)")


def stop(state: AppState) -> None:
    """Stop the running Holoscan app: soft (_finish_event) → hard (stop_execution).

    Blocks until the pipeline has actually exited, the run-time has been
    released, and ``state.status`` has reached ``"stopped"`` (or ``"error"``,
    if the run raised). Raises ``RuntimeError`` if the pipeline is not
    running, or if the hard-stop window expires (subprocess fallback would be
    needed in that case).
    """
    global _stop_requested

    with state._lock:
        if state.status not in ("starting", "running"):
            raise RuntimeError(f"App is not running (status={state.status!r})")

    with _state_lock:
        app = _app
        future = _app_future
        _stop_requested = True

    if app is None or future is None:
        # Shouldn't happen given the status check, but defend anyway.
        state.update(status="stopped")
        return

    # Soft: trip the natural-termination path so write_final lands cleanly.
    from holoptycho import ptycho_holo
    ptycho_holo._finish_event.set()
    logger.info("Stop requested — flushing pipeline and saving final results")

    try:
        future.result(timeout=_SOFT_STOP_TIMEOUT)
    except FutureTimeoutError:
        pass
    else:
        # Soft path drained the graph. The monitor thread will set status to
        # "stopped" (because _stop_requested is True) and clean up the
        # executor. Wait briefly for that bookkeeping to complete so callers
        # see a consistent state on return.
        _await_status_terminal(state)
        return

    # Hard: force every operator's scheduling condition to NEVER.
    logger.warning(
        "Soft stop did not drain within %.1f s — calling stop_execution()",
        _SOFT_STOP_TIMEOUT,
    )
    try:
        app.stop_execution()
    except Exception:
        logger.exception("stop_execution() raised")

    try:
        future.result(timeout=_HARD_STOP_TIMEOUT)
    except FutureTimeoutError:
        logger.error(
            "Pipeline did not exit within %.1f s after stop_execution(). "
            "An operator's compute() is wedged; subprocess fallback would be "
            "needed for full reliability.",
            _HARD_STOP_TIMEOUT,
        )
        raise RuntimeError(
            f"stop_execution() did not unblock run_async() within "
            f"{_HARD_STOP_TIMEOUT:.0f} s. API restart required."
        )

    _await_status_terminal(state)


def _await_status_terminal(state: AppState, timeout: float = 2.0) -> None:
    """Wait briefly for the monitor thread to publish a terminal status.

    The monitor thread runs after future.result() resolves, so there's a small
    window where stop() returns before status has flipped. Callers expect a
    consistent state machine, so spin briefly here.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        with state._lock:
            if state.status in ("stopped", "finished", "error"):
                return
        time.sleep(0.05)
