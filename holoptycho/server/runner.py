from __future__ import annotations

"""Runs the Holoscan application (PtychoApp) in a background thread and
keeps AppState in sync with its lifecycle."""

import logging
import os
import threading
import time

from .state import AppState, CONFIG_DIR
from . import db

logger = logging.getLogger("holoptycho.runner")

# Module-level reference to the runner thread so we can check is_alive().
_runner_thread: threading.Thread | None = None

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


def _run_app(app, state: AppState):
    """Target function for the runner thread."""
    try:
        state.update(status="running", start_time=time.time(), error=None)
        logger.info("Holoscan app starting")
        app.run()
        state.update(status="finished")
        logger.info("Holoscan app finished normally")
    except Exception as exc:
        state.update(status="error", error=str(exc))
        logger.exception("Holoscan app raised an exception")


def start(state: AppState, config: dict | None = None) -> None:
    """Start the Holoscan application in a daemon background thread.

    Parameters
    ----------
    state:
        Shared application state.
    config:
        Config dict to use for this run.  If None, the last config stored in
        the DB is used.  Raises RuntimeError if neither is available.

    Raises RuntimeError if an app is already running, if a previous runner
    thread is still alive, if no config is available, or if required ZMQ
    environment variables are not set.
    """
    global _runner_thread

    with state._lock:
        if state.status in ("starting", "running"):
            raise RuntimeError(
                f"App is already {state.status}. Stop it first."
            )

    # Guard against the race where stop() has been called but the thread
    # hasn't exited yet. Wait briefly for the prior thread before rejecting,
    # so back-to-back stop+start works without an external retry loop.
    if _runner_thread is not None and _runner_thread.is_alive():
        _runner_thread.join(timeout=30)
        if _runner_thread.is_alive():
            raise RuntimeError(
                "Previous runner thread is still shutting down after 30 s. "
                "Try again in a moment."
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

    state.update(
        status="starting",
        error=None,
    )

    _runner_thread = threading.Thread(
        target=_run_app,
        args=(app, state),
        daemon=True,
        name="holoscan-runner",
    )
    _runner_thread.start()
    logger.info("Runner thread started")


def stop(state: AppState) -> None:
    """Request the running Holoscan app to flush, save final results, and stop.

    Holoscan's synchronous Application.run() has no public interrupt hook, so
    we ask the pipeline to terminate gracefully via PtychoRecon's natural
    termination path: it sets _finish_event, PtychoRecon.compute() trips the
    iteration-cap branch on the next tick, SaveResult fires write_final, and
    the iteration loop goes quiescent. The runner thread does not exit (a
    fresh /run will need to wait for it), but no more GPU work is queued.
    """
    with state._lock:
        if state.status not in ("starting", "running"):
            raise RuntimeError(f"App is not running (status={state.status!r})")

    from holoptycho import ptycho_holo
    ptycho_holo._finish_event.set()
    state.update(status="stopped")
    logger.info("Stop requested — flushing pipeline and saving final results")
