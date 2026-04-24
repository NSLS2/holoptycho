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


def start(state: AppState) -> None:
    """Start the Holoscan application in a daemon background thread.

    Resolves the config path from the selected config in the database.
    Raises RuntimeError if an app is already running, if a previous runner
    thread is still alive, if no config is selected, or if required ZMQ
    environment variables are not set.
    """
    global _runner_thread

    with state._lock:
        if state.status in ("starting", "running"):
            raise RuntimeError(
                f"App is already {state.status}. Stop it first."
            )

    # Guard against the race where stop() has been called but the thread
    # hasn't exited yet.
    if _runner_thread is not None and _runner_thread.is_alive():
        raise RuntimeError(
            "Previous runner thread is still shutting down. Try again in a moment."
        )

    # Validate required ZMQ env vars before doing anything else.
    missing = [v for v in _REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        raise RuntimeError(
            f"Required environment variable(s) not set: {', '.join(missing)}. "
            "Set SERVER_STREAM_SOURCE and PANDA_STREAM_SOURCE to the ZMQ "
            "endpoints of the Eiger detector and PandA box respectively."
        )

    # Resolve config from DB
    config_name = state.selected_config
    if not config_name:
        raise RuntimeError(
            "No config selected. Run 'hp config select <name>' first."
        )
    config_path = db.write_config_ini(config_name, CONFIG_DIR)
    logger.info("Using config %r written to %s", config_name, config_path)

    # Import heavy GPU deps here so the FastAPI server can start without them.
    try:
        from holoptycho.ptycho_holo import PtychoApp
    except ImportError as exc:
        raise RuntimeError(f"Failed to import Holoscan app: {exc}") from exc

    resolved_engine = state.current_engine_path
    app = PtychoApp(config_path=config_path, engine_path=resolved_engine)

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
    """Request the running Holoscan app to stop."""
    with state._lock:
        if state.status not in ("starting", "running"):
            raise RuntimeError(f"App is not running (status={state.status!r})")

    state.update(status="stopped")
    logger.info("Stop requested — pipeline will finish current iteration")
