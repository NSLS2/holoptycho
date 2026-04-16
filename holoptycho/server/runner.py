"""Runs the Holoscan application (PtychoApp or PtychoSimulApp) in a
background thread and keeps AppState in sync with its lifecycle."""

import logging
import threading
import time

from .state import AppState

logger = logging.getLogger("holoptycho.runner")


def _run_app(app, state: AppState):
    """Target function for the runner thread."""
    try:
        state.update(status="running", start_time=time.time(), error=None)
        logger.info("Holoscan app starting (mode=%s)", state.mode)
        app.run()
        state.update(status="finished")
        logger.info("Holoscan app finished normally")
    except Exception as exc:
        state.update(status="error", error=str(exc))
        logger.exception("Holoscan app raised an exception")


def start(config_path: str, mode: str, state: AppState) -> None:
    """Start the Holoscan application in a daemon background thread.

    Raises RuntimeError if an app is already running.
    """
    with state._lock:
        if state.status in ("starting", "running"):
            raise RuntimeError(
                f"App is already {state.status}. Stop it first."
            )

    # Import heavy GPU deps here so the FastAPI server can start without them.
    try:
        from holoptycho.ptycho_holo import PtychoApp, PtychoSimulApp
    except ImportError as exc:
        raise RuntimeError(f"Failed to import Holoscan app: {exc}") from exc

    if mode == "simulate":
        app = PtychoSimulApp(config_path=config_path)
    elif mode == "live":
        app = PtychoApp(config_path=config_path)
    else:
        raise ValueError(f"Unknown mode {mode!r}. Use 'live' or 'simulate'.")

    state.update(
        status="starting",
        mode=mode,
        config_path=config_path,
        error=None,
    )

    thread = threading.Thread(
        target=_run_app,
        args=(app, state),
        daemon=True,
        name="holoscan-runner",
    )
    thread.start()
    logger.info("Runner thread started")


def stop(state: AppState) -> None:
    """Request the running Holoscan app to stop.

    Holoscan does not expose a public stop() on Application, so we set
    status to 'stopped' optimistically; the runner thread will overwrite it
    with 'finished' or 'error' when app.run() returns.
    """
    with state._lock:
        if state.status not in ("starting", "running"):
            raise RuntimeError(f"App is not running (status={state.status!r})")

    # Mark as stopped; the Holoscan pipeline will wind down naturally.
    # For simulate mode this happens quickly; for live mode the operator
    # threads exit on their own schedule.
    state.update(status="stopped")
    logger.info("Stop requested — pipeline will finish current iteration")
