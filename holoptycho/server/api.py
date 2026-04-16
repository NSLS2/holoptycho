"""FastAPI application for the holoptycho control & monitoring API.

Start with:
    pixi run start-api
or:
    uvicorn holoptycho.server.api:app --host 127.0.0.1 --port 8000
"""

import logging
import logging.handlers
import os
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .state import state
from . import runner, model_manager

# ---------------------------------------------------------------------------
# Logging — RotatingFileHandler so /logs can tail it
# ---------------------------------------------------------------------------
_log_file = Path(state.log_file)
_handler = logging.handlers.RotatingFileHandler(
    _log_file, maxBytes=10 * 1024 * 1024, backupCount=3
)
_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
)
logging.getLogger().addHandler(_handler)
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger("holoptycho.api")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="holoptycho API", version="0.1.0")

# Pre-populate engine path from environment variable if provided at startup.
_startup_engine = os.environ.get("HOLOPTYCHO_ENGINE_PATH")
if _startup_engine:
    state.update(current_engine_path=_startup_engine)
    logger.info("Engine path set from environment: %s", _startup_engine)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class RunRequest(BaseModel):
    mode: str  # "live" | "simulate"
    config_path: str
    engine_path: Optional[str] = None  # overrides HOLOPTYCHO_ENGINE_PATH if provided


class ModelSwapRequest(BaseModel):
    name: str
    version: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/status")
def get_status():
    """Return the current state of the Holoscan application."""
    return state.snapshot()


@app.post("/run", status_code=202)
def post_run(req: RunRequest):
    """Start the Holoscan application."""
    try:
        runner.start(config_path=req.config_path, mode=req.mode, state=state, engine_path=req.engine_path)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"detail": f"Starting in {req.mode!r} mode"}


@app.post("/stop", status_code=202)
def post_stop():
    """Request the running Holoscan application to stop."""
    try:
        runner.stop(state=state)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"detail": "Stop requested"}


@app.get("/logs")
def get_logs(lines: int = 100):
    """Return the last N lines from holoptycho.log."""
    log_path = Path(state.log_file)
    if not log_path.exists():
        return {"lines": []}
    with log_path.open() as f:
        all_lines = f.readlines()
    return {"lines": [l.rstrip("\n") for l in all_lines[-lines:]]}


@app.post("/model", status_code=202)
def post_model(req: ModelSwapRequest):
    """Trigger an async model swap. Poll GET /model/status for progress."""
    if state.model_status in ("downloading", "compiling", "loading"):
        raise HTTPException(
            status_code=409,
            detail=f"Model swap already in progress (status={state.model_status!r})",
        )
    t = threading.Thread(
        target=model_manager.swap_model,
        args=(req.name, req.version, state),
        daemon=True,
        name="model-swap",
    )
    t.start()
    return {"detail": f"Model swap started: {req.name} v{req.version}"}


@app.get("/model/status")
def get_model_status():
    """Return the current model swap status."""
    with state._lock:
        return {
            "model_status": state.model_status,
            "model_error": state.model_error,
            "current_engine_path": state.current_engine_path,
            "current_model_name": state.current_model_name,
            "current_model_version": state.current_model_version,
        }


@app.get("/model/list")
def get_model_list():
    """List available models from Azure ML."""
    try:
        return {"models": model_manager.list_models()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
