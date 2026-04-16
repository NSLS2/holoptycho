"""FastAPI application for the holoptycho control & monitoring API.

Start with:
    pixi run start-api
or:
    uvicorn holoptycho.server.api:app --host 127.0.0.1 --port 8000
"""

import logging
import logging.handlers
import threading
from pathlib import Path
from typing import Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .state import state
from . import db, runner, model_manager


@asynccontextmanager
async def lifespan(app):
    db.init_db()
    state.update(
        selected_config=db.get_setting("selected_config"),
        current_engine_path=db.get_setting("current_engine_path"),
        current_model_name=db.get_setting("current_model_name"),
        current_model_version=db.get_setting("current_model_version"),
    )
    logger.info("holoptycho API started")
    yield

# ---------------------------------------------------------------------------
# Logging
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
# FastAPI app + startup
# ---------------------------------------------------------------------------
app = FastAPI(title="holoptycho API", version="0.1.0", lifespan=lifespan)



# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    mode: str  # "live" | "simulate"


class ModelSwapRequest(BaseModel):
    name: str
    version: str


class RenameRequest(BaseModel):
    new_name: str


# ---------------------------------------------------------------------------
# Pipeline lifecycle
# ---------------------------------------------------------------------------

@app.get("/status")
def get_status():
    return state.snapshot()


@app.post("/run", status_code=202)
def post_run(req: RunRequest):
    """Start the Holoscan pipeline using the currently selected config."""
    try:
        runner.start(mode=req.mode, state=state)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"detail": f"Starting in {req.mode!r} mode"}


@app.post("/stop", status_code=202)
def post_stop():
    try:
        runner.stop(state=state)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"detail": "Stop requested"}


@app.post("/restart", status_code=202)
def post_restart():
    """Stop the running app and restart it with the same mode."""
    with state._lock:
        mode = state.mode
        current_status = state.status

    if current_status not in ("starting", "running", "finished", "error"):
        raise HTTPException(
            status_code=400,
            detail="No previous run to restart. Use POST /run to start.",
        )
    if mode is None:
        raise HTTPException(status_code=400, detail="No mode recorded — cannot restart.")

    if current_status in ("starting", "running"):
        try:
            runner.stop(state=state)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    import holoptycho.server.runner as runner_mod
    thread = runner_mod._runner_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=30)
        if thread.is_alive():
            raise HTTPException(
                status_code=500,
                detail="Runner thread did not exit within 30 s — cannot restart safely.",
            )

    try:
        runner.start(mode=mode, state=state)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {"detail": f"Restarting in {mode!r} mode"}


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

@app.get("/logs")
def get_logs(lines: int = 100):
    log_path = Path(state.log_file)
    if not log_path.exists():
        return {"lines": []}
    with log_path.open() as f:
        all_lines = f.readlines()
    return {"lines": [l.rstrip("\n") for l in all_lines[-lines:]]}


# ---------------------------------------------------------------------------
# Config management
# ---------------------------------------------------------------------------

@app.get("/config")
def get_config_list():
    """List all configs and show which is currently selected."""
    configs = db.list_configs()
    return {
        "configs": configs,
        "selected": state.selected_config,
    }


@app.get("/config/{name}")
def get_config(name: str):
    """Return a config as a flat JSON dict."""
    content = db.get_config(name)
    if content is None:
        raise HTTPException(status_code=404, detail=f"Config {name!r} not found")
    return content


@app.post("/config/{name}", status_code=201)
def post_config(name: str, content: dict):
    """Create or overwrite a config from a flat JSON dict."""
    db.set_config(name, content)
    return {"detail": f"Config {name!r} saved"}


@app.post("/config/select/{name}", status_code=200)
def post_config_select(name: str):
    """Select a config for the next pipeline run."""
    if db.get_config(name) is None:
        raise HTTPException(status_code=404, detail=f"Config {name!r} not found")
    state.update(selected_config=name)
    db.set_setting("selected_config", name)
    return {"detail": f"Config {name!r} selected"}


@app.post("/config/rename/{name}", status_code=200)
def post_config_rename(name: str, req: RenameRequest):
    """Rename a config."""
    ok = db.rename_config(name, req.new_name)
    if not ok:
        raise HTTPException(
            status_code=400,
            detail=f"Rename failed — {name!r} not found or {req.new_name!r} already exists",
        )
    # Keep selected_config in sync
    if state.selected_config == name:
        state.update(selected_config=req.new_name)
        db.set_setting("selected_config", req.new_name)
    return {"detail": f"Config {name!r} renamed to {req.new_name!r}"}


@app.delete("/config/{name}", status_code=200)
def delete_config(name: str):
    """Delete a config."""
    ok = db.delete_config(name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Config {name!r} not found")
    if state.selected_config == name:
        state.update(selected_config=None)
        db.set_setting("selected_config", None)
    return {"detail": f"Config {name!r} deleted"}


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

@app.post("/model", status_code=202)
def post_model(req: ModelSwapRequest):
    """Trigger async model selection (download + compile if not cached)."""
    if state.model_status in ("downloading", "compiling", "loading"):
        raise HTTPException(
            status_code=409,
            detail=f"Model swap already in progress (status={state.model_status!r})",
        )
    t = threading.Thread(
        target=_swap_model_and_persist,
        args=(req.name, req.version),
        daemon=True,
        name="model-swap",
    )
    t.start()
    return {"detail": f"Model swap started: {req.name} v{req.version}"}


def _swap_model_and_persist(name: str, version: str):
    """Run model swap and persist results to DB."""
    model_manager.swap_model(name, version, state)
    if state.model_status == "ready":
        db.set_setting("current_engine_path", state.current_engine_path)
        db.set_setting("current_model_name", state.current_model_name)
        db.set_setting("current_model_version", state.current_model_version)


@app.get("/model/status")
def get_model_status():
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
    try:
        return model_manager.list_models()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
