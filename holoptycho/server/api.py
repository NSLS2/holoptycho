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

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .state import state
from . import db, runner, model_manager


@asynccontextmanager
async def lifespan(app):
    db.init_db()
    state.update(
        last_config=db.get_last_config(),
        current_engine_path=db.get_setting("current_engine_path"),
        current_model_name=db.get_setting("current_model_name"),
        current_model_version=db.get_setting("current_model_version"),
    )
    logger.info(
        "holoptycho API started (log level=%s)",
        logging.getLevelName(logging.getLogger().level),
    )
    yield


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _resolve_log_level() -> int:
    level_name = os.environ.get("HOLOPTYCHO_LOG_LEVEL", "INFO").upper()
    level = logging.getLevelNamesMapping().get(level_name)
    if isinstance(level, int):
        return level
    logging.getLogger(__name__).warning(
        "Invalid HOLOPTYCHO_LOG_LEVEL=%r; falling back to INFO", level_name
    )
    return logging.INFO


_log_file = Path(state.log_file)
_handler = logging.handlers.RotatingFileHandler(
    _log_file, maxBytes=10 * 1024 * 1024, backupCount=3
)
_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
)
logging.getLogger().addHandler(_handler)
logging.getLogger().setLevel(_resolve_log_level())

logger = logging.getLogger("holoptycho.api")


def _unexpected_startup_error(exc: Exception) -> HTTPException:
    logger.exception("Unexpected pipeline lifecycle error")
    detail = str(exc) or exc.__class__.__name__
    return HTTPException(status_code=500, detail=detail)

# ---------------------------------------------------------------------------
# FastAPI app + startup
# ---------------------------------------------------------------------------
app = FastAPI(title="holoptycho API", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    config: Optional[dict] = None


class ModelSwapRequest(BaseModel):
    name: str
    version: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline lifecycle
# ---------------------------------------------------------------------------

@app.get("/status")
def get_status():
    return state.snapshot()


@app.get("/config")
def get_config():
    """Return the current config, or 404 if none has been set yet."""
    with state._lock:
        cfg = state.last_config
    if cfg is None:
        raise HTTPException(status_code=404, detail="No config has been set yet.")
    return cfg


@app.post("/run", status_code=202)
def post_run(req: RunRequest = RunRequest()):
    """Start the Holoscan pipeline, optionally with a new config."""
    try:
        runner.start(state=state, config=req.config)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise _unexpected_startup_error(exc)
    return {"detail": "Starting pipeline"}


@app.post("/stop", status_code=202)
def post_stop():
    try:
        runner.stop(state=state)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"detail": "Stop requested"}


@app.post("/restart", status_code=202)
def post_restart(req: RunRequest = RunRequest()):
    """Stop the running app and restart it, optionally with a new config."""
    with state._lock:
        current_status = state.status

    if current_status not in ("starting", "running", "finished", "error"):
        raise HTTPException(
            status_code=400,
            detail="No previous run to restart. Use POST /run to start.",
        )

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
        runner.start(state=state, config=req.config)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise _unexpected_startup_error(exc)

    return {"detail": "Restarting pipeline"}


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


@app.post("/logs/clear", status_code=200)
def clear_logs():
    """Truncate the active log file and remove rotated backups."""
    log_path = Path(state.log_file)
    removed = []
    # Truncate the active file via the open handler stream so the running
    # RotatingFileHandler keeps writing to the same fd without losing data.
    if _handler.stream is not None:
        _handler.acquire()
        try:
            _handler.stream.seek(0)
            _handler.stream.truncate()
        finally:
            _handler.release()
    # Sweep rotated backups (holoptycho.log.1, .2, .3 by default).
    for sibling in log_path.parent.glob(log_path.name + ".*"):
        try:
            sibling.unlink()
            removed.append(sibling.name)
        except OSError:
            pass
    logger.info("Logs cleared (removed rotated: %s)", ", ".join(removed) or "none")
    return {"detail": "Logs cleared", "removed_rotated": removed}


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
    version_label = req.version if req.version is not None else "latest"
    return {"detail": f"Model swap started: {req.name} v{version_label}"}


def _swap_model_and_persist(name: str, version: Optional[str]):
    """Run model swap and persist results to DB."""
    model_manager.swap_model(name, version, state)
    if state.model_status == "ready":
        db.init_db()
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
