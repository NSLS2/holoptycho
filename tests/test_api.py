"""Tests for the FastAPI server endpoints.

No GPU or Holoscan SDK required — the runner.start() call is mocked so the
Holoscan app never actually launches.
"""

import json
import os
import threading
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from holoptycho.server import state as state_module
from holoptycho.server import db as db_module

# Minimal valid config for tests
_VALID_CONFIG = {
    "scan_num": "320045",
    "nx": "128", "ny": "128",
    "x_range": "2.0", "y_range": "2.0",
    "x_num": "303", "y_num": "336",
    "det_roix0": "0", "det_roiy0": "0",
    "x_ratio": "-0.0001", "y_ratio": "-0.0001",
    "xray_energy_kev": "15.093",
    "ccd_pixel_um": "75.0",
    "distance": "30.0",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Each test gets its own in-memory-equivalent DB and reset AppState."""
    db_file = str(tmp_path / "test.db")
    monkeypatch.setattr(db_module, "DB_PATH", db_file)
    db_module.init_db()

    s = state_module.state
    s.status = "stopped"
    s.start_time = None
    s.error = None
    s.last_config = None
    s.model_status = "ready"
    s.model_error = None
    s.current_engine_path = None
    s.current_model_name = None
    s.current_model_version = None
    yield


# Import app AFTER patching so the startup event sees the test DB.
@pytest.fixture()
def client(isolated_db):
    from holoptycho.server.api import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------

def test_status_stopped(client):
    resp = client.get("/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "stopped"
    assert resp.json()["last_config"] is None


def test_status_reflects_last_config(client):
    state_module.state.update(last_config=_VALID_CONFIG)
    resp = client.get("/status")
    assert resp.json()["last_config"] == _VALID_CONFIG


# ---------------------------------------------------------------------------
# /config
# ---------------------------------------------------------------------------

def test_get_config_returns_last_config(client):
    state_module.state.update(last_config=_VALID_CONFIG)
    resp = client.get("/config")
    assert resp.status_code == 200
    assert resp.json() == _VALID_CONFIG


def test_get_config_404_when_none(client):
    resp = client.get("/config")
    assert resp.status_code == 404
    assert "No config" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# /run
# ---------------------------------------------------------------------------

def test_run_no_config_no_last_config_returns_400(client):
    with patch("holoptycho.server.runner.start", side_effect=RuntimeError("No config provided")):
        resp = client.post("/run")
    assert resp.status_code == 400
    assert "No config provided" in resp.json()["detail"]


def test_run_with_config_starts_app(client):
    with patch("holoptycho.server.runner.start") as mock_start:
        resp = client.post("/run", json={"config": _VALID_CONFIG})
    assert resp.status_code == 202
    mock_start.assert_called_once_with(state=state_module.state, config=_VALID_CONFIG)


def test_run_no_config_uses_last_config(client):
    state_module.state.update(last_config=_VALID_CONFIG)
    with patch("holoptycho.server.runner.start") as mock_start:
        resp = client.post("/run")
    assert resp.status_code == 202
    mock_start.assert_called_once_with(state=state_module.state, config=None)


def test_run_returns_400_if_already_running(client):
    with patch("holoptycho.server.runner.start", side_effect=RuntimeError("already running")):
        resp = client.post("/run", json={"config": _VALID_CONFIG})
    assert resp.status_code == 400


def test_run_blocked_while_thread_alive(client):
    import holoptycho.server.runner as runner_mod
    fake_thread = MagicMock(spec=threading.Thread)
    fake_thread.is_alive.return_value = True
    runner_mod._runner_thread = fake_thread
    try:
        with patch("holoptycho.server.runner.start", side_effect=RuntimeError("shutting down")):
            resp = client.post("/run", json={"config": _VALID_CONFIG})
        assert resp.status_code == 400
    finally:
        runner_mod._runner_thread = None


def test_run_no_body_uses_last_config(client):
    state_module.state.update(last_config=_VALID_CONFIG)
    with patch("holoptycho.server.runner.start") as mock_start:
        resp = client.post("/run")
    assert resp.status_code == 202


# ---------------------------------------------------------------------------
# /stop
# ---------------------------------------------------------------------------

def test_stop_running_app(client):
    state_module.state.update(status="running")
    with patch("holoptycho.server.runner.stop") as mock_stop:
        resp = client.post("/stop")
    assert resp.status_code == 202
    mock_stop.assert_called_once()


def test_stop_when_not_running(client):
    with patch("holoptycho.server.runner.stop", side_effect=RuntimeError("not running")):
        resp = client.post("/stop")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /restart
# ---------------------------------------------------------------------------

def test_restart_running_app(client):
    state_module.state.update(status="running", last_config=_VALID_CONFIG)
    with patch("holoptycho.server.runner.stop"), \
         patch("holoptycho.server.runner.start") as mock_start, \
         patch("holoptycho.server.runner._runner_thread", None):
        resp = client.post("/restart")
    assert resp.status_code == 202
    mock_start.assert_called_once_with(state=state_module.state, config=None)


def test_restart_with_new_config(client):
    state_module.state.update(status="running", last_config=_VALID_CONFIG)
    new_config = {**_VALID_CONFIG, "scan_num": "320046"}
    with patch("holoptycho.server.runner.stop"), \
         patch("holoptycho.server.runner.start") as mock_start, \
         patch("holoptycho.server.runner._runner_thread", None):
        resp = client.post("/restart", json={"config": new_config})
    assert resp.status_code == 202
    mock_start.assert_called_once_with(state=state_module.state, config=new_config)


def test_restart_no_previous_run(client):
    resp = client.post("/restart")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /logs
# ---------------------------------------------------------------------------

def test_logs_no_file(client, tmp_path):
    state_module.state.log_file = str(tmp_path / "nonexistent.log")
    resp = client.get("/logs")
    assert resp.status_code == 200
    assert resp.json()["lines"] == []


def test_logs_returns_last_n_lines(client, tmp_path):
    log_file = tmp_path / "holoptycho.log"
    log_file.write_text("\n".join(f"line {i}" for i in range(50)) + "\n")
    state_module.state.log_file = str(log_file)
    resp = client.get("/logs?lines=10")
    lines = resp.json()["lines"]
    assert len(lines) == 10
    assert lines[-1] == "line 49"


# ---------------------------------------------------------------------------
# /model
# ---------------------------------------------------------------------------

def test_model_swap_accepted(client):
    with patch("holoptycho.server.model_manager.swap_model"):
        resp = client.post("/model", json={"name": "ptycho_vit", "version": "3"})
    assert resp.status_code == 202


def test_model_swap_conflict(client):
    state_module.state.update(model_status="compiling")
    resp = client.post("/model", json={"name": "ptycho_vit", "version": "3"})
    assert resp.status_code == 409


def test_model_status_ready(client):
    resp = client.get("/model/status")
    assert resp.json()["model_status"] == "ready"


def test_model_list_error(client):
    with patch("holoptycho.server.model_manager.list_models", side_effect=Exception("error")):
        resp = client.get("/model/list")
    assert resp.status_code == 500
