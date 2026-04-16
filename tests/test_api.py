"""Tests for the FastAPI server endpoints.

No GPU or Holoscan SDK required — the runner.start() call is mocked so the
Holoscan app never actually launches.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Reset the singleton state before each test so tests are isolated.
from holoptycho.server import state as state_module
from holoptycho.server.api import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset AppState singleton to defaults before every test."""
    s = state_module.state
    s.status = "stopped"
    s.mode = None
    s.config_path = None
    s.start_time = None
    s.error = None
    s.model_status = "ready"
    s.model_error = None
    s.current_engine_path = None
    s.current_model_name = None
    s.current_model_version = None
    yield


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------

def test_status_stopped():
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "stopped"
    assert data["mode"] is None
    assert data["uptime_seconds"] is None


def test_status_running(reset_state):
    import time
    state_module.state.update(status="running", mode="simulate", start_time=time.time())
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"
    assert data["mode"] == "simulate"
    assert data["uptime_seconds"] is not None
    assert data["uptime_seconds"] >= 0


# ---------------------------------------------------------------------------
# /run
# ---------------------------------------------------------------------------

def test_run_starts_app():
    with patch("holoptycho.server.runner.start") as mock_start:
        resp = client.post("/run", json={"mode": "simulate", "config_path": "/tmp/cfg.txt"})
    assert resp.status_code == 202
    mock_start.assert_called_once_with(
        config_path="/tmp/cfg.txt", mode="simulate", state=state_module.state, engine_path=None,
    )


def test_run_passes_engine_path():
    with patch("holoptycho.server.runner.start") as mock_start:
        resp = client.post(
            "/run",
            json={"mode": "simulate", "config_path": "/tmp/cfg.txt", "engine_path": "/models/custom.engine"},
        )
    assert resp.status_code == 202
    mock_start.assert_called_once_with(
        config_path="/tmp/cfg.txt",
        mode="simulate",
        state=state_module.state,
        engine_path="/models/custom.engine",
    )


def test_run_returns_400_if_already_running():
    with patch("holoptycho.server.runner.start", side_effect=RuntimeError("already running")):
        resp = client.post("/run", json={"mode": "live", "config_path": "/tmp/cfg.txt"})
    assert resp.status_code == 400
    assert "already running" in resp.json()["detail"]


def test_run_blocked_while_thread_alive():
    """start() raises if the previous runner thread is still alive."""
    import threading
    import holoptycho.server.runner as runner_mod

    fake_thread = MagicMock(spec=threading.Thread)
    fake_thread.is_alive.return_value = True
    runner_mod._runner_thread = fake_thread

    try:
        resp = client.post("/run", json={"mode": "simulate", "config_path": "/tmp/cfg.txt"})
        assert resp.status_code == 400
        assert "shutting down" in resp.json()["detail"]
    finally:
        runner_mod._runner_thread = None


def test_run_invalid_mode():
    with patch("holoptycho.server.runner.start", side_effect=ValueError("Unknown mode")):
        resp = client.post("/run", json={"mode": "badmode", "config_path": "/tmp/cfg.txt"})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /stop
# ---------------------------------------------------------------------------

def test_stop_running_app():
    state_module.state.update(status="running")
    with patch("holoptycho.server.runner.stop") as mock_stop:
        resp = client.post("/stop")
    assert resp.status_code == 202
    mock_stop.assert_called_once()


def test_stop_when_not_running():
    with patch("holoptycho.server.runner.stop", side_effect=RuntimeError("not running")):
        resp = client.post("/stop")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /logs
# ---------------------------------------------------------------------------

def test_logs_no_file(tmp_path):
    state_module.state.log_file = str(tmp_path / "nonexistent.log")
    resp = client.get("/logs")
    assert resp.status_code == 200
    assert resp.json()["lines"] == []


def test_logs_returns_last_n_lines(tmp_path):
    log_file = tmp_path / "holoptycho.log"
    log_file.write_text("\n".join(f"line {i}" for i in range(50)) + "\n")
    state_module.state.log_file = str(log_file)

    resp = client.get("/logs?lines=10")
    assert resp.status_code == 200
    lines = resp.json()["lines"]
    assert len(lines) == 10
    assert lines[-1] == "line 49"


# ---------------------------------------------------------------------------
# /model
# ---------------------------------------------------------------------------

def test_model_swap_accepted():
    with patch("holoptycho.server.model_manager.swap_model") as mock_swap:
        resp = client.post("/model", json={"name": "ptycho_vit", "version": "3"})
    assert resp.status_code == 202
    assert "swap started" in resp.json()["detail"]


def test_model_swap_conflict():
    state_module.state.update(model_status="compiling")
    resp = client.post("/model", json={"name": "ptycho_vit", "version": "3"})
    assert resp.status_code == 409


def test_model_status_ready():
    resp = client.get("/model/status")
    assert resp.status_code == 200
    assert resp.json()["model_status"] == "ready"


def test_model_status_after_swap(reset_state):
    state_module.state.update(
        model_status="ready",
        current_model_name="ptycho_vit",
        current_model_version="3",
        current_engine_path="/models/ptycho_vit_v3.engine",
    )
    resp = client.get("/model/status")
    data = resp.json()
    assert data["current_model_name"] == "ptycho_vit"
    assert data["current_model_version"] == "3"


def test_model_list_error():
    with patch(
        "holoptycho.server.model_manager.list_models",
        side_effect=Exception("Azure not configured"),
    ):
        resp = client.get("/model/list")
    assert resp.status_code == 500
