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
    s.selected_config = None
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
    assert resp.json()["selected_config"] is None


def test_status_reflects_selected_config(client):
    state_module.state.update(selected_config="my_config")
    resp = client.get("/status")
    assert resp.json()["selected_config"] == "my_config"


# ---------------------------------------------------------------------------
# /run
# ---------------------------------------------------------------------------

def test_run_no_config_returns_400(client):
    with patch("holoptycho.server.runner.start", side_effect=RuntimeError("No config selected")):
        resp = client.post("/run")
    assert resp.status_code == 400
    assert "No config selected" in resp.json()["detail"]


def test_run_starts_app(client):
    with patch("holoptycho.server.runner.start") as mock_start:
        resp = client.post("/run")
    assert resp.status_code == 202
    mock_start.assert_called_once_with(state=state_module.state)


def test_run_returns_400_if_already_running(client):
    with patch("holoptycho.server.runner.start", side_effect=RuntimeError("already running")):
        resp = client.post("/run")
    assert resp.status_code == 400


def test_run_blocked_while_thread_alive(client):
    import holoptycho.server.runner as runner_mod
    fake_thread = MagicMock(spec=threading.Thread)
    fake_thread.is_alive.return_value = True
    runner_mod._runner_thread = fake_thread
    try:
        with patch("holoptycho.server.runner.start", side_effect=RuntimeError("shutting down")):
            resp = client.post("/run")
        assert resp.status_code == 400
    finally:
        runner_mod._runner_thread = None


def test_run_no_body_accepted(client):
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
    state_module.state.update(status="running")
    with patch("holoptycho.server.runner.stop"), \
         patch("holoptycho.server.runner.start") as mock_start, \
         patch("holoptycho.server.runner._runner_thread", None):
        resp = client.post("/restart")
    assert resp.status_code == 202
    mock_start.assert_called_once_with(state=state_module.state)


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
# /config
# ---------------------------------------------------------------------------

def test_config_list_empty(client):
    resp = client.get("/config")
    assert resp.status_code == 200
    assert resp.json()["configs"] == []
    assert resp.json()["selected"] is None


def test_config_set_and_get(client):
    content = {"scan_num": "339015", "x_range": "2.0"}
    resp = client.post("/config/my_config", json=content)
    assert resp.status_code == 201

    resp = client.get("/config/my_config")
    assert resp.status_code == 200
    assert resp.json()["scan_num"] == "339015"


def test_config_get_not_found(client):
    resp = client.get("/config/nonexistent")
    assert resp.status_code == 404


def test_config_select(client):
    client.post("/config/my_config", json={"scan_num": "1"})
    resp = client.post("/config/select/my_config")
    assert resp.status_code == 200
    assert state_module.state.selected_config == "my_config"


def test_config_select_not_found(client):
    resp = client.post("/config/select/nonexistent")
    assert resp.status_code == 404


def test_config_select_persisted_to_db(client):
    client.post("/config/my_config", json={"scan_num": "1"})
    client.post("/config/select/my_config")
    assert db_module.get_setting("selected_config") == "my_config"


def test_config_rename(client):
    client.post("/config/old_name", json={"scan_num": "1"})
    resp = client.post("/config/rename/old_name", json={"new_name": "new_name"})
    assert resp.status_code == 200
    assert client.get("/config/old_name").status_code == 404
    assert client.get("/config/new_name").status_code == 200


def test_config_rename_updates_selected(client):
    client.post("/config/my_config", json={"scan_num": "1"})
    client.post("/config/select/my_config")
    client.post("/config/rename/my_config", json={"new_name": "renamed"})
    assert state_module.state.selected_config == "renamed"


def test_config_rename_conflict(client):
    client.post("/config/a", json={"scan_num": "1"})
    client.post("/config/b", json={"scan_num": "2"})
    resp = client.post("/config/rename/a", json={"new_name": "b"})
    assert resp.status_code == 400


def test_config_delete(client):
    client.post("/config/my_config", json={"scan_num": "1"})
    resp = client.request("DELETE", "/config/my_config")
    assert resp.status_code == 200
    assert client.get("/config/my_config").status_code == 404


def test_config_delete_clears_selected(client):
    client.post("/config/my_config", json={"scan_num": "1"})
    client.post("/config/select/my_config")
    client.request("DELETE", "/config/my_config")
    assert state_module.state.selected_config is None


def test_config_delete_not_found(client):
    resp = client.request("DELETE", "/config/nonexistent")
    assert resp.status_code == 404


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
