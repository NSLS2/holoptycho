"""Tests for the hp CLI.

Uses Typer's CliRunner and mocks httpx so no real API server is needed.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from holoptycho.cli.main import app

runner = CliRunner()


def _mock_response(status_code: int, json_data) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.is_error = status_code >= 400
    resp.json.return_value = json_data
    resp.text = str(json_data)
    return resp


def _patch_get(json_data, status_code: int = 200):
    return patch("httpx.Client.get", return_value=_mock_response(status_code, json_data))


def _patch_post(json_data, status_code: int = 202):
    return patch("httpx.Client.post", return_value=_mock_response(status_code, json_data))


# ---------------------------------------------------------------------------
# hp status
# ---------------------------------------------------------------------------

def test_status_command():
    with _patch_get({"status": "stopped", "last_config": None}):
        result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "stopped" in result.output


# ---------------------------------------------------------------------------
# hp start
# ---------------------------------------------------------------------------

def test_start_command():
    with _patch_post({"detail": "Pipeline started"}):
        result = runner.invoke(app, ["start"])
    assert result.exit_code == 0
    assert "Started" in result.output or "Pipeline started" in result.output


def test_start_with_config_json():
    cfg = json.dumps({"scan_num": "320045", "nx": "128"})
    with _patch_post({"detail": "Pipeline started"}):
        result = runner.invoke(app, ["start", cfg])
    assert result.exit_code == 0


def test_start_invalid_json():
    result = runner.invoke(app, ["start", "not valid json"])
    assert result.exit_code == 1
    assert "Invalid JSON" in result.output


def test_start_no_config_error():
    with _patch_post({"detail": "No config provided and no last config found"}, status_code=400):
        result = runner.invoke(app, ["start"])
    assert result.exit_code == 1
    assert "No config" in result.output


# ---------------------------------------------------------------------------
# hp stop / restart
# ---------------------------------------------------------------------------

def test_stop_command():
    with _patch_post({"detail": "Stop requested"}):
        result = runner.invoke(app, ["stop"])
    assert result.exit_code == 0


def test_restart_command():
    with _patch_post({"detail": "Restarting"}):
        result = runner.invoke(app, ["restart"])
    assert result.exit_code == 0
    assert "Restarting" in result.output


def test_restart_with_config_json():
    cfg = json.dumps({"scan_num": "320046", "nx": "256"})
    with _patch_post({"detail": "Restarting"}):
        result = runner.invoke(app, ["restart", cfg])
    assert result.exit_code == 0


def test_restart_invalid_json():
    result = runner.invoke(app, ["restart", "{bad json"]
    )
    assert result.exit_code == 1
    assert "Invalid JSON" in result.output


# ---------------------------------------------------------------------------
# hp config show
# ---------------------------------------------------------------------------

def test_config_show_command():
    cfg = {"scan_num": "320045", "nx": "128"}
    with _patch_get(cfg):
        result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "320045" in result.output


def test_config_show_no_config():
    with _patch_get({"detail": "No config has been set yet."}, status_code=404):
        result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 1
    assert "No config" in result.output


# ---------------------------------------------------------------------------
# hp logs
# ---------------------------------------------------------------------------

def test_logs_command():
    with _patch_get({"lines": ["line 1", "line 2"]}):
        result = runner.invoke(app, ["logs", "--lines", "2"])
    assert result.exit_code == 0
    assert "line 1" in result.output


# ---------------------------------------------------------------------------
# hp model
# ---------------------------------------------------------------------------

def test_model_set_command():
    with _patch_post({"detail": "Model swap started: ptycho_vit v3"}):
        result = runner.invoke(app, ["model", "set", "ptycho_vit", "--version", "3"])
    assert result.exit_code == 0


def test_model_status_command():
    with _patch_get({"model_status": "ready", "current_model_name": "ptycho_vit"}):
        result = runner.invoke(app, ["model", "status"])
    assert result.exit_code == 0
    assert "ready" in result.output


def test_model_list_command():
    data = {
        "local": [{"filename": "ptycho_vit_v3.engine", "size_mb": 120.0}],
        "azure": [],
        "azure_available": False,
    }
    with _patch_get(data):
        result = runner.invoke(app, ["model", "list"])
    assert result.exit_code == 0
    assert "ptycho_vit_v3.engine" in result.output


# ---------------------------------------------------------------------------
# --url flag
# ---------------------------------------------------------------------------

def test_custom_url_flag():
    with _patch_get({"status": "stopped", "last_config": None}):
        result = runner.invoke(app, ["--url", "http://remote:9000", "status"])
    assert result.exit_code == 0
