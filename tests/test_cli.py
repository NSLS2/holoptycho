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


def _patch_delete(json_data, status_code: int = 200):
    return patch("httpx.Client.request", return_value=_mock_response(status_code, json_data))


# ---------------------------------------------------------------------------
# hp status
# ---------------------------------------------------------------------------

def test_status_command():
    with _patch_get({"status": "stopped", "selected_config": None}):
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


def test_start_no_config_error():
    with _patch_post({"detail": "No config selected"}, status_code=400):
        result = runner.invoke(app, ["start"])
    assert result.exit_code == 1
    assert "No config selected" in result.output


# ---------------------------------------------------------------------------
# hp stop / restart
# ---------------------------------------------------------------------------

def test_stop_command():
    with _patch_post({"detail": "Stop requested"}):
        result = runner.invoke(app, ["stop"])
    assert result.exit_code == 0


def test_restart_command():
    with _patch_post({"detail": "Restarting in 'simulate' mode"}):
        result = runner.invoke(app, ["restart"])
    assert result.exit_code == 0
    assert "Restarting" in result.output


# ---------------------------------------------------------------------------
# hp logs
# ---------------------------------------------------------------------------

def test_logs_command():
    with _patch_get({"lines": ["line 1", "line 2"]}):
        result = runner.invoke(app, ["logs", "--lines", "2"])
    assert result.exit_code == 0
    assert "line 1" in result.output


# ---------------------------------------------------------------------------
# hp config list
# ---------------------------------------------------------------------------

def test_config_list_command():
    data = {
        "configs": [
            {"name": "hxn_live", "updated": 1234567890.0},
            {"name": "hxn_sim", "updated": 1234567891.0},
        ],
        "selected": "hxn_sim",
    }
    with _patch_get(data):
        result = runner.invoke(app, ["config", "list"])
    assert result.exit_code == 0
    assert "hxn_live" in result.output
    assert "hxn_sim" in result.output


def test_config_list_empty():
    with _patch_get({"configs": [], "selected": None}):
        result = runner.invoke(app, ["config", "list"])
    assert result.exit_code == 0
    assert "No configs found" in result.output


# ---------------------------------------------------------------------------
# hp config show
# ---------------------------------------------------------------------------

def test_config_show_command():
    with _patch_get({"scan_num": "339015", "x_range": "2.0"}):
        result = runner.invoke(app, ["config", "show", "hxn_sim"])
    assert result.exit_code == 0
    assert "339015" in result.output


# ---------------------------------------------------------------------------
# hp config set
# ---------------------------------------------------------------------------

def test_config_set_command():
    with _patch_post({"detail": "Config 'hxn_sim' saved"}, status_code=201):
        result = runner.invoke(
            app, ["config", "set", "hxn_sim", '{"scan_num": "1"}']
        )
    assert result.exit_code == 0
    assert "saved" in result.output


def test_config_set_invalid_json():
    result = runner.invoke(app, ["config", "set", "hxn_sim", "not json"])
    assert result.exit_code == 1
    assert "Invalid JSON" in result.output


# ---------------------------------------------------------------------------
# hp config select
# ---------------------------------------------------------------------------

def test_config_select_command():
    with _patch_post({"detail": "Config 'hxn_sim' selected"}):
        result = runner.invoke(app, ["config", "select", "hxn_sim"])
    assert result.exit_code == 0
    assert "selected" in result.output


# ---------------------------------------------------------------------------
# hp config rename
# ---------------------------------------------------------------------------

def test_config_rename_command():
    with _patch_post({"detail": "Config 'old' renamed to 'new'"}):
        result = runner.invoke(app, ["config", "rename", "old", "new"])
    assert result.exit_code == 0
    assert "renamed" in result.output


# ---------------------------------------------------------------------------
# hp config delete
# ---------------------------------------------------------------------------

def test_config_delete_command():
    with _patch_delete({"detail": "Config 'hxn_sim' deleted"}):
        result = runner.invoke(app, ["config", "delete", "hxn_sim"])
    assert result.exit_code == 0
    assert "deleted" in result.output


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
    with _patch_get({"status": "stopped", "selected_config": None}):
        result = runner.invoke(app, ["--url", "http://remote:9000", "status"])
    assert result.exit_code == 0
