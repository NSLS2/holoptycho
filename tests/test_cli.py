"""Tests for the hp CLI.

Uses Typer's CliRunner and mocks httpx so no real API server is needed.
"""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from holoptycho.cli.main import app

runner = CliRunner()


def _mock_response(status_code: int, json_data: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.is_error = status_code >= 400
    resp.json.return_value = json_data
    resp.text = str(json_data)
    return resp


def _patch_get(json_data: dict, status_code: int = 200):
    resp = _mock_response(status_code, json_data)
    return patch("httpx.Client.get", return_value=resp)


def _patch_post(json_data: dict, status_code: int = 202):
    resp = _mock_response(status_code, json_data)
    return patch("httpx.Client.post", return_value=resp)


# ---------------------------------------------------------------------------
# hp status
# ---------------------------------------------------------------------------

def test_status_command():
    data = {"status": "stopped", "mode": None, "uptime_seconds": None}
    with _patch_get(data):
        result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "stopped" in result.output


# ---------------------------------------------------------------------------
# hp start
# ---------------------------------------------------------------------------

def test_start_command():
    with _patch_post({"detail": "Starting in 'simulate' mode"}):
        result = runner.invoke(
            app, ["start", "--mode", "simulate", "--config", "/tmp/cfg.txt"]
        )
    assert result.exit_code == 0
    assert "simulate" in result.output


def test_start_error_response():
    with _patch_post({"detail": "App is already running"}, status_code=400):
        result = runner.invoke(
            app, ["start", "--mode", "simulate", "--config", "/tmp/cfg.txt"]
        )
    assert result.exit_code == 1
    assert "400" in result.output


# ---------------------------------------------------------------------------
# hp stop
# ---------------------------------------------------------------------------

def test_stop_command():
    with _patch_post({"detail": "Stop requested"}):
        result = runner.invoke(app, ["stop"])
    assert result.exit_code == 0
    assert "Stop requested" in result.output


# ---------------------------------------------------------------------------
# hp logs
# ---------------------------------------------------------------------------

def test_logs_command():
    lines = ["2026-01-01 INFO starting", "2026-01-01 INFO running"]
    with _patch_get({"lines": lines}):
        result = runner.invoke(app, ["logs", "--lines", "2"])
    assert result.exit_code == 0
    assert "starting" in result.output
    assert "running" in result.output


# ---------------------------------------------------------------------------
# hp model set
# ---------------------------------------------------------------------------

def test_model_set_command():
    with _patch_post({"detail": "Model swap started: ptycho_vit v3"}):
        result = runner.invoke(app, ["model", "set", "ptycho_vit", "--version", "3"])
    assert result.exit_code == 0
    assert "swap" in result.output.lower()


def test_model_set_conflict():
    with _patch_post({"detail": "swap already in progress"}, status_code=409):
        result = runner.invoke(app, ["model", "set", "ptycho_vit", "--version", "3"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# hp model status
# ---------------------------------------------------------------------------

def test_model_status_command():
    data = {
        "model_status": "ready",
        "model_error": None,
        "current_model_name": "ptycho_vit",
        "current_model_version": "3",
        "current_engine_path": "/models/ptycho_vit_v3.engine",
    }
    with _patch_get(data):
        result = runner.invoke(app, ["model", "status"])
    assert result.exit_code == 0
    assert "ready" in result.output


# ---------------------------------------------------------------------------
# hp model list
# ---------------------------------------------------------------------------

def test_model_list_command():
    data = {
        "models": [
            {"name": "ptycho_vit", "version": "3", "description": "VIT model"},
            {"name": "ptycho_vit", "version": "2", "description": None},
        ]
    }
    with _patch_get(data):
        result = runner.invoke(app, ["model", "list"])
    assert result.exit_code == 0
    assert "ptycho_vit" in result.output


def test_model_list_empty():
    with _patch_get({"models": []}):
        result = runner.invoke(app, ["model", "list"])
    assert result.exit_code == 0
    assert "No models found" in result.output


# ---------------------------------------------------------------------------
# --url flag
# ---------------------------------------------------------------------------

def test_custom_url_flag():
    data = {"status": "stopped", "mode": None, "uptime_seconds": None}
    with _patch_get(data) as mock_get:
        result = runner.invoke(app, ["--url", "http://remote-host:9000", "status"])
    assert result.exit_code == 0
