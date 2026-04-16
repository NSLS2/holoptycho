"""hp — holoptycho CLI.

Connects to a running holoptycho API server and issues commands.

Base URL defaults to http://localhost:8000.
Override with --url or the HOLOPTYCHO_URL environment variable.

Usage examples:
    hp status
    hp start --mode simulate --config /path/to/config.txt
    hp stop
    hp logs --lines 50
    hp model set ptycho_vit --version 3
    hp model status
    hp model list
"""

import os
import sys
from typing import Optional

import httpx
import typer
from rich import print as rprint
from rich.table import Table
import json

app = typer.Typer(help="holoptycho control CLI", no_args_is_help=True)
model_app = typer.Typer(help="Model management commands", no_args_is_help=True)
app.add_typer(model_app, name="model")

_DEFAULT_URL = "http://localhost:8000"


def _base_url(ctx: typer.Context) -> str:
    return ctx.obj.get("url", _DEFAULT_URL) if ctx.obj else _DEFAULT_URL


def _client(url: str) -> httpx.Client:
    return httpx.Client(base_url=url, timeout=30.0)


def _handle_error(resp: httpx.Response) -> None:
    if resp.is_error:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        typer.echo(f"Error {resp.status_code}: {detail}", err=True)
        raise typer.Exit(1)


@app.callback()
def main(
    ctx: typer.Context,
    url: str = typer.Option(
        None,
        "--url",
        envvar="HOLOPTYCHO_URL",
        help="API base URL (default: http://localhost:8000)",
    ),
):
    ctx.ensure_object(dict)
    ctx.obj["url"] = url or _DEFAULT_URL


@app.command()
def status(ctx: typer.Context):
    """Show the current status of the Holoscan application."""
    with _client(_base_url(ctx)) as c:
        resp = c.get("/status")
    _handle_error(resp)
    data = resp.json()
    rprint(data)


@app.command()
def start(
    ctx: typer.Context,
    mode: str = typer.Option(..., "--mode", help="'live' or 'simulate'"),
    config: str = typer.Option(..., "--config", help="Path to ptycho config file"),
):
    """Start the Holoscan application."""
    with _client(_base_url(ctx)) as c:
        resp = c.post("/run", json={"mode": mode, "config_path": config})
    _handle_error(resp)
    typer.echo(resp.json().get("detail", "Started"))


@app.command()
def restart(ctx: typer.Context):
    """Restart the Holoscan application with the same mode and config."""
    with _client(_base_url(ctx)) as c:
        resp = c.post("/restart")
    _handle_error(resp)
    typer.echo(resp.json().get("detail", "Restarting"))


@app.command()
def stop(ctx: typer.Context):
    """Stop the running Holoscan application."""
    with _client(_base_url(ctx)) as c:
        resp = c.post("/stop")
    _handle_error(resp)
    typer.echo(resp.json().get("detail", "Stop requested"))


@app.command()
def logs(
    ctx: typer.Context,
    lines: int = typer.Option(100, "--lines", "-n", help="Number of log lines to show"),
):
    """Tail the holoptycho log."""
    with _client(_base_url(ctx)) as c:
        resp = c.get("/logs", params={"lines": lines})
    _handle_error(resp)
    for line in resp.json().get("lines", []):
        typer.echo(line)


# ---------------------------------------------------------------------------
# model sub-commands
# ---------------------------------------------------------------------------

@model_app.command("set")
def model_set(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Model name in Azure ML"),
    version: str = typer.Option(..., "--version", "-v", help="Model version"),
):
    """Trigger an async model swap."""
    with _client(_base_url(ctx)) as c:
        resp = c.post("/model", json={"name": name, "version": version})
    _handle_error(resp)
    typer.echo(resp.json().get("detail", "Model swap started"))


@model_app.command("status")
def model_status(ctx: typer.Context):
    """Show the current model swap status."""
    with _client(_base_url(ctx)) as c:
        resp = c.get("/model/status")
    _handle_error(resp)
    rprint(resp.json())


@model_app.command("list")
def model_list(ctx: typer.Context):
    """List available models (local cache and Azure ML)."""
    with _client(_base_url(ctx)) as c:
        resp = c.get("/model/list")
    _handle_error(resp)
    data = resp.json()

    local = data.get("local", [])
    azure = data.get("azure", [])
    azure_available = data.get("azure_available", False)

    typer.echo("Local cache:")
    if local:
        table = Table("File", "Size (MB)")
        for m in local:
            table.add_row(m["filename"], str(m["size_mb"]))
        rprint(table)
    else:
        typer.echo("  (no .engine files found in model folder)")

    typer.echo("")
    if not azure_available:
        typer.echo("Azure ML: not configured (set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE)")
    else:
        typer.echo("Azure ML:")
        if azure:
            table = Table("Name", "Version", "Cached", "Description")
            for m in azure:
                cached = "yes" if m.get("cached") else "no"
                table.add_row(m["name"], str(m["version"]), cached, m.get("description") or "")
            rprint(table)
        else:
            typer.echo("  (no models found in Azure ML)")
