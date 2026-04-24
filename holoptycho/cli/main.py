"""hp — holoptycho CLI.

Connects to a running holoptycho API server and issues commands.

Base URL defaults to http://localhost:8000.
Override with --url or the HOLOPTYCHO_URL environment variable.
"""

import json
import sys
from typing import Optional

import httpx
import typer
from rich import print as rprint
from rich.table import Table

app = typer.Typer(help="holoptycho control CLI", no_args_is_help=True)
model_app = typer.Typer(help="Model management commands", no_args_is_help=True)
config_app = typer.Typer(help="Config management commands", no_args_is_help=True)
app.add_typer(model_app, name="model")
app.add_typer(config_app, name="config")

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


# ---------------------------------------------------------------------------
# Pipeline lifecycle
# ---------------------------------------------------------------------------

@app.command()
def status(ctx: typer.Context):
    """Show the current status of the Holoscan pipeline."""
    with _client(_base_url(ctx)) as c:
        resp = c.get("/status")
    _handle_error(resp)
    rprint(resp.json())


@app.command()
def start(ctx: typer.Context):
    """Start the Holoscan pipeline using the currently selected config."""
    with _client(_base_url(ctx)) as c:
        resp = c.post("/run")
    _handle_error(resp)
    typer.echo(resp.json().get("detail", "Started"))


@app.command()
def stop(ctx: typer.Context):
    """Stop the running Holoscan pipeline."""
    with _client(_base_url(ctx)) as c:
        resp = c.post("/stop")
    _handle_error(resp)
    typer.echo(resp.json().get("detail", "Stop requested"))


@app.command()
def restart(ctx: typer.Context):
    """Restart the Holoscan pipeline with the same mode."""
    with _client(_base_url(ctx)) as c:
        resp = c.post("/restart")
    _handle_error(resp)
    typer.echo(resp.json().get("detail", "Restarting"))


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
# Config sub-commands
# ---------------------------------------------------------------------------

@config_app.callback()
def config_callback(ctx: typer.Context):
    ctx.ensure_object(dict)
    if ctx.parent and ctx.parent.obj:
        ctx.obj.update(ctx.parent.obj)


@config_app.command("list")
def config_list(ctx: typer.Context):
    """List all configs and show which is selected."""
    with _client(_base_url(ctx)) as c:
        resp = c.get("/config")
    _handle_error(resp)
    data = resp.json()
    selected = data.get("selected")
    configs = data.get("configs", [])
    if not configs:
        typer.echo("No configs found. Use 'hp config set <name> <json>' to add one.")
        return
    table = Table("Name", "Selected", "Updated")
    for cfg in configs:
        is_selected = "yes" if cfg["name"] == selected else ""
        table.add_row(cfg["name"], is_selected, str(cfg["updated"]))
    rprint(table)


@config_app.command("show")
def config_show(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Config name"),
):
    """Print a config as JSON."""
    with _client(_base_url(ctx)) as c:
        resp = c.get(f"/config/{name}")
    _handle_error(resp)
    typer.echo(json.dumps(resp.json(), indent=2))


@config_app.command("set")
def config_set(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Config name"),
    content: str = typer.Argument(..., help="Config as a JSON string"),
):
    """Create or overwrite a config from a JSON string."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        typer.echo(f"Invalid JSON: {e}", err=True)
        raise typer.Exit(1)
    with _client(_base_url(ctx)) as c:
        resp = c.post(f"/config/{name}", json=parsed)
    _handle_error(resp)
    typer.echo(resp.json().get("detail", "Saved"))


@config_app.command("select")
def config_select(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Config name to select"),
):
    """Select a config for the next pipeline run."""
    with _client(_base_url(ctx)) as c:
        resp = c.post(f"/config/select/{name}")
    _handle_error(resp)
    typer.echo(resp.json().get("detail", "Selected"))


@config_app.command("rename")
def config_rename(
    ctx: typer.Context,
    old_name: str = typer.Argument(..., help="Current config name"),
    new_name: str = typer.Argument(..., help="New config name"),
):
    """Rename a config."""
    with _client(_base_url(ctx)) as c:
        resp = c.post(f"/config/rename/{old_name}", json={"new_name": new_name})
    _handle_error(resp)
    typer.echo(resp.json().get("detail", "Renamed"))


@config_app.command("delete")
def config_delete(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Config name to delete"),
):
    """Delete a config."""
    with _client(_base_url(ctx)) as c:
        resp = c.request("DELETE", f"/config/{name}")
    _handle_error(resp)
    typer.echo(resp.json().get("detail", "Deleted"))


# ---------------------------------------------------------------------------
# Model sub-commands
# ---------------------------------------------------------------------------

@model_app.callback()
def model_callback(ctx: typer.Context):
    ctx.ensure_object(dict)
    if ctx.parent and ctx.parent.obj:
        ctx.obj.update(ctx.parent.obj)


@model_app.command("set")
def model_set(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Model name in Azure ML"),
    version: str = typer.Option(..., "--version", "-v", help="Model version"),
):
    """Select a model (downloads and compiles if not cached)."""
    with _client(_base_url(ctx)) as c:
        resp = c.post("/model", json={"name": name, "version": version})
    _handle_error(resp)
    typer.echo(resp.json().get("detail", "Model swap started"))


@model_app.command("status")
def model_status(ctx: typer.Context):
    """Show the current model status."""
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
