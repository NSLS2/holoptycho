"""hp — holoptycho CLI.

Connects to a running holoptycho API server and issues commands.

Base URL defaults to http://localhost:8000.
Override with --url or the HOLOPTYCHO_URL environment variable.
"""

import json
import sys
import time
from typing import Optional

import httpx
import typer
from rich import print as rprint
from rich.table import Table

app = typer.Typer(help="holoptycho control CLI", no_args_is_help=True)
model_app = typer.Typer(help="Model management commands", no_args_is_help=True)
config_app = typer.Typer(help="Config commands", no_args_is_help=True)
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


def _parse_config(config_str: Optional[str]) -> Optional[dict]:
    """Parse an optional JSON string into a dict, exiting on invalid JSON."""
    if config_str is None:
        return None
    try:
        return json.loads(config_str)
    except json.JSONDecodeError as e:
        typer.echo(f"Invalid JSON: {e}", err=True)
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
def start(
    ctx: typer.Context,
    config: Optional[str] = typer.Argument(
        None,
        help="Config as a JSON string. Uses last config if omitted.",
    ),
):
    """Start the Holoscan pipeline.

    Optionally pass a config JSON string to use for this run.
    If omitted, the current config is reused.
    """
    parsed = _parse_config(config)
    body = {"config": parsed} if parsed is not None else {}
    with _client(_base_url(ctx)) as c:
        resp = c.post("/run", json=body)
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
def restart(
    ctx: typer.Context,
    config: Optional[str] = typer.Argument(
        None,
        help="Config as a JSON string. Uses last config if omitted.",
    ),
):
    """Restart the Holoscan pipeline.

    Optionally pass a new config JSON string. If omitted, the last config is reused.
    """
    parsed = _parse_config(config)
    body = {"config": parsed} if parsed is not None else {}
    with _client(_base_url(ctx)) as c:
        resp = c.post("/restart", json=body)
    _handle_error(resp)
    typer.echo(resp.json().get("detail", "Restarting"))


@app.command()
def logs(
    ctx: typer.Context,
    lines: int = typer.Option(100, "--lines", "-n", help="Number of log lines to show"),
    follow: bool = typer.Option(
        False, "--follow", "-f",
        help="Stream new log lines as they arrive (Ctrl-C to stop).",
    ),
    interval: float = typer.Option(
        1.0, "--interval",
        help="Poll interval in seconds when --follow is used.",
    ),
):
    """Tail the holoptycho log."""
    # Window the server returns on each poll while following. Must be larger
    # than the number of lines the server can append between polls to avoid
    # gaps; safe default for INFO-level traffic at 1s intervals.
    follow_window = max(lines, 1000)

    with _client(_base_url(ctx)) as c:
        resp = c.get("/logs", params={"lines": lines})
        _handle_error(resp)
        current = resp.json().get("lines", [])
        for line in current:
            typer.echo(line)

        if not follow:
            return

        last_line = current[-1] if current else None
        try:
            while True:
                time.sleep(interval)
                resp = c.get("/logs", params={"lines": follow_window})
                _handle_error(resp)
                new_lines = resp.json().get("lines", [])
                # Find the last printed line in the new window and print
                # everything after it. If we can't find it, the log has rolled
                # past our anchor — print the whole window to avoid silent
                # gaps.
                start_idx = 0
                if last_line is not None:
                    for i in range(len(new_lines) - 1, -1, -1):
                        if new_lines[i] == last_line:
                            start_idx = i + 1
                            break
                    else:
                        start_idx = 0
                for line in new_lines[start_idx:]:
                    typer.echo(line)
                if new_lines:
                    last_line = new_lines[-1]
        except KeyboardInterrupt:
            pass


# ---------------------------------------------------------------------------
# Config sub-commands
# ---------------------------------------------------------------------------

@config_app.callback()
def config_callback(ctx: typer.Context):
    ctx.ensure_object(dict)
    if ctx.parent and ctx.parent.obj:
        ctx.obj.update(ctx.parent.obj)


@config_app.command("show")
def config_show(ctx: typer.Context):
    """Print the current config as JSON."""
    with _client(_base_url(ctx)) as c:
        resp = c.get("/config")
    _handle_error(resp)
    typer.echo(json.dumps(resp.json(), indent=2))


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
    version: Optional[str] = typer.Option(None, "--version", "-v", help="Model version (default: latest)"),
):
    """Select a model (downloads and compiles if not cached)."""
    body: dict = {"name": name}
    if version is not None:
        body["version"] = version
    with _client(_base_url(ctx)) as c:
        resp = c.post("/model", json=body)
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
