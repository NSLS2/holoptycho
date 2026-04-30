"""Subprocess entry point for a single Holoscan pipeline run.

Spawned by ``holoptycho.server.runner.start()`` via::

    python -m holoptycho.server._pipeline_subprocess

Lifecycle:

1. Read the JSON config from stdin (one line, then EOF).
2. Configure root logging to the same ``holoptycho.log`` the API writes to
   (path passed via ``HOLOPTYCHO_LOG_FILE`` env var). POSIX ``O_APPEND`` makes
   line-sized writes atomic, so the API and subprocess can share the file.
3. Install a SIGUSR1 handler that sets ``ptycho_holo._finish_event`` — the
   parent uses this for graceful "save final and stop" stops before
   escalating to SIGTERM/SIGKILL.
4. Build ``PtychoApp`` and call ``app.run()`` (synchronous).
5. Exit 0 on clean return, 1 on exception (with traceback to stderr/log).

Each subprocess invocation gets its own Python interpreter, CUDA context,
TensorRT/CuPy/numba state. The parent kills it cleanly between runs, so
back-to-back ``/run`` requests are isolated and reliable.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import signal
import sys
import traceback
from pathlib import Path

logger = logging.getLogger("holoptycho.pipeline_subprocess")


def _configure_logging() -> None:
    """Match the API's logging setup so log lines interleave correctly."""
    log_file = os.environ.get("HOLOPTYCHO_LOG_FILE", "holoptycho.log")
    handler = logging.handlers.RotatingFileHandler(
        Path(log_file), maxBytes=10 * 1024 * 1024, backupCount=3
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.addHandler(handler)
    level_name = os.environ.get("HOLOPTYCHO_LOG_LEVEL", "INFO").upper()
    root.setLevel(getattr(logging, level_name, logging.INFO))

    # Same third-party silencing the API does, for symmetry.
    for noisy in ("httpx", "httpcore", "tiled.client", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _install_finish_handler() -> None:
    """SIGUSR1 → ptycho_holo._finish_event.set() for graceful soft stop."""
    def _handler(signum, frame):
        try:
            from holoptycho import ptycho_holo
            ptycho_holo._finish_event.set()
            logger.info("SIGUSR1 received — set _finish_event for graceful stop")
        except Exception:
            logger.exception("Failed to set _finish_event on SIGUSR1")

    signal.signal(signal.SIGUSR1, _handler)


def main() -> int:
    _configure_logging()
    _install_finish_handler()

    raw = sys.stdin.read()
    if not raw.strip():
        logger.error("No config received on stdin")
        return 2
    try:
        config = json.loads(raw)
    except json.JSONDecodeError:
        logger.exception("Invalid JSON config on stdin")
        return 2

    config_path = os.environ.get("HOLOPTYCHO_CONFIG_PATH")
    engine_path = os.environ.get("HOLOPTYCHO_ENGINE_PATH") or None

    logger.info(
        "pipeline subprocess starting (config_path=%s, engine_path=%s)",
        config_path, engine_path,
    )

    try:
        from holoptycho.ptycho_holo import PtychoApp
    except ImportError:
        logger.exception("Failed to import PtychoApp")
        return 3

    try:
        app = PtychoApp(
            config_path=config_path,
            config_overrides=config,
            engine_path=engine_path,
        )
    except Exception:
        logger.exception("Failed to construct PtychoApp")
        return 4

    try:
        app.run()
    except Exception:
        logger.exception("PtychoApp.run() raised")
        traceback.print_exc(file=sys.stderr)
        return 5

    logger.info("pipeline subprocess finished cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
