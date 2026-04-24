"""SQLite persistence layer for holoptycho server.

Stores:
  - settings: single-row key/value store for persistent state
    (last_config, current_model_name, current_model_version, current_engine_path)

DB location defaults to ./holoptycho.db in the working directory.
Override with HOLOPTYCHO_DB_PATH env var.
"""

from __future__ import annotations

import configparser
import io
import json
import os
import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = os.environ.get("HOLOPTYCHO_DB_PATH", "holoptycho.db")

# INI section used by ptycho config files
_INI_SECTION = "GUI"


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
        """)


# ---------------------------------------------------------------------------
# INI conversion helpers
# ---------------------------------------------------------------------------

def config_to_ini(content: dict) -> str:
    """Convert a flat JSON dict to an INI string with a [GUI] section."""
    cp = configparser.ConfigParser()
    cp[_INI_SECTION] = content
    buf = io.StringIO()
    cp.write(buf)
    return buf.getvalue()


def write_config_ini(content: dict, config_dir: str) -> str:
    """Write a config dict to an INI file and return the file path.

    This file is required because ``ptycho.utils.parse_config`` (an upstream
    dependency we don't control) only accepts a file path, not a dict.
    The JSON config is serialised to INI format here so PtychoApp can read it.
    """
    path = Path(config_dir) / "config.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config_to_ini(content))
    return str(path)


# ---------------------------------------------------------------------------
# Last config
# ---------------------------------------------------------------------------

def get_last_config() -> Optional[dict]:
    """Return the last config as a flat JSON dict, or None if not set."""
    value = get_setting("last_config")
    return json.loads(value) if value is not None else None


def set_last_config(content: dict) -> None:
    """Persist the last config as a JSON blob."""
    set_setting("last_config", json.dumps(content))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def get_setting(key: str) -> Optional[str]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
    return row["value"] if row else None


def set_setting(key: str, value: Optional[str]) -> None:
    with _connect() as conn:
        if value is None:
            conn.execute("DELETE FROM settings WHERE key = ?", (key,))
        else:
            conn.execute("""
                INSERT INTO settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (key, value))
