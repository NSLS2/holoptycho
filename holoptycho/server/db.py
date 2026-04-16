"""SQLite persistence layer for holoptycho server.

Stores:
  - configs: named ptycho config files (JSON key/value, written as INI on disk)
  - settings: single-row key/value store for persistent state
    (selected_config, current_model_name, current_model_version, current_engine_path)

DB location defaults to ./holoptycho.db in the working directory.
Override with HOLOPTYCHO_DB_PATH env var.
"""

from __future__ import annotations

import configparser
import io
import json
import os
import sqlite3
import time
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
            CREATE TABLE IF NOT EXISTS configs (
                name    TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                created REAL NOT NULL,
                updated REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
        """)


# ---------------------------------------------------------------------------
# Config CRUD
# ---------------------------------------------------------------------------

def list_configs() -> list[dict]:
    """Return all config names with timestamps."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT name, created, updated FROM configs ORDER BY name"
        ).fetchall()
    return [dict(r) for r in rows]


def get_config(name: str) -> Optional[dict]:
    """Return config content as a flat JSON dict, or None if not found."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT content FROM configs WHERE name = ?", (name,)
        ).fetchone()
    if row is None:
        return None
    return json.loads(row["content"])


def set_config(name: str, content: dict) -> None:
    """Create or overwrite a config from a flat JSON dict."""
    now = time.time()
    payload = json.dumps(content)
    with _connect() as conn:
        conn.execute("""
            INSERT INTO configs (name, content, created, updated)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET content = excluded.content,
                                            updated = excluded.updated
        """, (name, payload, now, now))


def delete_config(name: str) -> bool:
    """Delete a config. Returns True if it existed."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM configs WHERE name = ?", (name,))
    return cur.rowcount > 0


def rename_config(old_name: str, new_name: str) -> bool:
    """Rename a config. Returns False if old_name doesn't exist or new_name already exists."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT content, created FROM configs WHERE name = ?", (old_name,)
        ).fetchone()
        if row is None:
            return False
        exists = conn.execute(
            "SELECT 1 FROM configs WHERE name = ?", (new_name,)
        ).fetchone()
        if exists:
            return False
        conn.execute("""
            INSERT INTO configs (name, content, created, updated)
            VALUES (?, ?, ?, ?)
        """, (new_name, row["content"], row["created"], time.time()))
        conn.execute("DELETE FROM configs WHERE name = ?", (old_name,))
    return True


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


def ini_to_config(ini_text: str) -> dict:
    """Parse an INI string and return the [GUI] section as a flat dict."""
    cp = configparser.ConfigParser()
    cp.read_string(ini_text)
    if _INI_SECTION not in cp:
        raise ValueError(f"Config file has no [{_INI_SECTION}] section")
    return dict(cp[_INI_SECTION])


def write_config_ini(name: str, config_dir: str) -> str:
    """Write a stored config to an INI file and return the file path."""
    content = get_config(name)
    if content is None:
        raise KeyError(f"Config {name!r} not found in database")
    path = Path(config_dir) / f"{name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config_to_ini(content))
    return str(path)


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
