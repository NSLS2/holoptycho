"""Thread-safe application state shared between the FastAPI server and the
Holoscan runner thread."""

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

# Where the server writes temporary INI files for the Holoscan app to read.
CONFIG_DIR = os.environ.get("HOLOPTYCHO_CONFIG_DIR", "configs")


@dataclass
class AppState:
    # Holoscan app lifecycle
    status: str = "stopped"  # stopped | starting | running | finished | error
    mode: Optional[str] = None  # live | simulate
    start_time: Optional[float] = None
    error: Optional[str] = None

    # Config selection (persisted in DB)
    selected_config: Optional[str] = None  # name of selected config

    # Model (persisted in DB)
    model_status: str = "ready"  # ready | downloading | compiling | loading | error
    model_error: Optional[str] = None
    current_engine_path: Optional[str] = None
    current_model_name: Optional[str] = None
    current_model_version: Optional[str] = None

    # Log file path
    log_file: str = "holoptycho.log"

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kwargs):
        """Thread-safe bulk update of fields."""
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        """Return a JSON-serialisable dict of all public fields."""
        with self._lock:
            uptime = (
                time.time() - self.start_time
                if self.start_time is not None
                else None
            )
            return {
                "status": self.status,
                "mode": self.mode,
                "uptime_seconds": uptime,
                "error": self.error,
                "selected_config": self.selected_config,
                "model_status": self.model_status,
                "model_error": self.model_error,
                "current_engine_path": self.current_engine_path,
                "current_model_name": self.current_model_name,
                "current_model_version": self.current_model_version,
                "log_file": self.log_file,
            }


# Module-level singleton shared across the FastAPI app and runner thread.
state = AppState()
