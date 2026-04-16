"""Thread-safe application state shared between the FastAPI server and the
Holoscan runner thread."""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AppState:
    # Holoscan app lifecycle
    status: str = "stopped"  # stopped | starting | running | finished | error
    mode: Optional[str] = None  # live | simulate
    config_path: Optional[str] = None
    start_time: Optional[float] = None
    error: Optional[str] = None

    # Model swap state
    model_status: str = "ready"  # ready | downloading | compiling | loading | error
    model_error: Optional[str] = None
    current_engine_path: Optional[str] = None
    current_model_name: Optional[str] = None
    current_model_version: Optional[str] = None

    # Log file path (set at server startup)
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
                "config_path": self.config_path,
                "uptime_seconds": uptime,
                "error": self.error,
                "model_status": self.model_status,
                "model_error": self.model_error,
                "current_engine_path": self.current_engine_path,
                "current_model_name": self.current_model_name,
                "current_model_version": self.current_model_version,
                "log_file": self.log_file,
            }


# Module-level singleton shared across the FastAPI app and runner thread.
state = AppState()
