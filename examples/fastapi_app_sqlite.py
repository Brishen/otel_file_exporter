"""
FastAPI demo that uses otel_file_exporter with the SQLite backend.

This example simply switches the exporter backend to SQLite and then
re-uses the existing `examples/fastapi_app.py` application object.

Run with:
    python examples\fastapi_app_sqlite.py
The SQLite database will be written to:  telemetry/telemetry.db
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Ensure the exporter backend is SQLite *before* the rest of the application  #
# is imported (exporters are initialised at import time).                     #
# --------------------------------------------------------------------------- #
os.environ.setdefault("EXPORTER_BACKEND", "sqlite")
# You can customise the DB location with SQLITE_DB_PATH – default is fine
# os.environ.setdefault("SQLITE_DB_PATH", "./telemetry/telemetry.db")

# --------------------------------------------------------------------------- #
# Import the existing example application                                     #
# --------------------------------------------------------------------------- #
EXAMPLES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES_DIR))  # make sure we can import fastapi_app
from fastapi_app import app  # noqa: E402  (import after env var change)

# --------------------------------------------------------------------------- #
# Launch with Uvicorn                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn
    from otel_file_exporter.otel import Config

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level=Config.LOG_LEVEL.lower(),
        access_log=True,
    )
