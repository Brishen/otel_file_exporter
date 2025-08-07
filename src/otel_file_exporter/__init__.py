"""
otel_file_exporter package.

Helpers to instrument FastAPI applications with OpenTelemetry and export
traces, logs and metrics to local JSONL files.
"""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("otel_file_exporter")
except _metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

# Re-export public helpers from the sub-module for convenient importing
from .otel import (  # noqa: F401
    Config,
    setup_resource,
    setup_tracing,
    setup_logging,
    setup_metrics,
    tracer,
    logger,
    otel_logger,
    app_metrics,
)

__all__ = [
    "Config",
    "setup_resource",
    "setup_tracing",
    "setup_logging",
    "setup_metrics",
    "tracer",
    "logger",
    "otel_logger",
    "app_metrics",
]
