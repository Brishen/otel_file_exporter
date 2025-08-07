import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from opentelemetry import metrics, trace
from opentelemetry._logs import get_logger, set_logger_provider
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.sdk._logs import LogData, LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs._internal.export import LogExportResult
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    LogExporter,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics._internal.aggregation import AggregationTemporality
from opentelemetry.sdk.metrics.export import (
    MetricExporter,
    MetricExportResult,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from sqlalchemy import (Column, create_engine, Float, Integer, MetaData, String, Table, text, Text)


class Config:
    SERVICE_NAME = os.getenv("SERVICE_NAME", "fastapi-otel-demo")
    SERVICE_VERSION = os.getenv("SERVICE_VERSION", "1.0.0")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    TRACE_SAMPLE_RATE = float(os.getenv("TRACE_SAMPLE_RATE", "1.0"))
    # Export metrics every 5 seconds by default so they appear quickly in the DB
    METRICS_EXPORT_INTERVAL = int(os.getenv("METRICS_EXPORT_INTERVAL", "5000"))
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./telemetry"))
    EXPORTER_BACKEND = os.getenv("EXPORTER_BACKEND", "file").lower()
    SQLITE_DB_PATH = Path(os.getenv("SQLITE_DB_PATH", "./telemetry/telemetry.db"))


# ─── Enhanced File Exporters ───────────────────────────────────────────────────
class ThreadSafeFileExporter:
    def __init__(self, filename: str):
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._filepath = Config.OUTPUT_DIR / filename
        self._file = None
        self._open_file()

    def _open_file(self):
        if self._file is None or self._file.closed:
            self._file = self._filepath.open("a", encoding="utf-8", buffering=1)

    def _write_json_line(self, data: Dict[str, Any]):
        """Write JSON object as single line with error handling"""
        try:
            self._open_file()
            json.dump(
                data, self._file, separators=(",", ":"), ensure_ascii=False, default=str
            )
            self._file.write("\n")
            self._file.flush()
        except Exception as e:
            logging.error(f"Failed to write to {self._filepath}: {e}")

    def shutdown(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()


class EnhancedFileSpanExporter(ThreadSafeFileExporter, SpanExporter):
    def __init__(self, filename: str = "traces.jsonl"):
        super().__init__(filename)

    def export(self, spans: Sequence) -> SpanExportResult:
        try:
            for span in spans:
                span_data = {
                    "trace_id": format(span.context.trace_id, "032x"),
                    "span_id": format(span.context.span_id, "016x"),
                    "name": span.name,
                    "start_time": span.start_time,
                    "end_time": span.end_time,
                    "duration_ms": (span.end_time - span.start_time) / 1_000_000,
                    "status": {
                        "code": span.status.status_code.name,
                        "message": span.status.description,
                    },
                    "attributes": dict(span.attributes) if span.attributes else {},
                    "events": [
                        {
                            "name": event.name,
                            "timestamp": event.timestamp,
                            "attributes": dict(event.attributes)
                            if event.attributes
                            else {},
                        }
                        for event in span.events
                    ]
                    if span.events
                    else [],
                    "resource": dict(span.resource.attributes) if span.resource else {},
                }
                self._write_json_line(span_data)
            return SpanExportResult.SUCCESS
        except Exception as e:
            logging.error(f"Failed to export spans: {e}")
            return SpanExportResult.FAILURE


class EnhancedFileLogExporter(ThreadSafeFileExporter, LogExporter):
    """Simple console log exporter"""

    def shutdown(self) -> None:
        pass

    def __init__(self, filename: str = "logs.jsonl"):
        super().__init__(filename)

    def export(self, batch: Sequence[LogData]) -> LogExportResult:
        try:
            for log_data in batch:
                lr = log_data.log_record

                # Extract timestamp - different LogRecord implementations may vary
                timestamp = None
                if hasattr(lr, "timestamp"):
                    timestamp = lr.timestamp
                elif hasattr(lr, "observed_timestamp"):
                    timestamp = lr.observed_timestamp
                else:
                    # Fallback to current time in nanoseconds
                    timestamp = int(time.time() * 1_000_000_000)

                # Extract severity/level
                level = "INFO"
                if hasattr(lr, "severity_text") and lr.severity_text:
                    level = lr.severity_text
                elif hasattr(lr, "severity_number"):
                    # Map severity numbers to text
                    severity_map = {
                        1: "TRACE",
                        2: "TRACE2",
                        3: "TRACE3",
                        4: "TRACE4",
                        5: "DEBUG",
                        6: "DEBUG2",
                        7: "DEBUG3",
                        8: "DEBUG4",
                        9: "INFO",
                        10: "INFO2",
                        11: "INFO3",
                        12: "INFO4",
                        13: "WARN",
                        14: "WARN2",
                        15: "WARN3",
                        16: "WARN4",
                        17: "ERROR",
                        18: "ERROR2",
                        19: "ERROR3",
                        20: "ERROR4",
                        21: "FATAL",
                        22: "FATAL2",
                        23: "FATAL3",
                        24: "FATAL4",
                    }
                    level = severity_map.get(lr.severity_number, "INFO")

                log_entry = {
                    "timestamp": timestamp,
                    "level": level,
                    "message": str(lr.body) if hasattr(lr, "body") else "",
                    "trace_id": format(lr.trace_id, "032x")
                    if hasattr(lr, "trace_id") and lr.trace_id and lr.trace_id != 0
                    else None,
                    "span_id": format(lr.span_id, "016x")
                    if hasattr(lr, "span_id") and lr.span_id and lr.span_id != 0
                    else None,
                    "attributes": dict(lr.attributes)
                    if hasattr(lr, "attributes") and lr.attributes
                    else {},
                    "resource": dict(lr.resource.attributes)
                    if hasattr(lr, "resource")
                       and lr.resource
                       and lr.resource.attributes
                    else {},
                }
                self._write_json_line(log_entry)
            return LogExportResult.SUCCESS
        except Exception as e:
            # Use Python's logging to avoid recursion
            import sys

            print(f"ERROR: Failed to export logs: {e}", file=sys.stderr)
            return LogExportResult.FAILURE

    def force_flush(self) -> bool:
        try:
            if self._file and not self._file.closed:
                self._file.flush()
            return True
        except Exception:
            return False


class EnhancedFileMetricExporter(ThreadSafeFileExporter, MetricExporter):
    def __init__(
            self,
            filename: str = "metrics.jsonl",
            preferred_temporality: Optional[Dict[type, AggregationTemporality]] = None,
            preferred_aggregation: Optional[Dict[type, object]] = None,
    ):
        super().__init__(filename)
        MetricExporter.__init__(
            self,
            preferred_temporality=preferred_temporality,
            preferred_aggregation=preferred_aggregation,
        )

    def export(
            self,
            metrics_data: Any,
            timeout_millis: float = 10000,
            **kwargs,
    ) -> MetricExportResult:
        try:
            # Handle different metrics data structures
            if hasattr(metrics_data, "resource_metrics"):
                # New structure with resource_metrics attribute
                resource_metrics = metrics_data.resource_metrics
            else:
                # Assume metrics_data is already iterable
                resource_metrics = metrics_data

            for resource_metric in resource_metrics:
                for scope_metric in resource_metric.scope_metrics:
                    for metric in scope_metric.metrics:
                        metric_data = {
                            "name": metric.name,
                            "description": metric.description or "",
                            "unit": metric.unit or "",
                            "type": type(metric.data).__name__,
                            "resource": dict(resource_metric.resource.attributes)
                            if resource_metric.resource
                               and resource_metric.resource.attributes
                            else {},
                            "scope": {
                                "name": scope_metric.scope.name
                                if scope_metric.scope
                                else "",
                                "version": scope_metric.scope.version
                                if scope_metric.scope
                                else "",
                            },
                            "data_points": [],
                            "timestamp": int(
                                time.time() * 1000
                            ),  # Add timestamp for debugging
                        }

                        # Handle different data point types
                        data_points = getattr(metric.data, "data_points", [])
                        for point in data_points:
                            point_data = {
                                "attributes": dict(point.attributes)
                                if hasattr(point, "attributes") and point.attributes
                                else {},
                                "start_time": getattr(point, "start_time_unix_nano", 0),
                                "time": getattr(point, "time_unix_nano", 0),
                            }

                            # Add value based on point type
                            if hasattr(point, "value"):
                                point_data["value"] = point.value
                            if hasattr(point, "sum"):
                                point_data["sum"] = point.sum
                            if hasattr(point, "count"):
                                point_data["count"] = point.count
                            if hasattr(point, "bucket_counts"):
                                point_data["bucket_counts"] = list(point.bucket_counts)
                            if hasattr(point, "explicit_bounds"):
                                point_data["explicit_bounds"] = list(
                                    point.explicit_bounds
                                )
                            if hasattr(point, "min"):
                                point_data["min"] = point.min
                            if hasattr(point, "max"):
                                point_data["max"] = point.max

                            metric_data["data_points"].append(point_data)

                        self._write_json_line(metric_data)
            return MetricExportResult.SUCCESS
        except Exception as e:
            # Use print to avoid recursion
            import sys

            print(f"ERROR: Failed to export metrics: {e}", file=sys.stderr)
            print(f"Metrics data type: {type(metrics_data)}", file=sys.stderr)
            if hasattr(metrics_data, "__dict__"):
                print(
                    f"Metrics data attributes: {list(metrics_data.__dict__.keys())}",
                    file=sys.stderr,
                )
            return MetricExportResult.FAILURE

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        try:
            if self._file and not self._file.closed:
                self._file.flush()
            return True
        except Exception:
            return False

    def shutdown(self, timeout_millis: int = 30000) -> None:
        """Shutdown the exporter - note: timeout_millis parameter for compatibility"""
        super().shutdown()


class SimpleConsoleLogExporter(LogExporter):
    """Simple console log exporter that doesn't rely on to_json()"""

    def export(self, batch: Sequence[LogData]) -> LogExportResult:
        try:
            for log_data in batch:
                lr = log_data.log_record

                # Extract basic info safely
                timestamp = getattr(lr, "timestamp", int(time.time() * 1_000_000_000))
                level = getattr(lr, "severity_text", "INFO")
                message = str(getattr(lr, "body", ""))

                # Simple console output
                print(f"[{level}] {message}")
            return LogExportResult.SUCCESS
        except Exception as e:
            print(f"Console log export error: {e}")
            return LogExportResult.FAILURE

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def shutdown(self) -> None:
        pass


# ─── Enhanced SQLite Exporters ─────────────────────────────────────────────────
class SQLiteExporterBase:
    """
    Thread-safe base exporter that maintains a single SQLAlchemy engine.
    """

    _engine = None
    _lock = threading.Lock()
    # Pre-declare attributes so static type checkers recognize them
    _tables: Dict[str, Table] = {}
    _metadata: Optional[MetaData] = None

    def __init__(self, db_path: Path = Config.SQLITE_DB_PATH):
        # Ensure the base output directory exists
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------------------- #
        # Resolve the DB path, avoiding "telemetry/telemetry/telemetry.db"  #
        # ----------------------------------------------------------------- #
        db_path = Path(db_path)
        if not db_path.is_absolute() and db_path.parent == Path("."):
            # Only prepend OUTPUT_DIR when the caller provided a bare filename
            db_path = Config.OUTPUT_DIR / db_path

        # Create the containing directory in case it does not yet exist
        db_path.parent.mkdir(parents=True, exist_ok=True)

        if SQLiteExporterBase._engine is None:
            SQLiteExporterBase._engine = create_engine(
                f"sqlite:///{db_path}",
                connect_args={"check_same_thread": False},
                future=True,
            )
            # Create tables once
            SQLiteExporterBase._init_schema()

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _init_schema():
        """Create tables using SQLAlchemy Core – executed once per process."""
        metadata = MetaData()

        spans_table = Table(
            "spans",
            metadata,
            Column("trace_id", String),
            Column("span_id", String),
            Column("name", String),
            Column("start_time", Integer),
            Column("end_time", Integer),
            Column("duration_ms", Float),
            Column("status_code", String),
            Column("status_message", String),
            Column("attributes", Text),
            Column("events", Text),
            Column("resource", Text),
        )

        logs_table = Table(
            "logs",
            metadata,
            Column("timestamp", Integer),
            Column("level", String),
            Column("message", Text),
            Column("trace_id", String, nullable=True),
            Column("span_id", String, nullable=True),
            Column("attributes", Text),
            Column("resource", Text),
        )

        metrics_table = Table(
            "metrics",
            metadata,
            Column("name", String),
            Column("description", Text),
            Column("unit", String),
            Column("type", String),
            Column("resource", Text),
            Column("scope_name", String),
            Column("scope_version", String),
            Column("attributes", Text),
            Column("value", Float),
            Column("start_time", Integer),
            Column("time", Integer),
            Column("timestamp", Integer),
        )

        # Create all tables
        metadata.create_all(SQLiteExporterBase._engine)

        # Stash metadata & tables for optional future use
        SQLiteExporterBase._metadata = metadata
        SQLiteExporterBase._tables = {
            "spans": spans_table,
            "logs": logs_table,
            "metrics": metrics_table,
        }

    def _execute(self, sql: str, params: tuple):
        """Execute parametrized SQL using SQLAlchemy Core."""
        with self._lock:
            with SQLiteExporterBase._engine.begin() as conn:
                conn.execute(text(sql), params)

    # ----------------------------------------------------------------- #
    # Public helper to insert rows with SQLAlchemy Core                  #
    # ----------------------------------------------------------------- #
    def _insert(self, table_name: str, rows: Sequence[Dict[str, Any]]):
        """
        Bulk-insert one or more rows into the given table using SQLAlchemy Core.

        Args:
            table_name: Name of the target table (``spans``, ``logs`` or ``metrics``)
            rows:      Sequence of dictionaries keyed by column name.
        """
        if not rows:
            return
        table = SQLiteExporterBase._tables[table_name]
        with self._lock:
            with SQLiteExporterBase._engine.begin() as conn:
                conn.execute(table.insert(), rows)

    # --------------------------------------------------------------------- #
    # OpenTelemetry hooks
    # --------------------------------------------------------------------- #
    def force_flush(self, timeout_millis: int = 30000) -> bool:  # compatibility
        return True

    def shutdown(self) -> None:
        with self._lock:
            if SQLiteExporterBase._engine:
                SQLiteExporterBase._engine.dispose()
                SQLiteExporterBase._engine = None


class EnhancedSQLiteSpanExporter(SQLiteExporterBase, SpanExporter):
    def _setup_schema(self):
        # Schema is created once globally by SQLiteExporterBase._init_schema()
        return

    def export(self, spans: Sequence) -> SpanExportResult:
        try:
            rows = []
            for span in spans:
                rows.append(
                    {
                        "trace_id": format(span.context.trace_id, "032x"),
                        "span_id": format(span.context.span_id, "016x"),
                        "name": span.name,
                        "start_time": span.start_time,
                        "end_time": span.end_time,
                        "duration_ms": (span.end_time - span.start_time) / 1_000_000,
                        "status_code": span.status.status_code.name,
                        "status_message": span.status.description,
                        "attributes": json.dumps(span.attributes or {}, default=str),
                        "events": json.dumps(
                            [
                                {
                                    "name": e.name,
                                    "timestamp": e.timestamp,
                                    "attributes": dict(e.attributes)
                                    if e.attributes
                                    else {},
                                }
                                for e in span.events
                            ],
                            default=str,
                        ),
                        "resource": json.dumps(
                            span.resource.attributes if span.resource else {},
                            default=str,
                        ),
                    }
                )
            self._insert("spans", rows)
            return SpanExportResult.SUCCESS
        except Exception as e:
            logging.error(f"Failed to export spans to SQLite: {e}")
            return SpanExportResult.FAILURE


class EnhancedSQLiteLogExporter(SQLiteExporterBase, LogExporter):
    def _setup_schema(self):
        # Schema is created once globally by SQLiteExporterBase._init_schema()
        return

    def export(self, batch: Sequence[LogData]) -> LogExportResult:
        try:
            rows = []
            for log_data in batch:
                lr = log_data.log_record
                timestamp = getattr(lr, "timestamp", int(time.time() * 1_000_000_000))
                level = getattr(lr, "severity_text", "INFO")
                message = str(getattr(lr, "body", ""))
                trace_id = (
                    format(lr.trace_id, "032x") if getattr(lr, "trace_id", 0) else None
                )
                span_id = (
                    format(lr.span_id, "016x") if getattr(lr, "span_id", 0) else None
                )

                rows.append(
                    {
                        "timestamp": timestamp,
                        "level": level,
                        "message": message,
                        "trace_id": trace_id,
                        "span_id": span_id,
                        "attributes": json.dumps(
                            getattr(lr, "attributes", {}), default=str
                        ),
                        "resource": json.dumps(
                            getattr(lr.resource, "attributes", {}) if getattr(lr, "resource",
                                                                              None) else {},
                            default=str
                        ),
                    }
                )
            self._insert("logs", rows)
            return LogExportResult.SUCCESS
        except Exception as e:
            logging.error(f"Failed to export logs to SQLite: {e}")
            return LogExportResult.FAILURE


class EnhancedSQLiteMetricExporter(SQLiteExporterBase, MetricExporter):
    def __init__(
            self,
            preferred_temporality: Optional[Dict[type, AggregationTemporality]] = None,
            preferred_aggregation: Optional[Dict[type, object]] = None,
    ):
        SQLiteExporterBase.__init__(self)
        MetricExporter.__init__(
            self,
            preferred_temporality=preferred_temporality,
            preferred_aggregation=preferred_aggregation,
        )

    def _setup_schema(self):
        # Schema is created once globally by SQLiteExporterBase._init_schema()
        return

    def export(
            self,
            metrics_data: Any,
            timeout_millis: float = 10000,
            **kwargs,
    ) -> MetricExportResult:
        try:
            resource_metrics = (
                metrics_data.resource_metrics
                if hasattr(metrics_data, "resource_metrics")
                else metrics_data
            )
            now_ms = int(time.time() * 1000)
            rows = []

            for resource_metric in resource_metrics:
                res_json = json.dumps(
                    resource_metric.resource.attributes
                    if resource_metric.resource
                    else {},
                    default=str,
                )
                for scope_metric in resource_metric.scope_metrics:
                    scope_name = scope_metric.scope.name if scope_metric.scope else ""
                    scope_version = (
                        scope_metric.scope.version if scope_metric.scope else ""
                    )
                    for metric in scope_metric.metrics:
                        data_type = type(metric.data).__name__
                        for point in getattr(metric.data, "data_points", []):
                            value = getattr(point, "value", None)
                            if value is None and hasattr(point, "sum"):
                                value = point.sum
                            attrs_json = json.dumps(point.attributes or {}, default=str)

                            rows.append(
                                {
                                    "name": metric.name,
                                    "description": metric.description or "",
                                    "unit": metric.unit or "",
                                    "type": data_type,
                                    "resource": res_json,
                                    "scope_name": scope_name,
                                    "scope_version": scope_version,
                                    "attributes": attrs_json,
                                    "value": value,
                                    "start_time": getattr(
                                        point, "start_time_unix_nano", 0
                                    ),
                                    "time": getattr(point, "time_unix_nano", 0),
                                    "timestamp": now_ms,
                                }
                            )
            self._insert("metrics", rows)
            return MetricExportResult.SUCCESS
        except Exception as e:
            logging.error(f"Failed to export metrics to SQLite: {e}")
            return MetricExportResult.FAILURE


# ─── Observability Setup ───────────────────────────────────────────────────────
def setup_resource() -> Resource:
    """Create a resource with service information"""
    return Resource.create(
        {
            SERVICE_NAME: Config.SERVICE_NAME,
            SERVICE_VERSION: Config.SERVICE_VERSION,
            "environment": Config.ENVIRONMENT,
            "host.name": os.getenv("HOSTNAME", Path.cwd().name),
        }
    )


def setup_tracing(resource: Resource):
    """Configure distributed tracing"""
    # Set up trace propagation
    set_global_textmap(B3MultiFormat())

    # Configure tracer provider with sampling
    tracer_provider = TracerProvider(
        resource=resource, sampler=TraceIdRatioBased(Config.TRACE_SAMPLE_RATE)
    )
    trace.set_tracer_provider(tracer_provider)

    # Add only our custom exporters to avoid compatibility issues
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            EnhancedSQLiteSpanExporter()
            if Config.EXPORTER_BACKEND == "sqlite"
            else EnhancedFileSpanExporter(),
            max_queue_size=2048,
            max_export_batch_size=512,
            export_timeout_millis=30000,
        )
    )

    return trace.get_tracer(__name__)


def setup_logging(resource: Resource):
    """Configure structured logging"""
    log_provider = LoggerProvider(resource=resource)
    set_logger_provider(log_provider)

    # Add only our custom processors - avoid problematic ConsoleLogExporter
    log_provider.add_log_record_processor(
        BatchLogRecordProcessor(
            EnhancedSQLiteLogExporter()
            if Config.EXPORTER_BACKEND == "sqlite"
            else EnhancedFileLogExporter()
        )
    )

    # Configure Python logging with a simple console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, Config.LOG_LEVEL.upper()))
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper()))
    root_logger.handlers.clear()  # Clear any existing handlers
    root_logger.addHandler(console_handler)

    # Add OpenTelemetry handler for structured logging
    otel_handler = LoggingHandler(
        level=getattr(logging, Config.LOG_LEVEL.upper()), logger_provider=log_provider
    )
    root_logger.addHandler(otel_handler)

    # Configure the instrumentation
    LoggingInstrumentor().instrument(
        set_logging_format=False
    )  # Don't let it override our format

    # Return both Python logger and OTel logger
    python_logger = logging.getLogger(__name__)
    otel_logger = get_logger(__name__)
    return python_logger, otel_logger


def setup_metrics(resource: Resource):
    """Configure application metrics"""
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[
            PeriodicExportingMetricReader(
                EnhancedSQLiteMetricExporter()
                if Config.EXPORTER_BACKEND == "sqlite"
                else EnhancedFileMetricExporter(),
                export_interval_millis=Config.METRICS_EXPORT_INTERVAL,
            ),
        ],
    )
    metrics.set_meter_provider(meter_provider)

    meter = metrics.get_meter(__name__)

    # Business metrics
    return {
        "request_counter": meter.create_counter(
            "http_requests_total", description="Total HTTP requests", unit="1"
        ),
        "request_duration": meter.create_histogram(
            "http_request_duration_seconds",
            description="HTTP request duration",
            unit="s",
        ),
        "active_connections": meter.create_up_down_counter(
            "http_active_connections", description="Active HTTP connections", unit="1"
        ),
        "error_counter": meter.create_counter(
            "http_errors_total", description="Total HTTP errors", unit="1"
        ),
    }


# ─── Initialize Observability ──────────────────────────────────────────────────
resource = setup_resource()
tracer = setup_tracing(resource)
logger, otel_logger = setup_logging(resource)
app_metrics = setup_metrics(resource)
