import json
import logging
# ─── Configuration ──────────────────────────────────────────────────────────────
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from opentelemetry import trace, metrics
from opentelemetry._logs import set_logger_provider, get_logger
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler, LogData
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
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from opentelemetry.semconv.trace import SpanAttributes
from pydantic import BaseModel, Field


class Config:
    SERVICE_NAME = os.getenv("SERVICE_NAME", "fastapi-otel-demo")
    SERVICE_VERSION = os.getenv("SERVICE_VERSION", "1.0.0")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    TRACE_SAMPLE_RATE = float(os.getenv("TRACE_SAMPLE_RATE", "1.0"))
    METRICS_EXPORT_INTERVAL = int(os.getenv("METRICS_EXPORT_INTERVAL", "30000"))
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./telemetry"))


# ─── Pydantic Models ───────────────────────────────────────────────────────────
class Item(BaseModel):
    item_id: int = Field(..., description="Unique identifier for the item")
    name: str = Field(..., description="Name of the item")
    description: Optional[str] = Field(None, description="Item description")
    price: float = Field(..., ge=0, description="Item price")
    in_stock: bool = Field(True, description="Whether item is in stock")


class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: str
    trace_id: Optional[str] = None


# ─── Enhanced File Exporters ───────────────────────────────────────────────────
class ThreadSafeFileExporter:
    """Base class with common file handling patterns using pathlib"""

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
            json.dump(data, self._file, separators=(",", ":"), ensure_ascii=False, default=str)
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
                        "message": span.status.description
                    },
                    "attributes": dict(span.attributes) if span.attributes else {},
                    "events": [
                        {
                            "name": event.name,
                            "timestamp": event.timestamp,
                            "attributes": dict(event.attributes) if event.attributes else {}
                        }
                        for event in span.events
                    ] if span.events else [],
                    "resource": dict(span.resource.attributes) if span.resource else {}
                }
                self._write_json_line(span_data)
            return SpanExportResult.SUCCESS
        except Exception as e:
            logging.error(f"Failed to export spans: {e}")
            return SpanExportResult.FAILURE


class EnhancedFileLogExporter(ThreadSafeFileExporter, LogExporter):
    """Simple console log exporter that doesn't rely on to_json()"""

    def export(self, batch: Sequence[LogData]) -> LogExportResult:
        try:
            for log_data in batch:
                lr = log_data.log_record

                # Extract basic info safely
                timestamp = getattr(lr, 'timestamp', int(time.time() * 1_000_000_000))
                level = getattr(lr, 'severity_text', 'INFO')
                message = str(getattr(lr, 'body', ''))

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

    def __init__(self, filename: str = "logs.jsonl"):
        super().__init__(filename)

    def export(self, batch: Sequence[LogData]) -> LogExportResult:
        try:
            for log_data in batch:
                lr = log_data.log_record

                # Extract timestamp - different LogRecord implementations may vary
                timestamp = None
                if hasattr(lr, 'timestamp'):
                    timestamp = lr.timestamp
                elif hasattr(lr, 'observed_timestamp'):
                    timestamp = lr.observed_timestamp
                else:
                    # Fallback to current time in nanoseconds
                    timestamp = int(time.time() * 1_000_000_000)

                # Extract severity/level
                level = "INFO"
                if hasattr(lr, 'severity_text') and lr.severity_text:
                    level = lr.severity_text
                elif hasattr(lr, 'severity_number'):
                    # Map severity numbers to text
                    severity_map = {
                        1: "TRACE", 2: "TRACE2", 3: "TRACE3", 4: "TRACE4",
                        5: "DEBUG", 6: "DEBUG2", 7: "DEBUG3", 8: "DEBUG4",
                        9: "INFO", 10: "INFO2", 11: "INFO3", 12: "INFO4",
                        13: "WARN", 14: "WARN2", 15: "WARN3", 16: "WARN4",
                        17: "ERROR", 18: "ERROR2", 19: "ERROR3", 20: "ERROR4",
                        21: "FATAL", 22: "FATAL2", 23: "FATAL3", 24: "FATAL4"
                    }
                    level = severity_map.get(lr.severity_number, "INFO")

                log_entry = {
                    "timestamp": timestamp,
                    "level": level,
                    "message": str(lr.body) if hasattr(lr, 'body') else "",
                    "trace_id": format(lr.trace_id, "032x") if hasattr(lr,
                                                                       'trace_id') and lr.trace_id and lr.trace_id != 0 else None,
                    "span_id": format(lr.span_id, "016x") if hasattr(lr,
                                                                     'span_id') and lr.span_id and lr.span_id != 0 else None,
                    "attributes": dict(lr.attributes) if hasattr(lr,
                                                                 'attributes') and lr.attributes else {},
                    "resource": dict(lr.resource.attributes) if hasattr(lr,
                                                                        'resource') and lr.resource and lr.resource.attributes else {}
                }
                self._write_json_line(log_entry)
            return LogExportResult.SUCCESS
        except Exception as e:
            # Use Python's logging to avoid recursion
            import sys
            print(f"ERROR: Failed to export logs: {e}", file=sys.stderr)
            return LogExportResult.FAILURE

    def force_flush(self, timeout_millis: int = 30000) -> bool:
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
            metrics_data: Sequence,
            timeout_millis: int = 10000,
    ) -> MetricExportResult:
        try:
            # Handle different metrics data structures
            if hasattr(metrics_data, 'resource_metrics'):
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
                            "resource": dict(
                                resource_metric.resource.attributes) if resource_metric.resource and resource_metric.resource.attributes else {},
                            "scope": {
                                "name": scope_metric.scope.name if scope_metric.scope else "",
                                "version": scope_metric.scope.version if scope_metric.scope else ""
                            },
                            "data_points": [],
                            "timestamp": int(time.time() * 1000)  # Add timestamp for debugging
                        }

                        # Handle different data point types
                        data_points = getattr(metric.data, 'data_points', [])
                        for point in data_points:
                            point_data = {
                                "attributes": dict(point.attributes) if hasattr(point,
                                                                                'attributes') and point.attributes else {},
                                "start_time": getattr(point, 'start_time_unix_nano', 0),
                                "time": getattr(point, 'time_unix_nano', 0),
                            }

                            # Add value based on point type
                            if hasattr(point, 'value'):
                                point_data['value'] = point.value
                            if hasattr(point, 'sum'):
                                point_data['sum'] = point.sum
                            if hasattr(point, 'count'):
                                point_data['count'] = point.count
                            if hasattr(point, 'bucket_counts'):
                                point_data['bucket_counts'] = list(point.bucket_counts)
                            if hasattr(point, 'explicit_bounds'):
                                point_data['explicit_bounds'] = list(point.explicit_bounds)
                            if hasattr(point, 'min'):
                                point_data['min'] = point.min
                            if hasattr(point, 'max'):
                                point_data['max'] = point.max

                            metric_data["data_points"].append(point_data)

                        self._write_json_line(metric_data)
            return MetricExportResult.SUCCESS
        except Exception as e:
            # Use print to avoid recursion
            import sys
            print(f"ERROR: Failed to export metrics: {e}", file=sys.stderr)
            print(f"Metrics data type: {type(metrics_data)}", file=sys.stderr)
            if hasattr(metrics_data, '__dict__'):
                print(f"Metrics data attributes: {list(metrics_data.__dict__.keys())}",
                      file=sys.stderr)
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
                timestamp = getattr(lr, 'timestamp', int(time.time() * 1_000_000_000))
                level = getattr(lr, 'severity_text', 'INFO')
                message = str(getattr(lr, 'body', ''))

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


# ─── Observability Setup ───────────────────────────────────────────────────────
def setup_resource() -> Resource:
    """Create a resource with service information"""
    return Resource.create({
        SERVICE_NAME: Config.SERVICE_NAME,
        SERVICE_VERSION: Config.SERVICE_VERSION,
        "environment": Config.ENVIRONMENT,
        "host.name": os.getenv("HOSTNAME", Path.cwd().name)
    })


def setup_tracing(resource: Resource):
    """Configure distributed tracing"""
    # Set up trace propagation
    set_global_textmap(B3MultiFormat())

    # Configure tracer provider with sampling
    tracer_provider = TracerProvider(
        resource=resource,
        sampler=TraceIdRatioBased(Config.TRACE_SAMPLE_RATE)
    )
    trace.set_tracer_provider(tracer_provider)

    # Add only our custom exporters to avoid compatibility issues
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            EnhancedFileSpanExporter(),
            max_queue_size=2048,
            max_export_batch_size=512,
            export_timeout_millis=30000
        )
    )

    return trace.get_tracer(__name__)


def setup_logging(resource: Resource):
    """Configure structured logging"""
    log_provider = LoggerProvider(resource=resource)
    set_logger_provider(log_provider)

    # Add only our custom processors - avoid problematic ConsoleLogExporter
    log_provider.add_log_record_processor(
        BatchLogRecordProcessor(EnhancedFileLogExporter())
    )

    # Configure Python logging with a simple console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, Config.LOG_LEVEL.upper()))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper()))
    root_logger.handlers.clear()  # Clear any existing handlers
    root_logger.addHandler(console_handler)

    # Add OpenTelemetry handler for structured logging
    otel_handler = LoggingHandler(
        level=getattr(logging, Config.LOG_LEVEL.upper()),
        logger_provider=log_provider
    )
    root_logger.addHandler(otel_handler)

    # Configure the instrumentation
    LoggingInstrumentor().instrument(set_logging_format=False)  # Don't let it override our format

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
                EnhancedFileMetricExporter(),
                export_interval_millis=Config.METRICS_EXPORT_INTERVAL
            ),
        ]
    )
    metrics.set_meter_provider(meter_provider)

    meter = metrics.get_meter(__name__)

    # Business metrics
    return {
        "request_counter": meter.create_counter(
            "http_requests_total",
            description="Total HTTP requests",
            unit="1"
        ),
        "request_duration": meter.create_histogram(
            "http_request_duration_seconds",
            description="HTTP request duration",
            unit="s"
        ),
        "active_connections": meter.create_up_down_counter(
            "http_active_connections",
            description="Active HTTP connections",
            unit="1"
        ),
        "error_counter": meter.create_counter(
            "http_errors_total",
            description="Total HTTP errors",
            unit="1"
        ),
        "item_operations": meter.create_counter(
            "item_operations_total",
            description="Total item operations",
            unit="1"
        )
    }


# ─── Initialize Observability ──────────────────────────────────────────────────
resource = setup_resource()
tracer = setup_tracing(resource)
logger, otel_logger = setup_logging(resource)
app_metrics = setup_metrics(resource)


# ─── Middleware ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[valid-type]
    """Handle application startup and shutdown"""
    # Startup
    logger.info("Application starting up")
    otel_logger.emit(logging.LogRecord(
        name=__name__,
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Application starting up",
        args=(),
        exc_info=None
    ))
    yield
    # Shutdown
    logger.info("Application shutting down")
    otel_logger.emit(logging.LogRecord(
        name=__name__,
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Application shutting down",
        args=(),
        exc_info=None
    ))
    trace.get_tracer_provider().shutdown()
    metrics.get_meter_provider().shutdown()


# ─── FastAPI Application ───────────────────────────────────────────────────────
app = FastAPI(
    title="OpenTelemetry Demo API",
    description="FastAPI application with comprehensive OpenTelemetry instrumentation",
    version=Config.SERVICE_VERSION,
    debug=Config.ENVIRONMENT == "development",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)


@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Custom middleware for additional request tracking"""
    start_time = time.time()

    # Track active connections
    app_metrics["active_connections"].add(1)

    response = None
    try:
        response = await call_next(request)

        # Add trace context to response headers
        span = trace.get_current_span()
        if span.is_recording():
            trace_id = format(span.context.trace_id, "032x")
            response.headers["X-Trace-ID"] = trace_id

        return response
    except Exception as e:
        # Handle any exceptions that occur during request processing
        logger.error(f"Error in request processing: {str(e)}", exc_info=True)
        raise
    finally:
        # Record metrics
        duration = time.time() - start_time

        # Extract route pattern safely
        route = "unknown"
        if hasattr(request, "scope") and "route" in request.scope:
            route = request.scope["route"].path if request.scope["route"] else "unknown"

        status_code = 0
        if response is not None:
            status_code = getattr(response, "status_code", 0)

        labels = {
            "method": request.method,
            "endpoint": route,
            "status": status_code
        }

        app_metrics["request_counter"].add(1, labels)
        app_metrics["request_duration"].record(duration, labels)
        app_metrics["active_connections"].add(-1)


# ─── Exception Handlers ────────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper observability"""
    span = trace.get_current_span()
    trace_id = format(span.context.trace_id, "032x") if span.is_recording() else None

    # Record error metrics
    app_metrics["error_counter"].add(1, {
        "status_code": str(exc.status_code),
        "endpoint": str(request.url.path)
    })

    # Log error
    logger.error(
        f"HTTP {exc.status_code}: {exc.detail}",
        extra={
            "event": "http_error",
            "status_code": exc.status_code,
            "detail": exc.detail,
            "path": str(request.url.path),
            "method": request.method
        }
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP_{exc.status_code}",
            message=exc.detail,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            trace_id=trace_id
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    span = trace.get_current_span()
    trace_id = format(span.context.trace_id, "032x") if span.is_recording() else None

    # Record error
    app_metrics["error_counter"].add(1, {
        "status_code": "500",
        "endpoint": str(request.url.path)
    })

    logger.error(
        f"Unexpected error: {str(exc)}",
        extra={
            "event": "unexpected_error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "path": str(request.url.path),
            "method": request.method
        }
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            trace_id=trace_id
        ).dict()
    )


# ─── API Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time(), "service": Config.SERVICE_NAME}


@app.get("/items/{item_id}", response_model=Item)
async def get_item(item_id: int, include_description: bool = False):
    """Get item by ID with enhanced observability"""

    # Input validation
    if item_id < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Item ID must be non-negative"
        )

    if item_id > 10000:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item not found"
        )

    # Add custom span attributes
    span = trace.get_current_span()
    span.set_attributes({
        SpanAttributes.HTTP_ROUTE: "/items/{item_id}",
        "item.id": item_id,
        "item.include_description": include_description,
        "business.operation": "get_item"
    })

    # Record business metrics
    app_metrics["item_operations"].add(1, {"operation": "get", "item_type": "standard"})

    try:
        with tracer.start_as_current_span(
                "item_processing",
                attributes={
                    "item.id": item_id,
                    "processing.type": "detailed" if include_description else "basic"
                }
        ) as processing_span:

            # Simulate some processing time
            processing_time = 0.05 if not include_description else 0.15
            time.sleep(processing_time)

            # Add span events
            processing_span.add_event(
                "item_data_retrieved",
                {"item.id": item_id, "processing_time_ms": processing_time * 1000}
            )

            # Create item data
            item_data = Item(
                item_id=item_id,
                name=f"Item-{item_id}",
                description=f"Detailed description for item {item_id}" if include_description else None,
                price=round(10.0 + (item_id * 0.5), 2),
                in_stock=item_id % 3 != 0  # Simulate some out-of-stock items
            )

            # Log successful operation
            logger.info(
                f"Retrieved item {item_id}",
                extra={
                    "event": "item_retrieved",
                    "item_id": item_id,
                    "in_stock": item_data.in_stock,
                    "processing_time_ms": processing_time * 1000,
                    "include_description": include_description
                }
            )

            return item_data

    except Exception as e:
        span.record_exception(e)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
        raise


@app.post("/items", response_model=Item, status_code=status.HTTP_201_CREATED)
async def create_item(item: Item):
    """Create a new item"""
    span = trace.get_current_span()
    span.set_attributes({
        "business.operation": "create_item",
        "item.name": item.name,
        "item.price": item.price
    })

    app_metrics["item_operations"].add(1, {"operation": "create", "item_type": "standard"})

    with tracer.start_as_current_span("item_validation_and_creation") as create_span:
        # Simulate validation and creation
        time.sleep(0.1)

        create_span.add_event("item_validated", {"item.name": item.name})
        create_span.add_event("item_created", {"item.id": item.item_id})

        logger.info(
            f"Created item {item.item_id}",
            extra={
                "event": "item_created",
                "item_id": item.item_id,
                "item_name": item.name,
                "item_price": item.price
            }
        )

        return item


@app.get("/metrics/summary")
async def get_metrics_summary():
    """Get basic metrics summary"""
    return {
        "service": Config.SERVICE_NAME,
        "version": Config.SERVICE_VERSION,
        "environment": Config.ENVIRONMENT,
        "sample_rate": Config.TRACE_SAMPLE_RATE,
        "uptime_seconds": time.time()
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level=Config.LOG_LEVEL.lower(),
        access_log=True
    )
