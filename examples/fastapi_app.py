"""
FastAPI demo that uses otel_file_exporter to write traces, logs and metrics
locally in JSON Lines format.

Run with:
    python examples\fastapi_app.py
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional
from pydantic import BaseModel, Field

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from opentelemetry import trace, metrics
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.semconv.attributes.http_attributes import HTTP_ROUTE

# Re-use helpers, data models and configured telemetry objects
from otel_file_exporter.otel import (
    Config,
    logger,
    otel_logger,
    app_metrics,
)

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


# ─── Example-specific Metrics ──────────────────────────────────────────────────
meter = metrics.get_meter(__name__)
app_metrics["item_operations"] = meter.create_counter(
    "item_operations_total",
    description="Total item operations",
    unit="1"
)

# ─── Middleware & lifespan ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
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
    # Gracefully shut down providers if the SDK exposes the hook
    _tp = trace.get_tracer_provider()
    if hasattr(_tp, "shutdown"):
        _tp.shutdown()  # type: ignore[attr-defined]

    _mp = metrics.get_meter_provider()
    if hasattr(_mp, "shutdown"):
        _mp.shutdown()  # type: ignore[attr-defined]


# ─── FastAPI Application ───────────────────────────────────────────────────────
app = FastAPI(
    title="OpenTelemetry Demo API",
    description="FastAPI application with comprehensive OpenTelemetry instrumentation",
    version=Config.SERVICE_VERSION,
    debug=Config.ENVIRONMENT == "development",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FastAPIInstrumentor.instrument_app(app)


@app.middleware("http")
async def track_requests(request: Request, call_next):
    start_time = time.time()
    app_metrics["active_connections"].add(1)

    response = None
    try:
        response = await call_next(request)
        span = trace.get_current_span()
        if span.is_recording():
            response.headers["X-Trace-ID"] = format(span.get_span_context().trace_id, "032x")
        return response
    finally:
        duration = time.time() - start_time
        route = request.scope.get("route").path if request.scope.get("route") else "unknown"
        status_code = response.status_code if response else 0
        labels = {"method": request.method, "endpoint": route, "status": status_code}
        app_metrics["request_counter"].add(1, labels)
        app_metrics["request_duration"].record(duration, labels)
        app_metrics["active_connections"].add(-1)


# ─── Exception Handlers ────────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x") if span.is_recording() else None
    app_metrics["error_counter"].add(1, {"status_code": str(exc.status_code), "endpoint": str(request.url.path)})
    logger.error(
        f"HTTP {exc.status_code}: {exc.detail}",
        extra={"event": "http_error", "status_code": exc.status_code, "detail": exc.detail,
               "path": str(request.url.path), "method": request.method}
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP_{exc.status_code}",
            message=exc.detail,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            trace_id=trace_id,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x") if span.is_recording() else None
    app_metrics["error_counter"].add(1, {"status_code": "500", "endpoint": str(request.url.path)})
    logger.error(
        f"Unexpected error: {exc}",
        extra={"event": "unexpected_error", "error_type": type(exc).__name__,
               "error_message": str(exc), "path": str(request.url.path), "method": request.method}
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            trace_id=trace_id,
        ).model_dump(),
    )


# ─── API Endpoints ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time(), "service": Config.SERVICE_NAME}


@app.get("/items/{item_id}", response_model=Item)
async def get_item(item_id: int, include_description: bool = False):
    if item_id < 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Item ID must be non-negative")
    if item_id > 10000:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")

    span = trace.get_current_span()
    span.set_attributes({
        HTTP_ROUTE: "/items/{item_id}",
        "item.id": item_id,
        "item.include_description": include_description,
        "business.operation": "get_item",
    })
    app_metrics["item_operations"].add(1, {"operation": "get", "item_type": "standard"})

    processing_time = 0.15 if include_description else 0.05
    time.sleep(processing_time)

    item = Item(
        item_id=item_id,
        name=f"Item-{item_id}",
        description=f"Detailed description for item {item_id}" if include_description else None,
        price=round(10.0 + (item_id * 0.5), 2),
        in_stock=item_id % 3 != 0,
    )
    logger.info(
        f"Retrieved item {item_id}",
        extra={"event": "item_retrieved", "item_id": item_id, "in_stock": item.in_stock,
               "processing_time_ms": processing_time * 1000, "include_description": include_description},
    )
    return item


@app.post("/items", response_model=Item, status_code=status.HTTP_201_CREATED)
async def create_item(item: Item):
    span = trace.get_current_span()
    span.set_attributes({"business.operation": "create_item", "item.name": item.name, "item.price": item.price})
    app_metrics["item_operations"].add(1, {"operation": "create", "item_type": "standard"})
    time.sleep(0.1)
    logger.info(
        f"Created item {item.item_id}",
        extra={"event": "item_created", "item_id": item.item_id, "item_name": item.name, "item_price": item.price},
    )
    return item


@app.get("/metrics/summary")
async def get_metrics_summary():
    return {
        "service": Config.SERVICE_NAME,
        "version": Config.SERVICE_VERSION,
        "environment": Config.ENVIRONMENT,
        "sample_rate": Config.TRACE_SAMPLE_RATE,
        "uptime_seconds": time.time(),
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level=Config.LOG_LEVEL.lower(),
        access_log=True,
    )
