# otel_file_exporter

A lightweight OpenTelemetry file exporter that writes traces, logs and metrics to local JSON Lines files – **no external back-ends required**.  
All traces, logs and metrics are written locally as _newline-delimited JSON_ (`*.jsonl`) so they can be inspected with
your favorite tools or fed into pipelines later.

```
telemetry/
├─ traces.jsonl   <- Spans
├─ logs.jsonl     <- Structured application logs
└─ metrics.jsonl  <- OTLP metrics exports
```

## Why?

* Zero-dependency observability for demos, local development and CI
* Uses the official OpenTelemetry SDK – switch to OTLP / Jaeger / etc. at any time
* Demonstrates **structured logging** and **custom OpenTelemetry exporters**

## Requirements

* Python **3.13** or newer (matches `pyproject.toml`)
* Windows, macOS or Linux – no native deps

## Quick start

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
python examples\fastapi_app.py
```

Then open `http://localhost:8000/docs` for the interactive Swagger UI.

## Example requests

```bash
curl http://localhost:8000/health
curl http://localhost:8000/items/123?include_description=true
```

Watch the `telemetry` directory fill with JSONL lines while you interact with the API.

## Environment variables

| Var                     | Default            | Purpose                                    |
|-------------------------|--------------------|--------------------------------------------|
| `SERVICE_NAME`          | `fastapi-otel-demo`| Resource attribute in traces/logs/metrics  |
| `SERVICE_VERSION`       | `1.0.0`            | Resource attribute                         |
| `ENVIRONMENT`           | `development`      | Resource attribute                         |
| `LOG_LEVEL`             | `INFO`             | Console & JSONL log level                  |
| `TRACE_SAMPLE_RATE`     | `1.0`              | 0.0-1.0 probability sampler                |
| `METRICS_EXPORT_INTERVAL`| `5000`            | Export interval in **ms**                  |
| `OUTPUT_DIR`            | `./telemetry`      | Where JSONL files are written              |
| `PORT`                  | `8000`             | HTTP port for the demo app                 |

Set them in the shell, a `.env` file or your container orchestrator.

## Library usage

Import the pre-configured helpers in your own code:

```python
from otel_file_exporter import tracer, logger, app_metrics

with tracer.start_as_current_span("my_work"):
    logger.info("Hello with trace context!")
```

## Project layout

```
examples/
└─ fastapi_app.py   # Self-contained demo API
src/otel_file_exporter/
    ├─ __init__.py
    └─ otel.py      # Custom exporters & bootstrap helpers
pyproject.toml
README.md
```

## Development

```bash
pip install -e ".[dev]"   # add your own dev dependencies extras
pytest -q                 # add tests as you go
```

Feel free to open issues or PRs – contributions are welcome!

## Publishing to PyPI

All releases are handled by `.github/workflows/publish.yml`.  
The workflow builds the wheel + sdist with **uv** and uploads them to PyPI
using `pypa/gh-action-pypi-publish@release/v1`.

Publishing is triggered automatically when you push a git tag that matches
`v*.*.*` or when you run the workflow manually from the **Actions** tab.

1. Create a scoped PyPI **API token** with *publish* privileges.  
2. Add the token to the repository secrets as `PYPI_API_TOKEN`.  
3. Tag and push a new version:

```cmd
git tag v0.2.0
git push --tags
```

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.
