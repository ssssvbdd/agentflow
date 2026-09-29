from __future__ import annotations

from fastapi import FastAPI, Response
from loguru import logger

from agentflow.settings import app_settings


def setup_observability(app: FastAPI) -> None:
    cfg = app_settings.observability or {}
    _setup_tracing(app, cfg)
    _setup_prometheus(app, cfg)


def _setup_tracing(app: FastAPI, cfg: dict) -> None:
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": cfg.get("service_name", "agentflow")})
        provider = TracerProvider(resource=resource)
        endpoint = cfg.get("otlp_endpoint")
        if endpoint:
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
        logger.info("OpenTelemetry FastAPI instrumentation enabled")
    except Exception as err:
        logger.warning(f"OpenTelemetry setup skipped: {err}")


def _setup_prometheus(app: FastAPI, cfg: dict) -> None:
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
        from starlette.middleware.base import BaseHTTPMiddleware
        import time

        request_counter = Counter(
            "agentflow_http_requests_total",
            "HTTP requests",
            ["method", "path", "status"],
        )
        latency_hist = Histogram(
            "agentflow_http_request_duration_seconds",
            "HTTP request latency",
            ["method", "path"],
        )
        active_requests = Gauge("agentflow_http_active_requests", "Active HTTP requests")

        class PrometheusMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                active_requests.inc()
                started = time.perf_counter()
                status = "500"
                try:
                    response = await call_next(request)
                    status = str(response.status_code)
                    return response
                finally:
                    path = request.scope.get("route").path if request.scope.get("route") else request.url.path
                    request_counter.labels(request.method, path, status).inc()
                    latency_hist.labels(request.method, path).observe(time.perf_counter() - started)
                    active_requests.dec()

        if cfg.get("prometheus_enabled", True):
            app.add_middleware(PrometheusMiddleware)

            @app.get(cfg.get("metrics_path", "/metrics"), include_in_schema=False)
            async def metrics():
                return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

            logger.info("Prometheus /metrics endpoint enabled")
    except Exception as err:
        logger.warning(f"Prometheus setup skipped: {err}")
