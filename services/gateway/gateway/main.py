"""FastAPI application for the public gateway."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import start_http_server
from sqlalchemy import text
from sqlalchemy.engine import Engine
from starlette.middleware.trustedhost import TrustedHostMiddleware

from gateway import __version__
from gateway.circuit_breaker import CircuitBreaker, CircuitState
from gateway.config import Settings, get_settings
from gateway.db import make_engine, make_session_factory
from gateway.model_client import ModelClient
from gateway.observability import (
    BREAKER_STATE,
    MaxBodySizeMiddleware,
    RequestContextMiddleware,
    configure_logging,
)
from gateway.ratelimit import RateLimiter
from gateway.routers import admin, auth, predictions

log = logging.getLogger("gateway")


def build_model_client(settings: Settings, **overrides) -> ModelClient:
    breaker = CircuitBreaker(
        failure_threshold=settings.breaker_failure_threshold,
        recovery_seconds=settings.breaker_recovery_seconds,
        on_state_change=lambda s: BREAKER_STATE.set(0 if s is CircuitState.CLOSED else 1),
    )
    return ModelClient(
        settings.model_service_url,
        settings.internal_api_token.get_secret_value(),
        settings.model_admin_token.get_secret_value(),
        breaker=breaker,
        timeout=settings.model_timeout_seconds,
        connect_timeout=settings.model_connect_timeout_seconds,
        max_retries=settings.model_max_retries,
        backoff=settings.model_backoff_seconds,
        **overrides,
    )


def create_app(
    settings: Settings | None = None,
    *,
    engine: Engine | None = None,
    model_client: ModelClient | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    engine = engine or make_engine(settings.database_url.get_secret_value())
    model_client = model_client or build_model_client(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if settings.metrics_port:
            start_http_server(settings.metrics_port)
        yield
        model_client.close()
        engine.dispose()

    app = FastAPI(
        title="Machine Risk - Gateway API",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
        debug=False,
    )
    app.state.settings = settings
    app.state.session_factory = make_session_factory(engine)
    app.state.model_client = model_client
    app.state.rate_limiter = RateLimiter(settings.rate_limit_storage_uri.get_secret_value())

    # Starlette runs the last-added middleware first.
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=settings.max_body_bytes)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "PATCH", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID", "Retry-After"],
            max_age=600,
        )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    app.add_middleware(RequestContextMiddleware, hsts=settings.environment == "production")

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:  # no stack traces to clients
        log.exception("unhandled error", exc_info=exc)
        return JSONResponse({"detail": "internal server error"}, status_code=500)

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    def ready() -> JSONResponse:
        checks = {"database": "ok", "model_service": model_client.breaker.state.value}
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            checks["database"] = "unavailable"
        healthy = checks["database"] == "ok" and checks["model_service"] != CircuitState.OPEN.value
        return JSONResponse(
            {"status": "ready" if healthy else "degraded", **checks}, status_code=200 if healthy else 503
        )

    app.include_router(auth.router)
    app.include_router(predictions.router)
    app.include_router(admin.router)
    return app


def app_factory() -> FastAPI:  # uvicorn --factory gateway.main:app_factory
    return create_app()
