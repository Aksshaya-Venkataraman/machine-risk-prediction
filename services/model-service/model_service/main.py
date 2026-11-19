"""FastAPI app for the model service. Only the gateway talks to it."""

from __future__ import annotations

import hmac
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse
from prometheus_client import start_http_server
from pydantic import SecretStr
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ml.contract import validate_features
from model_service import __version__
from model_service.config import Settings, get_settings
from model_service.model_store import IncompatibleModelError, LoadedModel, ModelLoadError, ModelStore
from model_service.observability import (
    INFERENCE_LATENCY,
    MODEL_INFO,
    PREDICTIONS,
    MaxBodySizeMiddleware,
    RequestContextMiddleware,
    configure_logging,
)
from model_service.schemas import ModelInfo, PredictRequest, PredictResponse, ReloadResponse

log = logging.getLogger("model_service")


def _bearer(expected: SecretStr) -> Callable[[str | None], None]:
    def check(authorization: str | None = Header(default=None)) -> None:
        scheme, _, supplied = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            supplied.encode(), expected.get_secret_value().encode()
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return check


def create_app(settings: Settings | None = None, store: ModelStore | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    store = store or ModelStore(settings.model_path, settings.manifest_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if store.current is None:
            model = store.load()  # fail fast: never serve without a verified model
            MODEL_INFO.labels(model.version).set(1)
        if settings.metrics_port:
            start_http_server(settings.metrics_port)
        yield

    app = FastAPI(
        title="Machine Risk - Model Service",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
        debug=False,
    )
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    app.add_middleware(RequestContextMiddleware, hsts=settings.environment == "production")

    internal = _bearer(settings.internal_api_token)
    admin = _bearer(settings.model_admin_token)

    def current_model() -> LoadedModel:
        model = store.current
        if model is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")
        return model

    @app.exception_handler(Exception)
    async def unhandled(_, exc: Exception) -> JSONResponse:  # never leak stack traces
        log.exception("unhandled error", exc_info=exc)
        return JSONResponse({"detail": "internal server error"}, status_code=500)

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    def ready() -> JSONResponse:
        model = store.current
        if model is None:
            return JSONResponse({"status": "not ready"}, status_code=503)
        return JSONResponse({"status": "ready", "model_version": model.version})

    @app.post("/v1/predict", response_model=PredictResponse, dependencies=[Depends(internal)])
    def predict(body: PredictRequest, model: LoadedModel = Depends(current_model)) -> PredictResponse:
        result = validate_features(model.contract, body.features)
        if not result.ok:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={"message": "features do not match the model contract", "errors": result.errors},
            )
        started = time.perf_counter()
        risk, probabilities = model.predict(result.clean)
        INFERENCE_LATENCY.observe(time.perf_counter() - started)
        PREDICTIONS.labels(risk).inc()
        return PredictResponse(
            risk_level=risk,
            probabilities=probabilities,
            model_version=model.version,
            ignored_features=result.ignored,
        )

    @app.get("/v1/model", response_model=ModelInfo, dependencies=[Depends(internal)])
    def model_info(model: LoadedModel = Depends(current_model)) -> ModelInfo:
        return ModelInfo(
            version=model.version,
            sha256=model.sha256,
            trained_at=model.trained_at,
            loaded_at=model.loaded_at,
            classes=model.contract["classes"],
            features=model.contract["features"],
            metrics=model.metrics,
        )

    @app.post("/v1/admin/reload", response_model=ReloadResponse, dependencies=[Depends(admin)])
    def reload_model() -> ReloadResponse:
        previous = store.current
        try:
            model, report = store.reload()
        except IncompatibleModelError as exc:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail={"message": "new model is not compatible", "breaking": exc.report.breaking},
            ) from exc
        except ModelLoadError as exc:
            log.error("reload failed: %s", exc)
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "model artifacts failed verification") from exc
        if previous is not None and previous.version != model.version:
            MODEL_INFO.labels(previous.version).set(0)
        MODEL_INFO.labels(model.version).set(1)
        log.info("model reloaded to %s", model.version)
        return ReloadResponse(version=model.version, warnings=report.warnings)

    return app


def app_factory() -> FastAPI:  # used by uvicorn --factory
    return create_app()
