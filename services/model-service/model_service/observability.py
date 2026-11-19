"""Request-ID tracing, security headers, body-size limits, metrics and JSON logs."""

from __future__ import annotations

import json
import logging
import re
import sys
import time
import uuid
from contextvars import ContextVar

from fastapi import HTTPException
from prometheus_client import Counter, Gauge, Histogram
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{8,128}")

HTTP_REQUESTS = Counter("model_service_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram("model_service_http_request_duration_seconds", "HTTP request latency", ["method", "route"])
PREDICTIONS = Counter("model_service_predictions_total", "Predictions served", ["risk_level"])
INFERENCE_LATENCY = Histogram("model_service_inference_seconds", "Model inference time")
MODEL_INFO = Gauge("model_service_model_loaded", "1 when a model is loaded", ["version"])

_API_HEADERS = (
    ("x-content-type-options", "nosniff"),
    ("x-frame-options", "DENY"),
    ("referrer-policy", "no-referrer"),
    ("permissions-policy", "camera=(), microphone=(), geolocation=()"),
    ("cross-origin-opener-policy", "same-origin"),
    ("cache-control", "no-store"),
)
_DOCS_PATHS = ("/docs", "/openapi.json")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_ctx.get(),
        }
        for key in ("method", "path", "status", "duration_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


access_log = logging.getLogger("model_service.access")


class RequestContextMiddleware:
    """Assigns/propagates X-Request-ID, adds security headers, records metrics."""

    def __init__(self, app: ASGIApp, *, hsts: bool = False) -> None:
        self.app = app
        self.hsts = hsts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get("x-request-id", "")
        request_id = incoming if _VALID_REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
        token = request_id_ctx.set(request_id)
        scope.setdefault("state", {})["request_id"] = request_id
        status = 500
        started = time.perf_counter()
        is_docs = scope["path"].startswith(_DOCS_PATHS)

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = MutableHeaders(scope=message)
                headers["x-request-id"] = request_id
                for name, value in _API_HEADERS:
                    headers[name] = value
                if not is_docs:
                    headers["content-security-policy"] = "default-src 'none'; frame-ancestors 'none'"
                if self.hsts:
                    headers["strict-transport-security"] = "max-age=63072000; includeSubDomains"
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - started
            route = scope.get("route")
            route_name = getattr(route, "path", "unmatched")
            HTTP_REQUESTS.labels(scope["method"], route_name, str(status)).inc()
            HTTP_LATENCY.labels(scope["method"], route_name).observe(elapsed)
            access_log.info(
                "request",
                extra={
                    "method": scope["method"],
                    "path": route_name,
                    "status": status,
                    "duration_ms": round(elapsed * 1000, 2),
                },
            )
            request_id_ctx.reset(token)


class MaxBodySizeMiddleware:
    """Reject request bodies larger than ``max_bytes`` before they are parsed."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > self.max_bytes):
            await _reject(send)
            return

        received = 0

        async def limited_receive() -> Message:
            # Covers chunked uploads that send no Content-Length header.
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise HTTPException(status_code=413, detail="request body too large")
            return message

        await self.app(scope, limited_receive, send)


async def _reject(send: Send) -> None:
    body = b'{"detail":"request body too large"}'
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body})
