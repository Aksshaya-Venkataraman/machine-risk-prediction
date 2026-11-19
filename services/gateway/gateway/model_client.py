"""HTTP client for the model service: timeouts, retries with backoff, circuit breaker."""

from __future__ import annotations

import logging
import secrets
import time
from collections.abc import Callable
from typing import Any

import httpx2

from gateway.circuit_breaker import CircuitBreaker
from gateway.observability import MODEL_CALLS, MODEL_RETRIES, request_id_ctx

log = logging.getLogger(__name__)
_rng = secrets.SystemRandom()


class ModelServiceError(Exception):
    """The model service could not produce an answer."""


class ModelServiceUnavailable(ModelServiceError):
    def __init__(self, message: str, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ModelRejectedInput(ModelServiceError):
    def __init__(self, detail: Any) -> None:
        super().__init__("model service rejected the input")
        self.detail = detail


class ModelClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        admin_token: str,
        *,
        breaker: CircuitBreaker,
        timeout: float = 2.0,
        connect_timeout: float = 0.5,
        max_retries: int = 2,
        backoff: float = 0.2,
        transport: httpx2.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.breaker = breaker
        self._token = token
        self._admin_token = admin_token
        self._max_retries = max_retries
        self._backoff = backoff
        self._sleep = sleep
        self._client = httpx2.Client(
            base_url=base_url,
            timeout=httpx2.Timeout(timeout, connect=connect_timeout),
            transport=transport,
            follow_redirects=False,
        )

    def close(self) -> None:
        self._client.close()

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        return self._call("POST", "/v1/predict", json={"features": features})

    def model_info(self) -> dict[str, Any]:
        return self._call("GET", "/v1/model")

    def reload(self) -> dict[str, Any]:
        return self._call("POST", "/v1/admin/reload", admin=True, retries=0)

    def _delay(self, attempt: int) -> float:
        jitter = _rng.uniform(0, self._backoff)
        return self._backoff * (2**attempt) + jitter

    def _call(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        admin: bool = False,
        retries: int | None = None,
    ) -> dict[str, Any]:
        if not self.breaker.allow_request():
            MODEL_CALLS.labels("circuit_open").inc()
            raise ModelServiceUnavailable("model service circuit is open", self.breaker.retry_after)

        headers = {
            "Authorization": f"Bearer {self._admin_token if admin else self._token}",
            "X-Request-ID": request_id_ctx.get(),
        }
        max_retries = self._max_retries if retries is None else retries
        last_error = "unknown error"

        for attempt in range(max_retries + 1):
            if attempt:
                MODEL_RETRIES.inc()
                self._sleep(self._delay(attempt - 1))
            try:
                response = self._client.request(method, path, json=json, headers=headers)
            except httpx2.TransportError as exc:  # includes timeouts and refused connections
                last_error = type(exc).__name__
                continue

            if response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
                continue

            # Any non-5xx answer proves the service is up.
            self.breaker.record_success()
            if response.status_code == 422:
                MODEL_CALLS.labels("rejected").inc()
                raise ModelRejectedInput(_safe_json(response).get("detail"))
            if response.status_code == 409:
                MODEL_CALLS.labels("conflict").inc()
                raise ModelRejectedInput(_safe_json(response).get("detail"))
            if response.status_code >= 400:
                MODEL_CALLS.labels("client_error").inc()
                log.error("model service returned %s for %s", response.status_code, path)
                raise ModelServiceError(f"model service returned HTTP {response.status_code}")
            MODEL_CALLS.labels("ok").inc()
            return _safe_json(response)

        self.breaker.record_failure()
        MODEL_CALLS.labels("failed").inc()
        log.warning("model service call failed after %d attempts: %s", max_retries + 1, last_error)
        raise ModelServiceUnavailable(f"model service unavailable ({last_error})")


def _safe_json(response: httpx2.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise ModelServiceError("model service returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise ModelServiceError("model service returned unexpected JSON")
    return data
