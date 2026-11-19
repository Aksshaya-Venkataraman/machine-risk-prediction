from __future__ import annotations

import httpx2
import pytest
from gateway.circuit_breaker import CircuitBreaker, CircuitState
from gateway.model_client import ModelClient, ModelRejectedInput, ModelServiceUnavailable
from gw_support import register_and_login

GOOD = {
    "risk_level": "Low",
    "probabilities": {"Low": 0.9, "Medium": 0.08, "High": 0.02},
    "model_version": "rf-test",
    "ignored_features": [],
}


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def scripted(*responses):
    """Transport that replays responses (or raises exceptions) in order."""
    calls: list[httpx2.Request] = []
    queue = list(responses)

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        item = queue.pop(0) if queue else queue_last[0]
        if isinstance(item, Exception):
            raise item
        return item

    queue_last = [responses[-1]]
    return httpx2.MockTransport(handler), calls


def make_client(transport, threshold=3, retries=2, clock=None):
    breaker = CircuitBreaker(failure_threshold=threshold, recovery_seconds=10, clock=clock or FakeClock())
    return ModelClient(
        "http://model",
        "t" * 40,
        "a" * 40,
        breaker=breaker,
        max_retries=retries,
        backoff=0.0,
        transport=transport,
        sleep=lambda _: None,
    )


def test_breaker_opens_after_threshold_and_recovers():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=5, clock=clock)
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert not breaker.allow_request()
    clock.now = 5.0
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.allow_request()  # one trial call allowed
    assert not breaker.allow_request()  # concurrent calls still blocked
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED


def test_failed_trial_reopens_circuit():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=5, clock=clock)
    breaker.record_failure()
    clock.now = 6.0
    assert breaker.allow_request()
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert breaker.retry_after == 5


def test_retries_transient_errors_then_succeeds():
    transport, calls = scripted(httpx2.ConnectError("refused"), httpx2.Response(503), httpx2.Response(200, json=GOOD))
    client = make_client(transport)
    assert client.predict({"rpm": 1})["risk_level"] == "Low"
    assert len(calls) == 3
    assert client.breaker.state is CircuitState.CLOSED


def test_timeouts_are_retried_then_reported():
    transport, calls = scripted(httpx2.ReadTimeout("slow"))
    client = make_client(transport, retries=2)
    with pytest.raises(ModelServiceUnavailable, match="ReadTimeout"):
        client.predict({"rpm": 1})
    assert len(calls) == 3


def test_client_errors_are_not_retried():
    transport, calls = scripted(httpx2.Response(422, json={"detail": {"errors": ["rpm: bad"]}}))
    client = make_client(transport)
    with pytest.raises(ModelRejectedInput):
        client.predict({"rpm": -1})
    assert len(calls) == 1


def test_open_circuit_fails_fast_without_network_calls():
    transport, calls = scripted(httpx2.Response(500))
    client = make_client(transport, threshold=2, retries=0)
    for _ in range(2):
        with pytest.raises(ModelServiceUnavailable):
            client.predict({"rpm": 1})
    assert len(calls) == 2
    with pytest.raises(ModelServiceUnavailable, match="circuit is open") as info:
        client.predict({"rpm": 1})
    assert info.value.retry_after == 10
    assert len(calls) == 2


def test_token_and_request_id_sent_to_model_service():
    transport, calls = scripted(httpx2.Response(200, json=GOOD))
    make_client(transport).predict({"rpm": 1})
    assert calls[0].headers["authorization"] == "Bearer " + "t" * 40
    assert "x-request-id" in calls[0].headers


def test_gateway_returns_503_with_retry_after_when_model_down(make_gateway, sample_features):
    transport, _ = scripted(httpx2.ConnectError("down"))
    client = make_gateway(transport, breaker_failure_threshold=1, model_max_retries=1)
    headers = register_and_login(client, "down@example.com")
    body = {"machine_id": "m-1", "features": sample_features}
    first = client.post("/api/v1/predictions", json=body, headers=headers)
    assert first.status_code == 503
    second = client.post("/api/v1/predictions", json=body, headers=headers)
    assert second.status_code == 503
    assert "retry-after" in second.headers
    assert client.get("/ready").status_code == 503
