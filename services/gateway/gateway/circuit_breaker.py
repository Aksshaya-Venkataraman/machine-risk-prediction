"""A small, thread-safe circuit breaker.

CLOSED     calls flow; consecutive failures are counted.
OPEN       calls fail fast until ``recovery_seconds`` have passed.
HALF_OPEN  exactly one trial call is let through; success closes the
           circuit, failure opens it again.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from enum import StrEnum


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        on_state_change: Callable[[CircuitState], None] | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self._threshold = failure_threshold
        self._recovery = recovery_seconds
        self._clock = clock
        self._on_change = on_state_change
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._trial_in_flight = False

    def _effective_state(self) -> CircuitState:
        if self._state is CircuitState.OPEN and self._clock() - self._opened_at >= self._recovery:
            return CircuitState.HALF_OPEN
        return self._state

    def _set(self, state: CircuitState) -> None:
        if state is not self._state:
            self._state = state
            if self._on_change:
                self._on_change(state)

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._effective_state()

    @property
    def retry_after(self) -> int:
        with self._lock:
            remaining = self._recovery - (self._clock() - self._opened_at)
            return max(1, int(remaining + 0.999))

    def allow_request(self) -> bool:
        with self._lock:
            state = self._effective_state()
            if state is CircuitState.CLOSED:
                return True
            if state is CircuitState.OPEN or self._trial_in_flight:
                return False
            self._set(CircuitState.HALF_OPEN)
            self._trial_in_flight = True
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._trial_in_flight = False
            self._set(CircuitState.CLOSED)

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            was_trial = self._trial_in_flight
            self._trial_in_flight = False
            if was_trial or self._failures >= self._threshold:
                self._opened_at = self._clock()
                self._set(CircuitState.OPEN)
