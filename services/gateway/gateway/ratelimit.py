"""Rate limiting on top of the `limits` library.

Use ``memory://`` for a single container and ``redis://`` when running more
than one gateway replica so every replica shares the same counters.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from fastapi import Depends, HTTPException, Request, status
from limits import RateLimitItem, parse
from limits.storage import storage_from_string
from limits.strategies import MovingWindowRateLimiter

from gateway.db import User
from gateway.security import get_current_user


class RateLimiter:
    def __init__(self, storage_uri: str) -> None:
        self._limiter = MovingWindowRateLimiter(storage_from_string(storage_uri))

    def hit(self, item: RateLimitItem, scope: str, key: str) -> None:
        if not self._limiter.hit(item, scope, key):
            reset_at, _ = self._limiter.get_window_stats(item, scope, key)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="rate limit exceeded",
                headers={"Retry-After": str(max(1, int(reset_at - time.time())))},
            )

    def reset(self) -> None:
        self._limiter.storage.reset()


def _client_ip(request: Request) -> str:
    # Uvicorn rewrites request.client from X-Forwarded-For only for proxies
    # listed in FORWARDED_ALLOW_IPS, so this cannot be spoofed by clients.
    return request.client.host if request.client else "unknown"


def limit_by_ip(scope: str, setting: str) -> Callable[[Request], None]:
    def dependency(request: Request) -> None:
        limit = parse(getattr(request.app.state.settings, setting))
        request.app.state.rate_limiter.hit(limit, scope, _client_ip(request))

    return dependency


def limit_by_user(scope: str, setting: str) -> Callable[..., None]:
    def dependency(request: Request, user: User = Depends(get_current_user)) -> None:
        limit = parse(getattr(request.app.state.settings, setting))
        request.app.state.rate_limiter.hit(limit, scope, f"user:{user.id}")

    return dependency
