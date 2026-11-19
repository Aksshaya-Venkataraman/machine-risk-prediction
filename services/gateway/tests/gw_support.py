"""Helpers shared by the gateway tests."""

from __future__ import annotations

import os
import secrets

from fastapi.testclient import TestClient
from gateway.config import Settings

PASSWORD = "correct-horse-battery-staple"


def gateway_settings(**overrides) -> Settings:
    values = {
        "environment": "test",
        "database_url": "sqlite://",
        "jwt_secret_key": secrets.token_urlsafe(48),
        "internal_api_token": os.environ["INTERNAL_API_TOKEN"],
        "model_admin_token": os.environ["MODEL_ADMIN_TOKEN"],
        "allowed_hosts": ["testserver"],
        "cors_origins": ["https://app.example.com"],
        "metrics_port": 0,
        "model_backoff_seconds": 0.0,
        "login_rate_limit": "100/minute",
        "register_rate_limit": "100/minute",
        "api_rate_limit": "1000/minute",
    }
    values.update(overrides)
    return Settings(**values)


def register_and_login(client: TestClient, email: str, password: str = PASSWORD) -> dict[str, str]:
    resp = client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert resp.status_code == 201, resp.text
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def make_admin(client: TestClient, email: str = "admin@example.com") -> dict[str, str]:
    from gateway.db import User

    headers = register_and_login(client, email)
    with client.app.state.session_factory() as db:
        user = db.query(User).filter_by(email=email).one()
        user.role = "admin"
        db.commit()
    return headers
