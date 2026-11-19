from __future__ import annotations

import os
import secrets

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from gateway.db import Base
from gw_support import PASSWORD, gateway_settings, make_admin, register_and_login
from sqlalchemy import create_engine

GATEWAY_DIR = os.path.join(os.path.dirname(__file__), "..")


def test_admin_routes_forbidden_for_normal_users(gateway):
    headers = register_and_login(gateway, "user@example.com")
    for method, url in [
        ("get", "/api/v1/admin/stats"),
        ("get", "/api/v1/admin/users"),
        ("post", "/api/v1/admin/model/reload"),
    ]:
        assert getattr(gateway, method)(url, headers=headers).status_code == 403
    assert gateway.get("/api/v1/admin/stats").status_code == 401


def test_admin_stats_and_model_reload(gateway, sample_features):
    admin = make_admin(gateway)
    gateway.post("/api/v1/predictions", json={"machine_id": "m-1", "features": sample_features}, headers=admin)
    stats = gateway.get("/api/v1/admin/stats", headers=admin).json()
    assert stats["predictions"] == 1
    assert stats["circuit_breaker"] == "closed"
    assert stats["model"]["version"].startswith("rf-")
    reload = gateway.post("/api/v1/admin/model/reload", headers=admin)
    assert reload.status_code == 200
    assert reload.json()["version"] == stats["model"]["version"]


def test_deactivated_user_is_locked_out_immediately(gateway):
    admin = make_admin(gateway)
    user = register_and_login(gateway, "leaver@example.com")
    user_id = gateway.get("/api/v1/auth/me", headers=user).json()["id"]
    resp = gateway.patch(f"/api/v1/admin/users/{user_id}", json={"is_active": False}, headers=admin)
    assert resp.status_code == 200
    assert gateway.get("/api/v1/auth/me", headers=user).status_code == 401
    login = gateway.post("/api/v1/auth/login", json={"email": "leaver@example.com", "password": PASSWORD})
    assert login.status_code == 401


def test_admin_cannot_demote_self(gateway):
    admin = make_admin(gateway)
    my_id = gateway.get("/api/v1/auth/me", headers=admin).json()["id"]
    resp = gateway.patch(f"/api/v1/admin/users/{my_id}", json={"role": "user"}, headers=admin)
    assert resp.status_code == 400


def test_security_headers_and_docs_disabled(gateway):
    resp = gateway.get("/health")
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["referrer-policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in resp.headers["content-security-policy"]
    assert gateway.get("/docs").status_code == 404
    assert gateway.get("/openapi.json").status_code == 404


def test_cors_allows_only_configured_origins(gateway):
    preflight = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization"}
    ok = gateway.options("/api/v1/predictions", headers={"Origin": "https://app.example.com", **preflight})
    assert ok.headers["access-control-allow-origin"] == "https://app.example.com"
    bad = gateway.options("/api/v1/predictions", headers={"Origin": "https://evil.example", **preflight})
    assert "access-control-allow-origin" not in bad.headers


def test_unknown_host_and_large_body_rejected(gateway):
    assert gateway.get("/health", headers={"Host": "attacker.example"}).status_code == 400
    big = gateway.post("/api/v1/auth/login", content=b"x" * 40_000, headers={"Content-Type": "application/json"})
    assert big.status_code == 413


@pytest.mark.parametrize(
    "override",
    [
        {"jwt_secret_key": "too-short"},
        {"jwt_secret_key": "change-me-please-" + "a" * 32},
        {"cors_origins": "*"},
    ],
    ids=["short-secret", "placeholder-secret", "wildcard-cors"],
)
def test_insecure_configuration_refused(override):
    with pytest.raises(ValueError):
        gateway_settings(**override)


def test_migrations_match_models(tmp_path):
    url = f"sqlite:///{tmp_path / 'migrate.db'}"
    cfg = Config(os.path.join(GATEWAY_DIR, "alembic.ini"))
    cfg.attributes["database_url"] = url
    command.upgrade(cfg, "head")
    engine = create_engine(url)
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    engine.dispose()
    assert diff == []
    command.downgrade(cfg, "base")


def test_cli_creates_admin(tmp_path, monkeypatch):
    from gateway import cli, config

    url = f"sqlite:///{tmp_path / 'cli.db'}"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    engine.dispose()
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("JWT_SECRET_KEY", secrets.token_urlsafe(48))
    config.get_settings.cache_clear()
    try:
        assert cli.create_admin("root@example.com", PASSWORD) == "created"
        assert cli.create_admin("root@example.com", PASSWORD) == "promoted"
        with pytest.raises(SystemExit):
            cli.create_admin("root@example.com", "short")
    finally:
        config.get_settings.cache_clear()
