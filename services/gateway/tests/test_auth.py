from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest
from gw_support import PASSWORD, register_and_login


def test_register_login_and_me(gateway):
    headers = register_and_login(gateway, "Alice@Example.com")
    me = gateway.get("/api/v1/auth/me", headers=headers).json()
    assert me["email"] == "alice@example.com"
    assert me["role"] == "user"
    assert "password_hash" not in me


def test_password_is_stored_as_argon2_hash(gateway):
    from gateway.db import User

    register_and_login(gateway, "hash@example.com")
    with gateway.app.state.session_factory() as db:
        stored = db.query(User).filter_by(email="hash@example.com").one().password_hash
    assert stored.startswith("$argon2id$")
    assert PASSWORD not in stored


def test_duplicate_registration_gives_generic_error(gateway):
    register_and_login(gateway, "dup@example.com")
    resp = gateway.post("/api/v1/auth/register", json={"email": "dup@example.com", "password": PASSWORD})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "registration failed"


@pytest.mark.parametrize("password", ["short", "aaaaaaaaaaaaaaaa", "x" * 129])
def test_weak_passwords_rejected(gateway, password):
    resp = gateway.post("/api/v1/auth/register", json={"email": "weak@example.com", "password": password})
    assert resp.status_code == 422


def test_login_failure_does_not_reveal_which_part_was_wrong(gateway):
    register_and_login(gateway, "bob@example.com")
    wrong_pw = gateway.post("/api/v1/auth/login", json={"email": "bob@example.com", "password": "nope-nope-nope"})
    no_user = gateway.post("/api/v1/auth/login", json={"email": "ghost@example.com", "password": PASSWORD})
    assert wrong_pw.status_code == no_user.status_code == 401
    assert wrong_pw.json() == no_user.json()


def test_unknown_fields_rejected(gateway):
    resp = gateway.post(
        "/api/v1/auth/register",
        json={"email": "role@example.com", "password": PASSWORD, "role": "admin"},
    )
    assert resp.status_code == 422


def _token(gateway, **claims):
    settings = gateway.app.state.settings
    now = datetime.now(UTC)
    base = {"sub": "1", "iss": settings.jwt_issuer, "iat": now, "exp": now + timedelta(minutes=5), "typ": "access"}
    base.update(claims)
    return jwt.encode(base, settings.jwt_secret_key.get_secret_value(), algorithm="HS256")


@pytest.mark.parametrize(
    "make_header",
    [
        lambda g: {},
        lambda g: {"Authorization": "Bearer not-a-jwt"},
        lambda g: {"Authorization": f"Bearer {_token(g, exp=datetime.now(UTC) - timedelta(minutes=1))}"},
        lambda g: {"Authorization": f"Bearer {_token(g, iss='someone-else')}"},
        lambda g: {"Authorization": f"Bearer {_token(g, typ='refresh')}"},
        lambda g: {"Authorization": "Bearer " + jwt.encode({"sub": "1"}, "wrong-key-" * 4, algorithm="HS256")},
        lambda g: {"Authorization": "Bearer eyJhbGciOiJub25lIn0.eyJzdWIiOiIxIn0."},
    ],
    ids=["missing", "garbage", "expired", "wrong-issuer", "wrong-type", "wrong-key", "alg-none"],
)
def test_invalid_tokens_rejected(gateway, make_header):
    register_and_login(gateway, "victim@example.com")
    assert gateway.get("/api/v1/auth/me", headers=make_header(gateway)).status_code == 401


def test_login_is_rate_limited(make_gateway, model_service):
    client = make_gateway(model_service._transport, model_service_url="http://testserver", login_rate_limit="3/minute")
    body = {"email": "brute@example.com", "password": "guess-guess-guess"}
    codes = [client.post("/api/v1/auth/login", json=body).status_code for _ in range(4)]
    assert codes == [401, 401, 401, 429]
    assert int(client.post("/api/v1/auth/login", json=body).headers["Retry-After"]) >= 1
