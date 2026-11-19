"""Password hashing (Argon2id), JWT access tokens and auth dependencies."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from gateway.config import Settings
from gateway.db import User, get_db

_hasher = PasswordHasher()  # Argon2id with library defaults (RFC 9106 profile)
# Verified when the email is unknown so login timing does not reveal accounts.
_DUMMY_HASH = _hasher.hash("timing-equaliser-not-a-real-password")
_bearer = HTTPBearer(auto_error=False)

_UNAUTHORISED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="invalid or expired credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    try:
        return _hasher.verify(password_hash or _DUMMY_HASH, password) and password_hash is not None
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    return _hasher.check_needs_rehash(password_hash)


def create_access_token(user: User, settings: Settings) -> tuple[str, int]:
    now = datetime.now(UTC)
    lifetime = timedelta(minutes=settings.access_token_minutes)
    claims = {
        "sub": str(user.id),
        "iss": settings.jwt_issuer,
        "iat": now,
        "nbf": now,
        "exp": now + lifetime,
        "jti": uuid.uuid4().hex,
        "typ": "access",
    }
    token = jwt.encode(claims, settings.jwt_secret_key.get_secret_value(), algorithm="HS256")
    return token, int(lifetime.total_seconds())


def _decode(token: str, settings: Settings) -> dict:
    return jwt.decode(
        token,
        settings.jwt_secret_key.get_secret_value(),
        algorithms=["HS256"],  # pinned: never accept "none" or a key-confusion alg
        issuer=settings.jwt_issuer,
        options={"require": ["sub", "exp", "iat", "iss", "typ"]},
    )


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _UNAUTHORISED
    settings: Settings = request.app.state.settings
    try:
        claims = _decode(credentials.credentials, settings)
        if claims.get("typ") != "access":
            raise _UNAUTHORISED
        user_id = int(claims["sub"])
    except (jwt.PyJWTError, ValueError) as exc:
        raise _UNAUTHORISED from exc

    user = db.get(User, user_id)
    # Role and active status are read from the database on every request,
    # so deactivating a user or removing admin rights takes effect at once.
    if user is None or not user.is_active:
        raise _UNAUTHORISED
    request.state.user_id = user.id
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="admin access required")
    return user
