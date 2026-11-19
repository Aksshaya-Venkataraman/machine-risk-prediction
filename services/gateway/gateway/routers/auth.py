from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from gateway.db import User, get_db
from gateway.ratelimit import limit_by_ip
from gateway.schemas import LoginRequest, RegisterRequest, TokenResponse, UserOut
from gateway.security import (
    create_access_token,
    get_current_user,
    hash_password,
    needs_rehash,
    verify_password,
)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
log = logging.getLogger(__name__)


@router.post(
    "/register",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(limit_by_ip("register", "register_rate_limit"))],
)
def register(body: RegisterRequest, db: Session = Depends(get_db)) -> User:
    user = User(email=body.email, password_hash=hash_password(body.password), role="user")
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        # Generic message: do not confirm whether an address is registered.
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "registration failed") from None
    db.refresh(user)
    return user


@router.post(
    "/login",
    response_model=TokenResponse,
    dependencies=[Depends(limit_by_ip("login", "login_rate_limit"))],
)
def login(body: LoginRequest, request: Request, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    valid = verify_password(body.password, user.password_hash if user else None)
    if user is None or not valid or not user.is_active:
        log.info("failed login")
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(body.password)
        db.commit()
    token, expires_in = create_access_token(user, request.app.state.settings)
    return TokenResponse(access_token=token, expires_in=expires_in)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> User:
    return user
