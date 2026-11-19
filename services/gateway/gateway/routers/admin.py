from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from gateway.db import Prediction, User, get_db
from gateway.model_client import ModelClient, ModelRejectedInput, ModelServiceError
from gateway.schemas import AdminStats, UserOut, UserUpdate
from gateway.security import require_admin

router = APIRouter(prefix="/api/v1/admin", tags=["admin"], dependencies=[Depends(require_admin)])
log = logging.getLogger(__name__)


@router.get("/stats", response_model=AdminStats)
def stats(request: Request, db: Session = Depends(get_db)) -> AdminStats:
    client: ModelClient = request.app.state.model_client
    by_risk = dict(db.execute(select(Prediction.risk_level, func.count()).group_by(Prediction.risk_level)).all())
    try:
        model = client.model_info()
        model.pop("features", None)
    except ModelServiceError:
        model = None
    return AdminStats(
        users=db.scalar(select(func.count()).select_from(User)) or 0,
        predictions=sum(by_risk.values()),
        predictions_by_risk={level: by_risk.get(level, 0) for level in ("Low", "Medium", "High")},
        circuit_breaker=client.breaker.state.value,
        model=model,
    )


@router.get("/users", response_model=list[UserOut])
def list_users(
    db: Session = Depends(get_db),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[User]:
    return list(db.scalars(select(User).order_by(User.id).limit(limit).offset(offset)).all())


@router.patch("/users/{user_id}", response_model=UserOut)
def update_user(
    user_id: int,
    body: UserUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
) -> User:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")
    if user.id == admin.id and (body.is_active is False or body.role == "user"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "admins cannot demote or deactivate themselves")
    if body.is_active is not None:
        user.is_active = body.is_active
    if body.role is not None:
        user.role = body.role
    db.commit()
    db.refresh(user)
    log.info("admin %s updated user %s", admin.id, user.id)
    return user


@router.post("/model/reload")
def reload_model(request: Request, admin: User = Depends(require_admin)) -> dict:
    client: ModelClient = request.app.state.model_client
    try:
        result = client.reload()
    except ModelRejectedInput as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, exc.detail) from exc
    except ModelServiceError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model service unavailable") from exc
    log.info("admin %s reloaded model to %s", admin.id, result.get("version"))
    return result
