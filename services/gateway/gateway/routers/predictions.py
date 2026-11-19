from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from gateway.db import Prediction, User, get_db
from gateway.model_client import ModelClient, ModelRejectedInput, ModelServiceError, ModelServiceUnavailable
from gateway.observability import request_id_ctx
from gateway.ratelimit import limit_by_user
from gateway.schemas import PredictionCreate, PredictionOut, PredictionPage, RiskLevel
from gateway.security import get_current_user

router = APIRouter(
    prefix="/api/v1/predictions",
    tags=["predictions"],
    dependencies=[Depends(limit_by_user("api", "api_rate_limit"))],
)


def _client(request: Request) -> ModelClient:
    return request.app.state.model_client


def _owned(db: Session, prediction_id: uuid.UUID, user: User) -> Prediction:
    # Ownership is part of the query, so another user's id behaves exactly
    # like an id that does not exist (no information leak, no IDOR).
    prediction = db.scalar(select(Prediction).where(Prediction.id == prediction_id, Prediction.owner_id == user.id))
    if prediction is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "prediction not found")
    return prediction


@router.post("", response_model=PredictionOut, status_code=status.HTTP_201_CREATED)
def create_prediction(
    body: PredictionCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    client: ModelClient = Depends(_client),
) -> Prediction:
    try:
        result = client.predict(body.features)
    except ModelRejectedInput as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, exc.detail) from exc
    except ModelServiceUnavailable as exc:
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "prediction service temporarily unavailable", headers
        ) from exc
    except ModelServiceError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "prediction service error") from exc

    prediction = Prediction(
        owner_id=user.id,
        machine_id=body.machine_id,
        features=body.features,  # stored verbatim as JSON, including inputs the model ignores
        risk_level=result["risk_level"],
        probabilities=result["probabilities"],
        model_version=result["model_version"],
        request_id=request_id_ctx.get(),
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


@router.get("", response_model=PredictionPage)
def list_predictions(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    machine_id: str | None = Query(default=None, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"),
    risk_level: RiskLevel | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=100_000),
) -> PredictionPage:
    filters = [Prediction.owner_id == user.id]
    if machine_id:
        filters.append(Prediction.machine_id == machine_id)
    if risk_level:
        filters.append(Prediction.risk_level == risk_level)
    total = db.scalar(select(func.count()).select_from(Prediction).where(*filters)) or 0
    rows = db.scalars(
        select(Prediction).where(*filters).order_by(Prediction.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return PredictionPage(
        items=[PredictionOut.model_validate(r) for r in rows], total=total, limit=limit, offset=offset
    )


@router.get("/{prediction_id}", response_model=PredictionOut)
def get_prediction(
    prediction_id: uuid.UUID, user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> Prediction:
    return _owned(db, prediction_id, user)


@router.delete("/{prediction_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_prediction(
    prediction_id: uuid.UUID, user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> Response:
    db.delete(_owned(db, prediction_id, user))
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
