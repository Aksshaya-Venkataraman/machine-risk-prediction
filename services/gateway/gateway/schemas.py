from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    field_validator,
)

Password = Annotated[str, StringConstraints(min_length=12, max_length=128)]
MachineId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")]
FeatureKey = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
FeatureText = Annotated[StrictStr, StringConstraints(pattern=r"^[A-Za-z0-9 _.-]{1,64}$")]
FeatureValue = StrictInt | StrictFloat | FeatureText
RiskLevel = Literal["Low", "Medium", "High"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RegisterRequest(_Strict):
    email: EmailStr
    password: Password

    @field_validator("email")
    @classmethod
    def _lower(cls, value: str) -> str:
        return value.lower()

    @field_validator("password")
    @classmethod
    def _not_trivial(cls, value: str) -> str:
        if len(set(value)) < 6:
            raise ValueError("password is too repetitive")
        return value


class LoginRequest(_Strict):
    email: EmailStr
    password: Annotated[str, StringConstraints(min_length=1, max_length=128)]


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"  # noqa: S105 - OAuth token type, not a secret
    expires_in: int


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    role: str
    is_active: bool
    created_at: datetime


class UserUpdate(_Strict):
    is_active: bool | None = None
    role: Literal["user", "admin"] | None = None


class PredictionCreate(_Strict):
    machine_id: MachineId
    features: dict[FeatureKey, FeatureValue] = Field(min_length=1, max_length=50)


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    machine_id: str
    features: dict[str, int | float | str]
    risk_level: RiskLevel
    probabilities: dict[str, float]
    model_version: str
    request_id: str
    created_at: datetime


class PredictionPage(BaseModel):
    items: list[PredictionOut]
    total: int
    limit: int
    offset: int


class AdminStats(BaseModel):
    users: int
    predictions: int
    predictions_by_risk: dict[str, int]
    circuit_breaker: str
    model: dict | None
