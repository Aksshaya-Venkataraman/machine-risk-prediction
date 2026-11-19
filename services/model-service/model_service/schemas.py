from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, StringConstraints

FeatureKey = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
FeatureValue = StrictInt | StrictFloat | Annotated[StrictStr, StringConstraints(max_length=64)] | None


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[FeatureKey, FeatureValue] = Field(min_length=1, max_length=50)


class PredictResponse(BaseModel):
    risk_level: str
    probabilities: dict[str, float]
    model_version: str
    ignored_features: list[str]


class ModelInfo(BaseModel):
    version: str
    sha256: str
    trained_at: str
    loaded_at: str
    classes: list[str]
    features: dict[str, dict]
    metrics: dict


class ReloadResponse(BaseModel):
    version: str
    warnings: list[str]
