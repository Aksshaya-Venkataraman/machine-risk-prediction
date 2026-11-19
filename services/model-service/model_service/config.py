from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_WEAK_MARKERS = ("change", "secret", "password", "example", "xxxx")


def _check_secret(value: SecretStr, name: str) -> SecretStr:
    raw = value.get_secret_value()
    if len(raw) < 32:
        raise ValueError(f"{name} must be at least 32 characters")
    if any(marker in raw.lower() for marker in _WEAK_MARKERS):
        raise ValueError(f"{name} looks like a placeholder; generate a random value")
    return value


class Settings(BaseSettings):
    """Configuration comes only from environment variables (never from code)."""

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    environment: Literal["development", "test", "production"] = "production"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    model_path: Path = Path("/app/artifacts/model.joblib")
    manifest_path: Path = Path("/app/artifacts/model_manifest.json")

    internal_api_token: SecretStr
    model_admin_token: SecretStr

    allowed_hosts: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["model-service", "localhost", "127.0.0.1"]
    )
    metrics_port: int = Field(default=9100, ge=0, le=65535)
    max_body_bytes: int = Field(default=16_384, ge=1024)
    enable_docs: bool = False

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def _split_hosts(cls, value: object) -> object:
        if isinstance(value, str):
            return [h.strip() for h in value.split(",") if h.strip()]
        return value

    @field_validator("internal_api_token")
    @classmethod
    def _internal_strong(cls, value: SecretStr) -> SecretStr:
        return _check_secret(value, "INTERNAL_API_TOKEN")

    @field_validator("model_admin_token")
    @classmethod
    def _admin_strong(cls, value: SecretStr) -> SecretStr:
        return _check_secret(value, "MODEL_ADMIN_TOKEN")


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
