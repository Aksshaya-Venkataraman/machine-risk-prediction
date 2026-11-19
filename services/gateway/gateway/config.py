from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_WEAK_MARKERS = ("change", "secret", "password", "example", "xxxx")


def _strong(value: SecretStr, name: str) -> SecretStr:
    raw = value.get_secret_value()
    if len(raw) < 32:
        raise ValueError(f"{name} must be at least 32 characters")
    if any(marker in raw.lower() for marker in _WEAK_MARKERS):
        raise ValueError(f"{name} looks like a placeholder; generate a random value")
    return value


def _split(value: object) -> object:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    """All configuration comes from environment variables; nothing secret is in code."""

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    environment: Literal["development", "test", "production"] = "production"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    database_url: SecretStr
    jwt_secret_key: SecretStr
    jwt_issuer: str = "machine-risk-gateway"
    access_token_minutes: int = Field(default=30, ge=5, le=240)

    model_service_url: str = "http://model-service:8001"
    internal_api_token: SecretStr
    model_admin_token: SecretStr
    model_timeout_seconds: float = Field(default=2.0, gt=0, le=30)
    model_connect_timeout_seconds: float = Field(default=0.5, gt=0, le=10)
    model_max_retries: int = Field(default=2, ge=0, le=5)
    model_backoff_seconds: float = Field(default=0.2, ge=0, le=5)
    breaker_failure_threshold: int = Field(default=5, ge=1)
    breaker_recovery_seconds: float = Field(default=30.0, gt=0)

    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    allowed_hosts: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["localhost"])

    rate_limit_storage_uri: SecretStr = SecretStr("memory://")
    login_rate_limit: str = "5/minute"
    register_rate_limit: str = "5/hour"
    api_rate_limit: str = "120/minute"

    metrics_port: int = Field(default=9100, ge=0, le=65535)
    max_body_bytes: int = Field(default=32_768, ge=1024)
    enable_docs: bool = False

    @field_validator("cors_origins", "allowed_hosts", mode="before")
    @classmethod
    def _split_lists(cls, value: object) -> object:
        return _split(value)

    @field_validator("cors_origins")
    @classmethod
    def _no_wildcard(cls, value: list[str]) -> list[str]:
        if "*" in value:
            raise ValueError("CORS_ORIGINS must list explicit origins, not '*'")
        return value

    @field_validator("jwt_secret_key")
    @classmethod
    def _jwt_strong(cls, value: SecretStr) -> SecretStr:
        return _strong(value, "JWT_SECRET_KEY")

    @field_validator("internal_api_token")
    @classmethod
    def _internal_strong(cls, value: SecretStr) -> SecretStr:
        return _strong(value, "INTERNAL_API_TOKEN")

    @field_validator("model_admin_token")
    @classmethod
    def _admin_strong(cls, value: SecretStr) -> SecretStr:
        return _strong(value, "MODEL_ADMIN_TOKEN")


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
