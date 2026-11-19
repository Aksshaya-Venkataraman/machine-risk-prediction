from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from fastapi import Request
from sqlalchemy import JSON, DateTime, ForeignKey, Index, String, Uuid, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

# JSONB on PostgreSQL, plain JSON elsewhere. Storing raw features as JSON means
# a new sensor input (e.g. humidity_pct) is saved without a schema migration.
JSONType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), default="user")
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (Index("ix_predictions_owner_created", "owner_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    machine_id: Mapped[str] = mapped_column(String(64), index=True)
    features: Mapped[dict[str, Any]] = mapped_column(JSONType)
    risk_level: Mapped[str] = mapped_column(String(16), index=True)
    probabilities: Mapped[dict[str, float]] = mapped_column(JSONType)
    model_version: Mapped[str] = mapped_column(String(64))
    request_id: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False}, poolclass=StaticPool)
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def get_db(request: Request) -> Iterator[Session]:
    with request.app.state.session_factory() as session:
        yield session
