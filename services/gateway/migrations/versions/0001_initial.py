"""initial schema: users and predictions

Revision ID: 0001
Revises:
Create Date: 2026-11-19
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(254), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "predictions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("machine_id", sa.String(64), nullable=False),
        sa.Column("features", json_type, nullable=False),
        sa.Column("risk_level", sa.String(16), nullable=False),
        sa.Column("probabilities", json_type, nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("request_id", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_predictions_owner_id", "predictions", ["owner_id"])
    op.create_index("ix_predictions_machine_id", "predictions", ["machine_id"])
    op.create_index("ix_predictions_risk_level", "predictions", ["risk_level"])
    op.create_index("ix_predictions_owner_created", "predictions", ["owner_id", "created_at"])


def downgrade() -> None:
    op.drop_table("predictions")
    op.drop_table("users")
