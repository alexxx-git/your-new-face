"""add target age to tasks

Revision ID: 20260510_0002
Revises: 20260509_0001
Create Date: 2026-05-10
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "20260510_0002"
down_revision: str | None = "20260509_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("tasks", sa.Column("target_age", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("tasks", "target_age")
