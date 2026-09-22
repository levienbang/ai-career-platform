"""Require unique job source URLs when present."""

from collections.abc import Sequence

from alembic import op

revision: str = "20260920_02"
down_revision: str | None = "20260917_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_unique_constraint("uq_jobs_source_url", "jobs", ["source_url"])


def downgrade() -> None:
    op.drop_constraint("uq_jobs_source_url", "jobs", type_="unique")
