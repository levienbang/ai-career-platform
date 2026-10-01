"""Allow jobs without a company and track skill origin."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261001_03"
down_revision: str | None = "20260920_02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("jobs") as batch:
        batch.alter_column("company_id", existing_type=sa.Integer(), nullable=True)
    with op.batch_alter_table("skills") as batch:
        batch.add_column(
            sa.Column("origin", sa.String(20), nullable=False, server_default="curated")
        )
        batch.create_check_constraint("ck_skills_origin", "origin IN ('curated', 'extracted')")


def downgrade() -> None:
    bind = op.get_bind()
    missing_companies = bind.scalar(sa.text("SELECT COUNT(*) FROM jobs WHERE company_id IS NULL"))
    if missing_companies:
        raise RuntimeError("Cannot downgrade: jobs with no company exist; assign a company first")
    with op.batch_alter_table("skills") as batch:
        batch.drop_constraint("ck_skills_origin", type_="check")
        batch.drop_column("origin")
    with op.batch_alter_table("jobs") as batch:
        batch.alter_column("company_id", existing_type=sa.Integer(), nullable=False)
