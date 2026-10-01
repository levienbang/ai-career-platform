"""Deduplicate raw source records before extraction."""

import sqlalchemy as sa
from alembic import op

revision = "20261002_04"
down_revision = "20261001_03"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("jobs") as batch:
        batch.add_column(sa.Column("raw_hash", sa.String(64), nullable=True))
        batch.create_unique_constraint("uq_jobs_raw_hash", ["raw_hash"])


def downgrade() -> None:
    with op.batch_alter_table("jobs") as batch:
        batch.drop_constraint("uq_jobs_raw_hash", type_="unique")
        batch.drop_column("raw_hash")
