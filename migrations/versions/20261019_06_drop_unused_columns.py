"""Drop columns that ingestion never populates.

companies.industry/website had no source, companies.location copied the first job's
location, and job_skills.importance was never extracted. Downgrade restores empty columns.
"""

import sqlalchemy as sa
from alembic import op

revision = "20261019_06"
down_revision = "20261003_05"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("companies") as batch:
        batch.drop_column("industry")
        batch.drop_column("location")
        batch.drop_column("website")
    # PostgreSQL drops the single-column importance CHECK constraint with the column.
    with op.batch_alter_table("job_skills") as batch:
        batch.drop_column("importance")


def downgrade() -> None:
    with op.batch_alter_table("job_skills") as batch:
        batch.add_column(sa.Column("importance", sa.Numeric(3, 2)))
        batch.create_check_constraint(
            "ck_job_skills_importance_range",
            "importance IS NULL OR (importance >= 0 AND importance <= 1)",
        )
    with op.batch_alter_table("companies") as batch:
        batch.add_column(sa.Column("industry", sa.String(255)))
        batch.add_column(sa.Column("location", sa.String(255)))
        batch.add_column(sa.Column("website", sa.String(2048)))
