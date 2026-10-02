"""Persist alias provenance and learning decisions.

Downgrade permanently loses learned decisions and alias provenance.
"""

import sqlalchemy as sa
from alembic import op

revision = "20261003_05"
down_revision = "20261002_04"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("skill_aliases") as batch:
        batch.add_column(
            sa.Column("source", sa.String(20), nullable=False, server_default="curated")
        )
        batch.add_column(sa.Column("confidence", sa.Numeric(3, 2)))
        batch.add_column(sa.Column("reason", sa.Text()))
        batch.add_column(
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            )
        )
        batch.create_check_constraint(
            "ck_skill_aliases_source", "source IN ('curated', 'merge', 'llm')"
        )
    op.create_table(
        "skill_decisions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("match_key", sa.String(255), nullable=False, unique=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("decision", sa.String(20), nullable=False),
        sa.Column("skill_id", sa.Integer(), sa.ForeignKey("skills.id", ondelete="SET NULL")),
        sa.Column("confidence", sa.Numeric(3, 2)),
        sa.Column("reason", sa.Text()),
        sa.Column("model", sa.String(100)),
        sa.Column(
            "decided_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint(
            "decision IN ('alias', 'new', 'pending', 'rejected')",
            name="ck_skill_decisions_decision",
        ),
    )


def downgrade() -> None:
    op.drop_table("skill_decisions")
    with op.batch_alter_table("skill_aliases") as batch:
        batch.drop_constraint("ck_skill_aliases_source", type_="check")
        for column in ("created_at", "reason", "confidence", "source"):
            batch.drop_column(column)
