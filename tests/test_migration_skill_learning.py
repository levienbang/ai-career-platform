import importlib.util
from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError


def test_alias_learning_upgrade_defaults_constraints_and_downgrade():
    path = Path(__file__).parents[1] / "migrations/versions/20261003_05_skill_alias_learning.py"
    spec = importlib.util.spec_from_file_location("m13_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    assert migration.down_revision == "20261002_04"
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("PRAGMA foreign_keys = ON"))
        connection.execute(text("CREATE TABLE skills (id INTEGER PRIMARY KEY)"))
        connection.execute(
            text(
                "CREATE TABLE skill_aliases (id INTEGER PRIMARY KEY, "
                "skill_id INTEGER REFERENCES skills(id), alias VARCHAR(255))"
            )
        )
        connection.execute(text("INSERT INTO skills VALUES (1)"))
        connection.execute(text("INSERT INTO skill_aliases VALUES (1, 1, 'py')"))
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
        row = connection.execute(
            text("SELECT source, confidence, reason, created_at FROM skill_aliases WHERE id = 1")
        ).one()
        assert row[0] == "curated" and row[1:3] == (None, None) and row[3]
        for statement in [
            "UPDATE skill_aliases SET source='invalid' WHERE id=1",
            "INSERT INTO skill_decisions (match_key, name, decision) "
            "VALUES ('py', 'py', 'invalid')",
        ]:
            with connection.begin_nested(), pytest.raises(IntegrityError):
                connection.execute(text(statement))
        connection.execute(
            text(
                "INSERT INTO skill_decisions (match_key, name, decision, skill_id) "
                "VALUES ('py', 'py', 'new', 1)"
            )
        )
        with connection.begin_nested(), pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO skill_decisions (match_key, name, decision) "
                    "VALUES ('py', 'other', 'pending')"
                )
            )
        connection.execute(text("DELETE FROM skill_aliases"))
        connection.execute(text("DELETE FROM skills WHERE id=1"))
        assert connection.scalar(text("SELECT skill_id FROM skill_decisions")) is None
        with Operations.context(MigrationContext.configure(connection)):
            migration.downgrade()
        assert "skill_decisions" not in inspect(connection).get_table_names()
        assert [column["name"] for column in inspect(connection).get_columns("skill_aliases")] == [
            "id",
            "skill_id",
            "alias",
        ]
