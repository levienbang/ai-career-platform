import importlib.util
from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text


def _migration():
    path = (
        Path(__file__).parents[1]
        / "migrations/versions/20261001_03_optional_company_skill_origin.py"
    )
    spec = importlib.util.spec_from_file_location("milestone9_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_upgrade_downgrade_and_null_guard():
    migration = _migration()
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE companies (id INTEGER PRIMARY KEY)"))
        connection.execute(
            text(
                "CREATE TABLE jobs (id INTEGER PRIMARY KEY, company_id INTEGER NOT NULL "
                "REFERENCES companies(id))"
            )
        )
        connection.execute(
            text("CREATE TABLE skills (id INTEGER PRIMARY KEY, canonical_name VARCHAR(255))")
        )
        connection.execute(text("INSERT INTO companies (id) VALUES (1)"))
        connection.execute(text("INSERT INTO jobs (id, company_id) VALUES (1, 1)"))
        connection.execute(text("INSERT INTO skills (id, canonical_name) VALUES (1, 'Python')"))
        context = MigrationContext.configure(connection)
        with Operations.context(context):
            migration.upgrade()
        assert inspect(connection).get_columns("jobs")[1]["nullable"]
        assert connection.scalar(text("SELECT origin FROM skills WHERE id = 1")) == "curated"
        connection.execute(text("INSERT INTO jobs (id, company_id) VALUES (2, NULL)"))
        with Operations.context(context), pytest.raises(RuntimeError, match="assign a company"):
            migration.downgrade()
        connection.execute(text("DELETE FROM jobs WHERE id = 2"))
        with Operations.context(context):
            migration.downgrade()
        assert not inspect(connection).get_columns("jobs")[1]["nullable"]
        assert "origin" not in [
            column["name"] for column in inspect(connection).get_columns("skills")
        ]
