import importlib.util
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text


def _migration():
    path = Path(__file__).parents[1] / "migrations/versions/20261019_06_drop_unused_columns.py"
    spec = importlib.util.spec_from_file_location("drop_unused_columns_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _columns(connection, table: str) -> set[str]:
    return {column["name"] for column in inspect(connection).get_columns(table)}


def test_migration_drops_and_restores_unused_columns():
    migration = _migration()
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE companies (id INTEGER PRIMARY KEY, name VARCHAR(255), "
                "industry VARCHAR(255), location VARCHAR(255), website VARCHAR(2048))"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE job_skills (id INTEGER PRIMARY KEY, job_id INTEGER, "
                "skill_id INTEGER, importance NUMERIC(3, 2), evidence_text TEXT)"
            )
        )
        connection.execute(text("INSERT INTO companies (id, name) VALUES (1, 'Acme')"))
        connection.execute(
            text(
                "INSERT INTO job_skills (id, job_id, skill_id, evidence_text) VALUES (1, 1, 1, 'x')"
            )
        )
        context = MigrationContext.configure(connection)

        with Operations.context(context):
            migration.upgrade()
        assert _columns(connection, "companies") == {"id", "name"}
        assert _columns(connection, "job_skills") == {"id", "job_id", "skill_id", "evidence_text"}
        assert connection.scalar(text("SELECT name FROM companies WHERE id = 1")) == "Acme"

        with Operations.context(context):
            migration.downgrade()
        assert {"industry", "location", "website"} <= _columns(connection, "companies")
        assert "importance" in _columns(connection, "job_skills")
        assert connection.scalar(text("SELECT evidence_text FROM job_skills WHERE id = 1")) == "x"
