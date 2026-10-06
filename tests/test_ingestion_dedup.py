import importlib.util
from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from app.ingestion.cleaner import clean_job_record
from app.ingestion.pipeline import JobIngestionPipeline, build_raw_record_hash
from app.schemas.ingestion import RawJobRecord


def test_raw_hash_cleaning_and_key_order():
    first = {"title": " AI  Engineer ", "description": " Python  work ", "company": "null"}
    second = {"company": None, "description": "Python work", "title": "AI Engineer"}
    assert build_raw_record_hash(RawJobRecord(**clean_job_record(first))) == build_raw_record_hash(
        RawJobRecord(**second)
    )
    assert build_raw_record_hash(RawJobRecord(**second, source_record_hash="a" * 64)) == "a" * 64
    for invalid in ["A" * 64, "xyz", None]:
        record = RawJobRecord(**second, source_record_hash=invalid)
        value = build_raw_record_hash(record)
        assert len(value) == 64 and value != invalid
        assert value == build_raw_record_hash(record)


def test_raw_dedup_before_extraction(db_session, fake_extractor):
    calls = []
    original = fake_extractor.extract
    fake_extractor.extract = lambda record: (calls.append(record), original(record))[1]
    pipeline = JobIngestionPipeline(db_session, fake_extractor)
    record = {"title": "AI", "description": "Python work"}
    first = pipeline.import_records([record, record])
    assert (first.inserted, first.skipped_duplicates, len(calls)) == (1, 1, 1)
    calls.clear()
    second = pipeline.import_records([record, record])
    assert (second.skipped_duplicates, len(calls)) == (2, 0)


def test_raw_hash_migration_upgrade_unique_downgrade():
    path = Path(__file__).parents[1] / "migrations/versions/20261002_04_job_raw_hash.py"
    spec = importlib.util.spec_from_file_location("m10_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    assert migration.down_revision == "20261001_03"
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE jobs (id INTEGER PRIMARY KEY)"))
        connection.execute(text("INSERT INTO jobs (id) VALUES (1)"))
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
        assert connection.scalar(text("SELECT raw_hash FROM jobs WHERE id = 1")) is None
        connection.execute(text("INSERT INTO jobs (id, raw_hash) VALUES (2, 'abc')"))
        with connection.begin_nested(), pytest.raises(IntegrityError):
            connection.execute(text("INSERT INTO jobs (id, raw_hash) VALUES (3, 'abc')"))
        with Operations.context(MigrationContext.configure(connection)):
            migration.downgrade()
        assert "raw_hash" not in [c["name"] for c in inspect(connection).get_columns("jobs")]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        ("", []),
        ("['SQL', 'Python', 'sql']", ["SQL", "Python"]),
        ([" Python ", 2, "python", ""], ["Python"]),
        ("SQL;Python\nDocker", ["SQL", "Python", "Docker"]),
        ("[1, 2", ["[1", "2"]),
        ("__import__('os')", ["__import__('os')"]),
        (["x" * 101, "SQL"], ["SQL"]),
        ("[1, 2]", []),
    ],
)
def test_parse_skill_list(value, expected):
    from app.ingestion.structured import parse_skill_list

    assert parse_skill_list(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("3 năm", 3),
        ("6 tháng", 0),
        ("18 tháng", 1),
        ("Không yêu cầu", 0),
        ("2 years", 2),
        ("3 YRS", 3),
        ("18 months", 1),
        ("no experience", 0),
        ("unknown", None),
        (4, 4),
        (-1, None),
        (None, None),
        (True, None),
    ],
)
def test_parse_experience(value, expected):
    from app.ingestion.structured import parse_experience_years

    assert parse_experience_years(value) == expected


def test_extended_fast_path_and_evidence(db_session, taxonomy, fake_extractor):
    from sqlalchemy import select

    from app.db.models import Job, Skill

    fake_extractor.extract = lambda _: pytest.fail("must not call LLM")
    result = JobIngestionPipeline(db_session, fake_extractor).import_records(
        [
            {
                "title": "Kỹ sư dữ liệu",
                "description": "Lập trình Python",
                "company": None,
                "location": "hà nội",
                "employment_type": "Toàn thời gian",
                "experience": "3 năm",
                "technical_skills": "['Python', 'Spark']",
                "required_skills": ["Python", "Invented"],
            }
        ]
    )
    assert (result.inserted, result.failed) == (1, 0)
    job = db_session.scalar(select(Job))
    assert (job.location, job.employment_type, job.experience_years_min) == (
        "hà nội",
        "Toàn thời gian",
        3,
    )
    assert [(item.skill.canonical_name, item.requirement_type) for item in job.skills] == [
        ("Python", "required")
    ]
    assert db_session.scalar(select(Skill).where(Skill.canonical_name == "Invented")) is None


def test_llm_groups_original_indices_and_logs(db_session, fake_extractor, caplog):
    from app.config import Settings

    calls = []
    original = fake_extractor.extract_many

    def extract_many(records):
        calls.append(records)
        outputs = original(records)
        if len(calls) == 2:
            outputs[1] = None
        return outputs

    fake_extractor.extract_many = extract_many
    records = [{"title": f"Job {i}", "description": f"Python work {i}"} for i in range(10)]
    records.insert(2, {"description": ""})
    records.insert(4, records[0])
    with caplog.at_level("INFO", logger="app.ingestion.pipeline"):
        result = JobIngestionPipeline(
            db_session, fake_extractor, Settings(llm_batch_size=4, _env_file=None)
        ).import_records(records)
    assert [len(group) for group in calls] == [4, 4, 2]
    assert (result.inserted, result.failed, result.skipped_duplicates) == (9, 2, 1)
    assert [e.record for e in result.errors] == [3, 8]
    assert (
        "processed=12 skipped_before_llm=1 deterministic=0 llm_records=10 llm_calls=3 failed=2"
        in caplog.text
    )
    assert "Python work" not in caplog.text


def test_fast_path_disabled_and_six_key_payload(db_session, fake_extractor):
    from app.config import Settings

    calls = []
    original = fake_extractor.extract_many
    fake_extractor.extract_many = lambda group: (calls.append(group), original(group))[1]
    source = {
        "title": "AI",
        "company": None,
        "location": "hà nội",
        "employment_type": "Toàn thời gian",
        "description": "Python work",
        "source_url": None,
    }
    result = JobIngestionPipeline(
        db_session, fake_extractor, Settings(llm_structured_fast_path=False, _env_file=None)
    ).import_records([{**source, "technical_skills": "['Python']"}])
    assert result.inserted == 1 and len(calls) == 1
    result = JobIngestionPipeline(db_session, fake_extractor).import_records(
        [{**source, "title": "Another job"}]
    )
    assert result.inserted == 1 and len(calls) == 2


def test_technical_skills_required_and_preferred_priority(db_session, fake_extractor):
    from sqlalchemy import select

    from app.db.models import Job

    fake_extractor.extract_many = lambda _: pytest.fail("must not call LLM")
    result = JobIngestionPipeline(db_session, fake_extractor).import_records(
        [
            {
                "title": "Data",
                "description": "Work with Docker",
                "technical_skills": "['Spark', 'SQL']",
                "preferred_skills": ["Docker"],
                "experience": "18 tháng",
            },
            {
                "title": "Backend",
                "description": "Python and SQL",
                "required_skills": "Python\nSQL",
            },
        ]
    )
    assert result.inserted == 2
    jobs = list(db_session.scalars(select(Job).order_by(Job.id)))
    assert jobs[0].experience_years_min == 1
    assert [(s.skill.canonical_name, s.requirement_type) for s in jobs[0].skills] == [
        ("Spark", "required"),
        ("SQL", "required"),
        ("Docker", "preferred"),
    ]
    assert [s.skill.canonical_name for s in jobs[1].skills] == ["Python", "SQL"]


def test_source_url_pre_llm_and_business_dedup(db_session, fake_extractor):
    from app.config import Settings

    calls = []
    original = fake_extractor.extract_many
    fake_extractor.extract_many = lambda group: (calls.append(group), original(group))[1]
    pipeline = JobIngestionPipeline(db_session, fake_extractor, Settings(_env_file=None))
    first = {"title": "AI", "description": "Python work", "source_url": "https://example.com/1"}
    assert pipeline.import_records([first, {**first, "description": "Changed text"}]).inserted == 1
    assert len(calls[0]) == 1
    calls.clear()
    assert (
        pipeline.import_records([{**first, "description": "Changed again"}]).skipped_duplicates == 1
    )
    assert not calls
    # Different raw payload and URL still meet the existing business content hash.
    second = {**first, "source_url": "https://example.com/2", "salary": "ignored"}
    assert pipeline.import_records([second]).skipped_duplicates == 1
    assert len(calls) == 1


def test_persist_savepoint_and_sorted_original_errors(db_session, fake_extractor, monkeypatch):
    from sqlalchemy import select
    from sqlalchemy.exc import IntegrityError

    from app.db.models import Job, Skill
    from app.db.repositories import JobRepository

    add = JobRepository.add

    def guarded_add(repository, job):
        if job.title == "Rejected":
            raise IntegrityError("insert", {}, Exception("constraint"))
        return add(repository, job)

    monkeypatch.setattr(JobRepository, "add", guarded_add)
    result = JobIngestionPipeline(db_session, fake_extractor).import_records(
        [
            {
                "title": "Rejected",
                "description": "UniqueBadSkill",
                "required_skills": ["UniqueBadSkill"],
            },
            {"description": ""},
            {"title": "Good", "description": "Python", "technical_skills": "['Python']"},
        ]
    )
    assert (result.inserted, result.failed) == (1, 2)
    assert [e.record for e in result.errors] == [1, 2]
    assert db_session.scalar(select(Job)).title == "Good"
    assert db_session.scalar(select(Skill).where(Skill.canonical_name == "UniqueBadSkill")) is None


def test_import_api_batch_partial_indices(client, db_session, monkeypatch):
    import json
    from types import SimpleNamespace

    from app.api.jobs import get_job_extractor
    from app.config import Settings, get_settings
    from app.ingestion.extractor import LangChainJobExtractor
    from app.main import app

    model_calls = []

    def invoke(messages):
        model_calls.append(messages)
        return {
            "items": [
                {"record_index": 1, "title": "Model", "description": "Python"},
                {"record_index": 2, "title": "Model", "description": "Python"},
                {"record_index": 2, "title": "Model", "description": "Python"},
                {"record_index": 9, "title": "Model", "description": "Python"},
                {"record_index": 3, "title": "Model", "description": "Python"},
            ]
        }

    monkeypatch.setattr(
        "app.ingestion.extractor.build_structured_chat_model",
        lambda *_args, **_kwargs: SimpleNamespace(
            provider="deepseek", runnable=SimpleNamespace(invoke=invoke)
        ),
    )
    settings = Settings(deepseek_api_key="fake", llm_batch_size=4, _env_file=None)
    extractor = LangChainJobExtractor(settings)
    app.dependency_overrides[get_job_extractor] = lambda: extractor
    app.dependency_overrides[get_settings] = lambda: settings
    payload = [
        {"title": "Structured", "description": "Python", "technical_skills": "['Python']"},
        {"title": "LLM 1", "description": "Python work 1"},
        {"title": "Invalid", "description": ""},
        {"title": "LLM 2", "description": "Python work 2"},
        {"title": "LLM 3", "description": "Python work 3"},
    ]
    response = client.post(
        "/jobs/import", files={"file": ("jobs.json", json.dumps(payload), "application/json")}
    )
    assert response.status_code == 200
    body = response.json()
    assert (body["inserted"], body["failed"]) == (3, 2)
    assert [error["record"] for error in body["errors"]] == [3, 4]
    assert len(model_calls) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("llm_batch_size", 0),
        ("llm_batch_size", 26),
        ("llm_requests_per_minute", -1),
        ("llm_requests_per_minute", 1001),
    ],
)
def test_ingestion_settings_bounds(field, value):
    from pydantic import ValidationError

    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(**{field: value}, _env_file=None)
