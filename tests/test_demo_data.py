"""Compare the new commands with snapshots captured before the M14 refactor."""

import json
from hashlib import sha256
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.db.models import Company, Job, JobSkill, Skill, SkillAlias
from app.skills.taxonomy import load_skill_aliases
from scripts import load_demo_jobs, sync_skills

BASELINE = Path(__file__).parent / "fixtures" / "milestone14_baseline.json"


def test_taxonomy_directory_default_empty_and_override(monkeypatch, tmp_path):
    monkeypatch.delenv("SKILL_DATA_DIR", raising=False)
    assert Settings(_env_file=None).skill_data_dir == Path("data/taxonomy")
    monkeypatch.setenv("SKILL_DATA_DIR", "")
    assert Settings(_env_file=None).skill_data_dir == Path("data/taxonomy")
    monkeypatch.setenv("SKILL_DATA_DIR", str(tmp_path))
    assert Settings(_env_file=None).skill_data_dir == tmp_path


def database_snapshot(session):
    result = {"taxonomy": load_skill_aliases(sync_skills.DATA_DIR / "skill_aliases.json")}
    for model in (Company, Job, Skill, SkillAlias, JobSkill):
        columns = [
            column
            for column in model.__table__.columns
            if column.name not in ("created_at", "updated_at", "ingested_at")
        ]
        result[model.__tablename__] = sorted(
            [
                dict(zip([column.name for column in columns], row, strict=True))
                for row in session.execute(select(*columns))
            ],
            key=lambda row: json.dumps(row, sort_keys=True, default=str),
        )
    return result


@pytest.mark.parametrize("sync_first", [False, True])
def test_demo_commands_match_previous_database_and_are_idempotent(
    db_session, monkeypatch, sync_first
):
    factory = sessionmaker(bind=db_session.get_bind())
    monkeypatch.setattr(sync_skills, "SessionLocal", factory)
    monkeypatch.setattr(load_demo_jobs, "SessionLocal", factory)
    if sync_first:
        sync_skills.sync_skills()
    load_demo_jobs.load_demo_jobs()
    first = database_snapshot(db_session)
    expected = json.loads(BASELINE.read_text())
    assert {
        key: sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()
        for key, value in first.items()
    } == expected
    db_session.commit()
    sync_skills.sync_skills()
    sync_skills.sync_skills()
    load_demo_jobs.load_demo_jobs()
    assert database_snapshot(db_session) == first


def test_sync_only_creates_skills(db_session, monkeypatch):
    monkeypatch.setattr(sync_skills, "SessionLocal", sessionmaker(bind=db_session.get_bind()))
    sync_skills.sync_skills()
    assert db_session.scalar(select(func.count()).select_from(Skill)) == 55
    assert db_session.scalar(select(func.count()).select_from(SkillAlias)) == 56
    assert db_session.scalar(select(func.count()).select_from(Job)) == 0
    assert db_session.scalar(select(func.count()).select_from(Company)) == 0


def test_demo_command_rolls_back_taxonomy_and_jobs_on_failure(db_session, monkeypatch):
    monkeypatch.setattr(load_demo_jobs, "SessionLocal", sessionmaker(bind=db_session.get_bind()))
    jobs = load_demo_jobs.read_demo_jobs()
    jobs[1]["skills"] = ["Unknown demo skill"]
    monkeypatch.setattr(load_demo_jobs, "read_demo_jobs", lambda: jobs)
    with pytest.raises(KeyError, match="Unknown demo skill"):
        load_demo_jobs.load_demo_jobs()
    for model in (Company, Job, Skill, SkillAlias, JobSkill):
        assert db_session.scalar(select(func.count()).select_from(model)) == 0
