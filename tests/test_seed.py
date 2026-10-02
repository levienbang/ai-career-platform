from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from app.db.models import Company, Job, JobSkill, Skill, SkillAlias
from scripts import seed as seed_script


def test_seed_loads_data_file_and_is_idempotent(db_session, monkeypatch) -> None:
    session_factory = sessionmaker(bind=db_session.get_bind())
    monkeypatch.setattr(seed_script, "SessionLocal", session_factory)

    seed_script.seed()
    seed_script.seed()

    seed_data = seed_script.load_seed_data()
    assert db_session.scalar(select(func.count()).select_from(Job)) == len(seed_data["jobs"])
    assert db_session.scalar(select(func.count()).select_from(Company)) == len(seed_data["jobs"])
    assert db_session.scalar(select(func.count()).select_from(Skill)) == len(seed_data["taxonomy"])
    assert db_session.scalar(select(func.count()).select_from(SkillAlias)) == sum(
        len(definition["aliases"]) for definition in seed_data["taxonomy"].values()
    )
    assert db_session.scalar(select(func.count()).select_from(JobSkill)) == sum(
        len(job["skills"]) for job in seed_data["jobs"]
    )
    assert set(db_session.scalars(select(Job.source_url))) == {
        job["source_url"] for job in seed_data["jobs"]
    }
