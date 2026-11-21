from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from app.db.models import Company, Job, Skill, SkillAlias
from app.skills.taxonomy import load_skill_aliases
from scripts import load_demo_jobs, sync_skills


def test_sync_and_demo_loading(db_session, monkeypatch):
    factory = sessionmaker(bind=db_session.get_bind())
    monkeypatch.setattr(sync_skills, "SessionLocal", factory)
    monkeypatch.setattr(load_demo_jobs, "SessionLocal", factory)
    sync_skills.sync_skills()
    expected = load_skill_aliases(sync_skills.DATA_DIR / "skill_aliases.json")
    assert db_session.scalar(select(func.count()).select_from(Skill)) == len(expected)
    assert db_session.scalar(select(func.count()).select_from(SkillAlias)) > 0
    assert db_session.scalar(select(func.count()).select_from(Job)) == 0
    assert db_session.scalar(select(func.count()).select_from(Company)) == 0
    db_session.commit()
    load_demo_jobs.load_demo_jobs()
    assert db_session.scalar(select(func.count()).select_from(Job)) == len(
        load_demo_jobs.read_demo_jobs()
    )
    db_session.commit()
    sync_skills.sync_skills()
    load_demo_jobs.load_demo_jobs()
    assert db_session.scalar(select(func.count()).select_from(Job)) == len(
        load_demo_jobs.read_demo_jobs()
    )
