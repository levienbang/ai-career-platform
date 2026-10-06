from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from app.db.models import Company, Job, Skill, SkillAlias
from app.schemas.cv import CVExtraction, RecognizedSkill
from app.services.cv_match import build_cv_query
from app.services.skill_taxonomy import load_skill_aliases
from scripts import load_demo_jobs, sync_skills


def test_cv_query_order_limit_and_no_raw_contacts():
    profile = CVExtraction(
        summary="AI summary",
        projects=["Vision project"],
        experience=["Python internship"],
        education=["Computer Science"],
    )
    recognized = {1: RecognizedSkill(skill="Python", evidence="python@example.com 0901234567")}
    assert build_cv_query(profile, recognized) == (
        "AI summary\nPython\nVision project\nPython internship\nComputer Science"
    )
    query = build_cv_query(profile.model_copy(update={"summary": "x" * 2500}), recognized)
    assert len(query) == 2000
    assert "python@example.com" not in query and "0901234567" not in query
    assert build_cv_query(CVExtraction(experience=["Video processing"], education=["BSc"]), {}) == (
        "Video processing\nBSc"
    )


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
