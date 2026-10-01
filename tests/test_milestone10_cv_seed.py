from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from app.db.models import Company, Job, Skill, SkillAlias
from app.schemas.cv import CVExtraction, RecognizedSkill
from app.services.cv_match import build_cv_query
from scripts import seed as seed_module


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


def test_taxonomy_only_and_default_seed(db_session, monkeypatch):
    monkeypatch.setattr(seed_module, "SessionLocal", sessionmaker(bind=db_session.get_bind()))
    seed_module.seed(taxonomy_only=True)
    expected = seed_module.load_seed_data()
    assert db_session.scalar(select(func.count()).select_from(Skill)) == len(expected["taxonomy"])
    assert db_session.scalar(select(func.count()).select_from(SkillAlias)) > 0
    assert db_session.scalar(select(func.count()).select_from(Job)) == 0
    assert db_session.scalar(select(func.count()).select_from(Company)) == 0
    db_session.commit()
    seed_module.seed()
    assert db_session.scalar(select(func.count()).select_from(Job)) == len(expected["jobs"])
    db_session.commit()
    seed_module.seed(taxonomy_only=True)
    seed_module.seed()
    assert db_session.scalar(select(func.count()).select_from(Job)) == len(expected["jobs"])
