import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import Skill, SkillAlias
from app.db.session import get_db
from app.main import app
from app.schemas.ingestion import JobExtraction, RawJobRecord


class FakeJobExtractor:
    @staticmethod
    def _skills(value: list[str] | str | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [item.strip() for item in value.replace(",", ";").split(";") if item.strip()]

    def extract(self, record: RawJobRecord) -> JobExtraction:
        return JobExtraction(
            title=record.title,
            company=record.company,
            location=record.location,
            employment_type=record.employment_type,
            experience_years_min=record.experience_years_min,
            required_skills=self._skills(record.required_skills),
            preferred_skills=self._skills(record.preferred_skills),
            description=record.description,
            source_url=record.source_url,
        )


@pytest.fixture
def db_session() -> Session:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    with session_factory() as session:
        yield session
    Base.metadata.drop_all(engine)


@pytest.fixture
def client(db_session: Session) -> TestClient:
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def fake_extractor() -> FakeJobExtractor:
    return FakeJobExtractor()


@pytest.fixture
def taxonomy(db_session: Session) -> dict[str, Skill]:
    definitions = {
        "Python": ["python"],
        "Docker": ["docker"],
        "FastAPI": ["fast api", "fastapi"],
        "PostgreSQL": ["postgres", "postgresql", "postgre sql"],
        "Computer Vision": ["computer vision", "cv"],
    }
    skills: dict[str, Skill] = {}
    for name, aliases in definitions.items():
        skill = Skill(canonical_name=name)
        skill.aliases = [SkillAlias(alias=alias) for alias in aliases]
        db_session.add(skill)
        skills[name] = skill
    db_session.commit()
    return skills
