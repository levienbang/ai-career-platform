from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import Skill, SkillAlias
from app.db.session import get_db
from app.main import app
from app.schemas.ingestion import JobExtraction, RawJobRecord


class FakeEmbeddingProvider:
    dimensions = 4

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.casefold()
        vector = [
            float(sum(term in lowered for term in ("vision", "image", "video"))),
            float(sum(term in lowered for term in ("backend", "api", "fastapi"))),
            float(sum(term in lowered for term in ("docker", "deploy", "container"))),
            float(sum(term in lowered for term in ("postgres", "data", "database"))),
        ]
        return vector if any(vector) else [0.5, 0.5, 0.5, 0.5]

    def embed_documents(self, texts: list[str], titles: list[str]) -> list[list[float]]:
        assert len(texts) == len(titles)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeJobReranker:
    def rerank(self, query: str, candidates):
        del query
        ordered = list(reversed(candidates))
        return [
            replace(candidate, score=1 - rank / max(len(ordered), 1))
            for rank, candidate in enumerate(ordered)
        ]


class FakeJobExtractor:
    def extract_many(self, records: list[RawJobRecord]) -> list[JobExtraction | None]:
        from pydantic import ValidationError

        outputs = []
        for record in records:
            try:
                outputs.append(self.extract(record))
            except ValidationError:
                outputs.append(None)
        return outputs

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
def fake_embedder() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def fake_reranker() -> FakeJobReranker:
    return FakeJobReranker()


@pytest.fixture
def qdrant_client() -> QdrantClient:
    return QdrantClient(":memory:")


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
