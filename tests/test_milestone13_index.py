from types import SimpleNamespace

import pytest
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.db.models import Job, JobSkill, RequirementType, Skill
from app.retrieval.embeddings import EmbeddingServiceError
from app.retrieval.qdrant import QdrantJobIndex
from app.services.skill_taxonomy import load_skill_aliases
from scripts import merge_skills, seed


def test_scripts_find_taxonomy_from_other_cwd(db_session, monkeypatch, tmp_path):
    factory = sessionmaker(bind=db_session.get_bind())
    monkeypatch.setattr(seed, "SessionLocal", factory)
    monkeypatch.setattr(merge_skills, "SessionLocal", factory)
    monkeypatch.chdir(tmp_path)
    seed.seed(taxonomy_only=True)
    assert merge_skills.merge_skills() == 0
    with pytest.raises(FileNotFoundError, match="SKILL_DATA_DIR") as caught:
        load_skill_aliases(tmp_path / "missing.json")
    assert str(tmp_path / "missing.json") in str(caught.value)
    assert Settings(_env_file=None, skill_data_dir="").skill_data_dir.name == "data"


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def __call__(self):
        return self.now


class Error(RuntimeError):
    def __init__(self, status, retry_after=None):
        self.status_code = status
        self.response = SimpleNamespace(headers={"retry-after": retry_after})
        super().__init__("provider failure")


class Embedder:
    dimensions = 4

    def __init__(self, client, errors=None):
        self.client = client
        self.calls = []
        self.errors = errors or {}

    def embed_documents(self, texts, titles):
        self.calls.append((len(texts), self.client.count("jobs", exact=True).count))
        if len(self.calls) in self.errors:
            raise self.errors[len(self.calls)]
        return [[1, 0, 0, 0] for _ in texts]


def make_jobs(session, count=45):
    jobs = [
        Job(title=f"Job {n}", description=f"Work {n}", content_hash=f"{n:064x}")
        for n in range(count)
    ]
    session.add_all(jobs)
    session.commit()
    return jobs


def index(client, embedder, clock, **settings):
    return QdrantJobIndex(
        client,
        embedder,
        "jobs",
        clock=clock,
        sleep=clock.sleep,
        settings=Settings(_env_file=None, **settings),
    )


def test_batches_retry_and_incremental_index(db_session, qdrant_client):
    jobs = make_jobs(db_session)
    clock = Clock()
    embedder = Embedder(qdrant_client, {2: Error(429, "7")})
    service = index(qdrant_client, embedder, clock)
    result = service.index_all(db_session)
    assert result.indexed == 45 and result.skipped == 0
    assert embedder.calls == [(20, 0), (20, 20), (20, 20), (5, 40)]
    assert clock.sleeps == [7]
    assert service.index_all(db_session).skipped == 45
    assert len(embedder.calls) == 4
    skill = Skill(canonical_name="Python")
    jobs[0].skills.append(JobSkill(skill=skill, requirement_type=RequirementType.REQUIRED))
    db_session.commit()
    changed = service.index_all(db_session)
    assert changed.indexed == 1 and changed.skipped == 44
    assert service.index_all(db_session, full=True).indexed == 45
    db_session.delete(jobs[-1])
    db_session.commit()
    result = service.index_all(db_session)
    assert result.deleted == 1 and result.skipped == 44 and result.indexed == 0


@pytest.mark.parametrize("status,retries,calls,sleeps", [(400, 5, 2, []), (503, 1, 3, [20])])
def test_partial_failure_and_resume(db_session, qdrant_client, status, retries, calls, sleeps):
    make_jobs(db_session)
    clock = Clock()
    embedder = Embedder(qdrant_client, {2: Error(status), 3: Error(status)})
    service = index(qdrant_client, embedder, clock, embedding_max_retries=retries)
    with pytest.raises(EmbeddingServiceError, match="indexing 20/45"):
        service.index_all(db_session)
    assert len(embedder.calls) == calls and clock.sleeps == sleeps
    assert qdrant_client.count("jobs", exact=True).count == 20
    embedder.errors.clear()
    assert service.index_all(db_session).indexed == 25


def test_rpm_throttle_and_wrapped_errors(db_session, qdrant_client):
    make_jobs(db_session)
    clock = Clock()
    wrapped = EmbeddingServiceError("wrapped")
    wrapped.__cause__ = Error(500)
    embedder = Embedder(qdrant_client, {2: wrapped})
    service = index(qdrant_client, embedder, clock, embedding_requests_per_minute=2)
    assert service.index_all(db_session).indexed == 45
    assert clock.sleeps == [30, 20, 10, 30]
