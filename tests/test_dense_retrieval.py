import pytest

from app.retrieval.dense import DenseSearchService
from app.retrieval.qdrant import IndexedJobNotFoundError, QdrantJobIndex
from tests.factories import add_job


def test_index_twice_upserts_without_duplicate_points(
    db_session, qdrant_client, fake_embedder
) -> None:
    first = add_job(
        db_session,
        title="Computer Vision Intern",
        description="Analyze images and video.",
        company_name="Vision",
        source_url="https://example.com/dense-vision",
        required_skills=("Computer Vision",),
    )
    add_job(
        db_session,
        title="Backend Engineer",
        description="Build Python APIs.",
        company_name="Backend",
        source_url="https://example.com/dense-backend",
        required_skills=("FastAPI",),
    )
    index = QdrantJobIndex(qdrant_client, fake_embedder, "test_jobs")

    first_run = index.index_all(db_session)
    second_run = index.index_all(db_session)

    assert first_run.indexed == 2
    assert second_run.indexed == 0
    assert second_run.skipped == 2
    assert qdrant_client.count("test_jobs", exact=True).count == 2
    point = qdrant_client.retrieve("test_jobs", ids=[first.id], with_payload=True)[0]
    assert point.payload["title"] == "Computer Vision Intern"


def test_dense_search_uses_query_embedding(db_session, qdrant_client, fake_embedder) -> None:
    vision = add_job(
        db_session,
        title="Computer Vision Intern",
        description="Analyze images and video.",
        company_name="Vision",
        source_url="https://example.com/dense-search-vision",
    )
    add_job(
        db_session,
        title="Data Engineer",
        description="Maintain PostgreSQL data pipelines.",
        company_name="Data",
        source_url="https://example.com/dense-search-data",
    )
    index = QdrantJobIndex(qdrant_client, fake_embedder, "test_jobs")
    index.index_all(db_session)

    results = DenseSearchService(qdrant_client, fake_embedder, "test_jobs").search(
        "image recognition", limit=1
    )

    assert results[0].job_id == vision.id
    assert results[0].score > 0


def test_full_sync_removes_points_for_deleted_jobs(
    db_session, qdrant_client, fake_embedder
) -> None:
    job = add_job(
        db_session,
        title="Temporary Job",
        description="Build APIs.",
        company_name="Temporary",
        source_url="https://example.com/deleted-job",
    )
    index = QdrantJobIndex(qdrant_client, fake_embedder, "test_jobs")
    index.index_all(db_session)
    db_session.delete(job)
    db_session.commit()

    result = index.index_all(db_session)

    assert result.deleted == 1
    assert qdrant_client.count("test_jobs", exact=True).count == 0


def test_index_unknown_job_fails(db_session, qdrant_client, fake_embedder) -> None:
    index = QdrantJobIndex(qdrant_client, fake_embedder, "test_jobs")

    with pytest.raises(IndexedJobNotFoundError):
        index.index_job(db_session, 999)
