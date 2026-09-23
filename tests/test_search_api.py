from app.api.search import get_embedding_provider, get_vector_client
from tests.factories import add_job


def _override_search_dependencies(client, fake_embedder, qdrant_client) -> None:
    client.app.dependency_overrides[get_embedding_provider] = lambda: fake_embedder
    client.app.dependency_overrides[get_vector_client] = lambda: qdrant_client


def test_keyword_search_api_returns_ranked_jobs(client, db_session) -> None:
    job = add_job(
        db_session,
        title="Computer Vision Intern",
        description="Analyze images.",
        company_name="Vision",
        source_url="https://example.com/api-keyword",
    )

    response = client.post("/search/keyword", json={"query": "computer vision images", "limit": 5})

    assert response.status_code == 200
    assert response.json()["method"] == "keyword"
    assert response.json()["results"][0]["job_id"] == job.id


def test_index_and_dense_search_api(client, db_session, fake_embedder, qdrant_client) -> None:
    _override_search_dependencies(client, fake_embedder, qdrant_client)
    job = add_job(
        db_session,
        title="Computer Vision Intern",
        description="Analyze images and video.",
        company_name="Vision",
        source_url="https://example.com/api-dense",
    )

    indexed = client.post("/search/index", json={})
    searched = client.post("/search/semantic", json={"query": "image recognition", "limit": 1})

    assert indexed.status_code == 200
    assert indexed.json()["indexed"] == 1
    assert searched.status_code == 200
    assert searched.json()["method"] == "dense"
    assert searched.json()["results"][0]["job_id"] == job.id


def test_index_unknown_job_returns_404(client, fake_embedder, qdrant_client) -> None:
    _override_search_dependencies(client, fake_embedder, qdrant_client)

    response = client.post("/search/index", json={"job_id": 999})

    assert response.status_code == 404
    assert response.json() == {"detail": "Job not found"}


def test_dense_search_requires_an_index(client, fake_embedder, qdrant_client) -> None:
    _override_search_dependencies(client, fake_embedder, qdrant_client)

    response = client.post("/search/semantic", json={"query": "python"})

    assert response.status_code == 409
    assert "index jobs first" in response.json()["detail"]


def test_index_returns_503_when_qdrant_is_unavailable(client, fake_embedder) -> None:
    class UnavailableQdrant:
        def collection_exists(self, _collection_name):
            raise ConnectionError("offline")

    client.app.dependency_overrides[get_embedding_provider] = lambda: fake_embedder
    client.app.dependency_overrides[get_vector_client] = lambda: UnavailableQdrant()

    response = client.post("/search/index", json={})

    assert response.status_code == 503
    assert response.json() == {"detail": "Qdrant is unavailable"}


def test_index_twice_via_api_keeps_one_point(
    client, db_session, fake_embedder, qdrant_client
) -> None:
    _override_search_dependencies(client, fake_embedder, qdrant_client)
    add_job(
        db_session,
        title="Backend Engineer",
        description="Build APIs.",
        company_name="Backend",
        source_url="https://example.com/api-idempotent",
    )

    assert client.post("/search/index", json={}).status_code == 200
    assert client.post("/search/index", json={}).status_code == 200

    assert qdrant_client.count("jobs", exact=True).count == 1
