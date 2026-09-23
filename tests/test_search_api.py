from app.api.search import get_embedding_provider, get_job_reranker, get_vector_client
from app.retrieval.reranker import RerankerServiceError
from tests.factories import add_job


def _override_search_dependencies(client, fake_embedder, qdrant_client) -> None:
    client.app.dependency_overrides[get_embedding_provider] = lambda: fake_embedder
    client.app.dependency_overrides[get_vector_client] = lambda: qdrant_client


def _override_reranker(client, fake_reranker) -> None:
    client.app.dependency_overrides[get_job_reranker] = lambda: fake_reranker


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


def test_hybrid_search_api_fuses_keyword_and_dense(
    client, db_session, fake_embedder, qdrant_client
) -> None:
    _override_search_dependencies(client, fake_embedder, qdrant_client)
    add_job(
        db_session,
        title="Computer Vision Intern",
        description="Analyze images and video.",
        company_name="Vision",
        source_url="https://example.com/api-hybrid-vision",
    )
    add_job(
        db_session,
        title="Backend Engineer",
        description="Build APIs.",
        company_name="Backend",
        source_url="https://example.com/api-hybrid-backend",
    )
    assert client.post("/search/index", json={}).status_code == 200

    response = client.post(
        "/search/hybrid", json={"query": "image recognition internship", "limit": 2}
    )

    assert response.status_code == 200
    assert response.json()["method"] == "hybrid"
    assert len({item["job_id"] for item in response.json()["results"]}) == 2


def test_reranked_search_api_uses_fake_reranker(
    client, db_session, fake_embedder, qdrant_client, fake_reranker
) -> None:
    _override_search_dependencies(client, fake_embedder, qdrant_client)
    _override_reranker(client, fake_reranker)
    add_job(
        db_session,
        title="Computer Vision Intern",
        description="Analyze images.",
        company_name="Vision",
        source_url="https://example.com/api-reranked-vision",
    )
    add_job(
        db_session,
        title="Backend Engineer",
        description="Build APIs.",
        company_name="Backend",
        source_url="https://example.com/api-reranked-backend",
    )
    assert client.post("/search/index", json={}).status_code == 200

    response = client.post("/search/reranked", json={"query": "image recognition", "limit": 2})

    assert response.status_code == 200
    assert response.json()["method"] == "reranked"
    assert len(response.json()["results"]) == 2


def test_reranked_search_returns_503_when_provider_fails(
    client, db_session, fake_embedder, qdrant_client
) -> None:
    class FailingReranker:
        def rerank(self, query, candidates):
            del query, candidates
            raise RerankerServiceError("reranker unavailable")

    _override_search_dependencies(client, fake_embedder, qdrant_client)
    _override_reranker(client, FailingReranker())
    add_job(
        db_session,
        title="Computer Vision Intern",
        description="Analyze images.",
        company_name="Vision",
        source_url="https://example.com/api-reranker-failure",
    )
    assert client.post("/search/index", json={}).status_code == 200

    response = client.post("/search/reranked", json={"query": "image recognition", "limit": 1})

    assert response.status_code == 503
    assert response.json() == {"detail": "reranker unavailable"}
