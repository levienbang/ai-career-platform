from types import SimpleNamespace

import pytest
from qdrant_client import models

from app.api.cv import get_cv_match_service
from app.api.search import get_embedding_provider, get_job_reranker, get_vector_client
from app.config import Settings
from app.cv.match import CVMatchService
from app.retrieval.dense import DenseSearchService
from app.retrieval.qdrant import SearchIndexNotReadyError
from app.schemas.cv import CVExtraction


def empty_collection(qdrant_client, dimensions, name="jobs"):
    qdrant_client.create_collection(
        name, vectors_config=models.VectorParams(size=dimensions, distance=models.Distance.COSINE)
    )


def test_empty_dense_collection_errors_before_embedding(qdrant_client, fake_embedder, monkeypatch):
    empty_collection(qdrant_client, fake_embedder.dimensions)
    monkeypatch.setattr(fake_embedder, "embed_query", lambda _q: pytest.fail("must not embed"))
    with pytest.raises(SearchIndexNotReadyError, match="'jobs' is empty; index jobs first"):
        DenseSearchService(qdrant_client, fake_embedder, "jobs").search("Python")


@pytest.mark.parametrize("endpoint", ["semantic", "hybrid", "reranked"])
def test_empty_dense_search_endpoints_return_503(
    endpoint, client, qdrant_client, fake_embedder, fake_reranker
):
    empty_collection(qdrant_client, fake_embedder.dimensions)
    client.app.dependency_overrides[get_embedding_provider] = lambda: fake_embedder
    client.app.dependency_overrides[get_vector_client] = lambda: qdrant_client
    client.app.dependency_overrides[get_job_reranker] = lambda: fake_reranker
    response = client.post(f"/search/{endpoint}", json={"query": "Python"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Qdrant collection 'jobs' is empty; index jobs first"


def test_cv_match_empty_index_returns_503(client, db_session, qdrant_client, fake_embedder):
    empty_collection(qdrant_client, fake_embedder.dimensions)
    service = CVMatchService(
        db_session,
        SimpleNamespace(load=lambda _c, **_kw: "Python"),
        SimpleNamespace(extract=lambda _t: CVExtraction(summary="Python")),
        DenseSearchService(qdrant_client, fake_embedder, "jobs"),
        Settings(_env_file=None),
    )
    client.app.dependency_overrides[get_cv_match_service] = lambda: service
    response = client.post("/cv/match", files={"file": ("cv.pdf", b"%PDF-test", "application/pdf")})
    assert response.status_code == 503
    assert response.json()["detail"] == "Qdrant collection 'jobs' is empty; index jobs first"
