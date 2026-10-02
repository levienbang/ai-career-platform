import logging
from types import SimpleNamespace

import pytest
from qdrant_client import models

from app.api.cv import get_cv_match_service
from app.api.search import get_embedding_provider, get_job_reranker, get_vector_client
from app.config import Settings
from app.ingestion.extractor import ExtractionError, LangChainJobExtractor
from app.retrieval.dense import DenseSearchService
from app.retrieval.qdrant import SearchIndexNotReadyError, VectorStoreUnavailableError
from app.schemas.cv import CVExtraction
from app.schemas.ingestion import RawJobRecord
from app.services.cv_match import CVMatchService


def failing_extractor(monkeypatch, error, *, retries=1):
    def invoke(_messages):
        raise error

    monkeypatch.setattr(
        "app.ingestion.extractor.build_structured_chat_model",
        lambda *_a, **_kw: SimpleNamespace(runnable=SimpleNamespace(invoke=invoke)),
    )
    return LangChainJobExtractor(
        Settings(
            deepseek_api_key="fake",
            llm_max_retries=retries,
            llm_requests_per_minute=0,
            _env_file=None,
        )
    )


@pytest.mark.parametrize("single", [True, False])
def test_extraction_logs_redacted_bounded_errors_and_terminal_summary(monkeypatch, caplog, single):
    source = RawJobRecord(title="PRIVATE_TITLE", description="PRIVATE_JOB_CONTENT")
    error = RuntimeError("HTTP 400 invalid model sk-test-secret PRIVATE_JOB_CONTENT " + "x" * 500)
    extractor = failing_extractor(monkeypatch, error)
    with caplog.at_level(logging.WARNING, logger="app.ingestion.extractor"):
        if single:
            with pytest.raises(ExtractionError):
                extractor.extract(source)
        else:
            assert extractor.extract_many([source, source]) == [None, None]
    messages = [record.getMessage() for record in caplog.records]
    failures = [message for message in messages if "exception=" in message]
    assert len(failures) == (2 if single else 6)
    assert "1 records failed" in messages[-1] if single else "2 records failed" in messages[-1]
    assert all("exception=RuntimeError" in message for message in failures)
    assert all("attempt=" in message and "records=" in message for message in failures)
    assert all(len(message.split("message=", 1)[1]) <= 300 for message in failures)
    assert "HTTP 400 invalid model ***" in failures[0]
    assert all(
        word not in caplog.text for word in ["PRIVATE_TITLE", "PRIVATE_JOB_CONTENT", "sk-test"]
    )


def test_parser_error_logs_never_include_model_output(monkeypatch, caplog):
    from langchain_core.exceptions import OutputParserException

    extractor = failing_extractor(
        monkeypatch,
        OutputParserException("MODEL_OUTPUT_PRIVATE", llm_output="MODEL_OUTPUT_PRIVATE"),
        retries=0,
    )
    with caplog.at_level(logging.WARNING):
        assert extractor.extract_many([RawJobRecord(description="PRIVATE_JOB")]) == [None]
    assert "OutputParserException" in caplog.text
    assert "MODEL_OUTPUT_PRIVATE" not in caplog.text


def test_partial_batch_logs_failed_record_count(monkeypatch, caplog):
    model = SimpleNamespace(
        invoke=lambda _m: {"items": [{"record_index": 1, "title": "AI", "description": "Python"}]}
    )
    monkeypatch.setattr(
        "app.ingestion.extractor.build_structured_chat_model",
        lambda *_a, **_kw: SimpleNamespace(runnable=model),
    )
    extractor = LangChainJobExtractor(
        Settings(deepseek_api_key="fake", llm_requests_per_minute=0, _env_file=None)
    )
    with caplog.at_level(logging.WARNING):
        result = extractor.extract_many([RawJobRecord(description="Python")] * 2)
    assert result[0] is not None and result[1] is None
    assert "1 records failed after retries/split" in caplog.text


def empty_collection(qdrant_client, dimensions, name="jobs"):
    qdrant_client.create_collection(
        name, vectors_config=models.VectorParams(size=dimensions, distance=models.Distance.COSINE)
    )


def test_empty_dense_collection_errors_before_embedding(qdrant_client, fake_embedder, monkeypatch):
    empty_collection(qdrant_client, fake_embedder.dimensions)
    monkeypatch.setattr(fake_embedder, "embed_query", lambda _q: pytest.fail("must not embed"))
    with pytest.raises(SearchIndexNotReadyError, match="'jobs' is empty; index jobs first"):
        DenseSearchService(qdrant_client, fake_embedder, "jobs").search("Python")


def test_collection_info_failure_is_unavailable(fake_embedder):
    client = SimpleNamespace(
        collection_exists=lambda _name: True,
        get_collection=lambda _name: (_ for _ in ()).throw(ConnectionError("offline")),
    )
    with pytest.raises(VectorStoreUnavailableError, match="Qdrant is unavailable"):
        DenseSearchService(client, fake_embedder, "jobs").search("Python")


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


def test_provider_error_preserves_diagnostic_without_response_body(monkeypatch, caplog):
    error = RuntimeError('Error code: 400 - {"request": "PRIVATE_SOURCE"}')
    error.body = {
        "error": {"message": "Model Not Exist sk-fake-secret"},
        "request": "PRIVATE_SOURCE",
    }
    extractor = failing_extractor(monkeypatch, error, retries=0)
    with caplog.at_level(logging.WARNING):
        extractor.extract_many([RawJobRecord(description="PRIVATE_SOURCE")])
    assert "Error code: 400" in caplog.text and "Model Not Exist ***" in caplog.text
    assert "PRIVATE_SOURCE" not in caplog.text and "sk-fake-secret" not in caplog.text
