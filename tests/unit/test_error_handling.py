import logging
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.ingestion.extractor import ExtractionError, LangChainJobExtractor
from app.retrieval.dense import DenseSearchService
from app.retrieval.qdrant import VectorStoreUnavailableError
from app.schemas.ingestion import RawJobRecord


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


def test_collection_info_failure_is_unavailable(fake_embedder):
    client = SimpleNamespace(
        collection_exists=lambda _name: True,
        get_collection=lambda _name: (_ for _ in ()).throw(ConnectionError("offline")),
    )
    with pytest.raises(VectorStoreUnavailableError, match="Qdrant is unavailable"):
        DenseSearchService(client, fake_embedder, "jobs").search("Python")


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
