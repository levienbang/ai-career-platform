import sys
from dataclasses import replace

import pytest

from app.config import Settings
from app.retrieval import reranker as reranker_module
from app.retrieval.reranker import (
    CrossEncoderJobReranker,
    LangChainJobReranker,
    RerankerConfigurationError,
    RerankerResponseError,
    RerankerServiceError,
    build_reranker,
)
from app.retrieval.types import SearchHit


def _hit(job_id: int, text: str) -> SearchHit:
    return SearchHit(
        job_id=job_id,
        score=0.5,
        title=f"Job {job_id}",
        company=None,
        location=None,
        employment_type=None,
        experience_years_min=None,
        required_skills=[],
        preferred_skills=[],
        document_text=text,
    )


class KeywordCountModel:
    """Stands in for CrossEncoder.predict: score = query words found in the document."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs, *, batch_size, show_progress_bar):
        del batch_size, show_progress_bar
        self.calls.append(list(pairs))
        return [sum(word in doc for word in query.split()) for query, doc in pairs]


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_cross_encoder_orders_by_score_and_keeps_every_candidate() -> None:
    model = KeywordCountModel()
    candidates = [_hit(1, "kotlin android"), _hit(2, "nlp chatbot llm"), _hit(3, "chatbot")]

    ranked = CrossEncoderJobReranker(_settings(), model=model).rerank("nlp chatbot", candidates)

    assert [hit.job_id for hit in ranked] == [2, 3, 1]
    assert [hit.score for hit in ranked] == [2.0, 1.0, 0.0]
    assert model.calls == [[("nlp chatbot", hit.document_text) for hit in candidates]]


def test_cross_encoder_breaks_ties_by_original_rank() -> None:
    candidates = [_hit(7, "same"), _hit(3, "same")]

    ranked = CrossEncoderJobReranker(_settings(), model=KeywordCountModel()).rerank(
        "same", candidates
    )

    assert [hit.job_id for hit in ranked] == [7, 3]


def test_cross_encoder_falls_back_to_title_without_document_text() -> None:
    model = KeywordCountModel()
    candidate = replace(_hit(1, ""), document_text=None)

    CrossEncoderJobReranker(_settings(), model=model).rerank("query", [candidate])

    assert model.calls == [[("query", "Job 1")]]


def test_cross_encoder_skips_model_for_empty_candidates() -> None:
    model = KeywordCountModel()

    assert CrossEncoderJobReranker(_settings(), model=model).rerank("q", []) == []
    assert model.calls == []


def test_cross_encoder_wraps_model_failures() -> None:
    class BrokenModel:
        def predict(self, pairs, **kwargs):
            raise RuntimeError("out of memory")

    with pytest.raises(RerankerServiceError):
        CrossEncoderJobReranker(_settings(), model=BrokenModel()).rerank("q", [_hit(1, "x")])


def test_cross_encoder_rejects_wrong_score_count() -> None:
    class ShortModel:
        def predict(self, pairs, **kwargs):
            return [1.0]

    with pytest.raises(RerankerResponseError):
        CrossEncoderJobReranker(_settings(), model=ShortModel()).rerank(
            "q", [_hit(1, "a"), _hit(2, "b")]
        )


def test_build_reranker_selects_provider(monkeypatch) -> None:
    monkeypatch.setattr(reranker_module, "load_cross_encoder", lambda *args: KeywordCountModel())

    assert isinstance(
        build_reranker(_settings(reranker_provider="cross_encoder")), CrossEncoderJobReranker
    )
    assert isinstance(
        build_reranker(_settings(reranker_provider="llm", deepseek_api_key="key")),
        LangChainJobReranker,
    )


def test_missing_rerank_extra_is_a_configuration_error(monkeypatch) -> None:
    reranker_module.load_cross_encoder.cache_clear()
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(RerankerConfigurationError, match="--extra rerank"):
        reranker_module.load_cross_encoder("BAAI/bge-reranker-v2-m3", 512, None)
