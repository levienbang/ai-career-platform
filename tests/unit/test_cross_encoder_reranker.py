import sys
from dataclasses import replace

import pytest

from app.config import Settings
from app.retrieval import reranker as reranker_module
from app.retrieval.reranker import (
    CrossEncoderJobReranker,
    FallbackJobReranker,
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
        build_reranker(_settings(reranker_provider="cross_encoder", reranker_fallback="none")),
        CrossEncoderJobReranker,
    )
    assert isinstance(
        build_reranker(
            _settings(reranker_provider="llm", deepseek_api_key="key", reranker_fallback="none")
        ),
        LangChainJobReranker,
    )


def test_build_reranker_wraps_the_provider_with_a_fallback(monkeypatch) -> None:
    monkeypatch.setattr(reranker_module, "load_cross_encoder", lambda *args: KeywordCountModel())

    llm = build_reranker(_settings(reranker_provider="llm", deepseek_api_key="key"))
    local = build_reranker(_settings(reranker_provider="cross_encoder"))

    assert isinstance(llm, FallbackJobReranker)
    assert isinstance(llm.primary, LangChainJobReranker)
    assert llm.backup_name == "cross_encoder"
    # A local primary has no second local model to fall back to: only hybrid order.
    assert isinstance(local, FallbackJobReranker)
    assert local.backup_name == "none"


def test_missing_deepseek_key_still_fails_loudly() -> None:
    # Fallback covers request-time failures, not a misconfigured deployment.
    with pytest.raises(RerankerConfigurationError):
        build_reranker(_settings(reranker_provider="llm", deepseek_api_key=None))


class FailingReranker:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def rerank(self, query, candidates):
        self.calls += 1
        raise self.error


class ReversingReranker:
    def rerank(self, query, candidates):
        return [replace(hit, score=1.0 - rank) for rank, hit in enumerate(reversed(candidates))]


class DroppingReranker:
    """Mimics the LLM omitting a candidate without raising."""

    def rerank(self, query, candidates):
        return candidates[1:]


CANDIDATES = [_hit(1, "a"), _hit(2, "b"), _hit(3, "c")]


def test_working_primary_is_used_without_touching_the_backup() -> None:
    def backup():
        raise AssertionError("backup must not be built")

    reranker = FallbackJobReranker(ReversingReranker(), "llm", backup=backup)

    ranked = reranker.rerank("q", CANDIDATES)

    assert [hit.job_id for hit in ranked] == [3, 2, 1]
    assert (reranker.used, reranker.fell_back) == ("llm", False)


@pytest.mark.parametrize(
    "primary",
    [
        FailingReranker(RerankerServiceError("timeout")),
        FailingReranker(RerankerResponseError("duplicate job IDs")),
        DroppingReranker(),
    ],
    ids=["service-error", "bad-ids", "missing-id-without-error"],
)
def test_failed_primary_falls_back_to_the_local_model(primary) -> None:
    model = KeywordCountModel()
    reranker = FallbackJobReranker(
        primary,
        "llm",
        backup=lambda: CrossEncoderJobReranker(_settings(), model=model),
        backup_name="cross_encoder",
    )

    ranked = reranker.rerank("b", CANDIDATES)

    assert [hit.job_id for hit in ranked] == [2, 1, 3]
    assert (reranker.used, reranker.fell_back) == ("cross_encoder", True)


def test_unavailable_backup_keeps_hybrid_order() -> None:
    def backup():
        raise RerankerConfigurationError("uv sync --extra rerank")

    reranker = FallbackJobReranker(
        FailingReranker(RerankerServiceError("timeout")), "llm", backup=backup
    )

    ranked = reranker.rerank("q", CANDIDATES)

    assert ranked == CANDIDATES
    assert (reranker.used, reranker.fell_back) == ("none", True)


def test_failing_backup_keeps_hybrid_order() -> None:
    reranker = FallbackJobReranker(
        FailingReranker(RerankerResponseError("bad ids")),
        "llm",
        backup=lambda: FailingReranker(RerankerServiceError("mps crashed")),
        backup_name="cross_encoder",
    )

    assert reranker.rerank("q", CANDIDATES) == CANDIDATES
    assert (reranker.used, reranker.fell_back) == ("none", True)


def test_backup_is_built_once_and_status_resets_per_request() -> None:
    builds = []
    primary = FailingReranker(RerankerServiceError("timeout"))

    def backup():
        builds.append(1)
        return ReversingReranker()

    reranker = FallbackJobReranker(primary, "llm", backup=backup, backup_name="cross_encoder")
    reranker.rerank("q", CANDIDATES)
    reranker.rerank("q", CANDIDATES)
    assert builds == [1]

    reranker.primary = ReversingReranker()
    reranker.rerank("q", CANDIDATES)
    assert (reranker.used, reranker.fell_back) == ("llm", False)


def test_unexpected_errors_are_not_swallowed() -> None:
    # Only known reranker failures fall back; a programming error must surface.
    reranker = FallbackJobReranker(FailingReranker(KeyError("bug")), "llm")

    with pytest.raises(KeyError):
        reranker.rerank("q", CANDIDATES)


def test_fallback_log_never_contains_the_query(caplog) -> None:
    reranker = FallbackJobReranker(FailingReranker(RerankerServiceError("timeout")), "llm")

    reranker.rerank("lương 50 triệu ở MoMo", CANDIDATES)

    assert "RerankerServiceError" in caplog.text
    assert "MoMo" not in caplog.text


def test_missing_rerank_extra_is_a_configuration_error(monkeypatch) -> None:
    reranker_module._load_cross_encoder.cache_clear()
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(RerankerConfigurationError, match="--extra rerank"):
        reranker_module.load_cross_encoder("BAAI/bge-reranker-v2-m3", 512, None)
    reranker_module._load_cross_encoder.cache_clear()


def test_warmup_is_skipped_for_the_llm_reranker() -> None:
    warmup = reranker_module.CrossEncoderWarmup()

    warmup.start(_settings(reranker_provider="llm"))

    assert warmup.wait(1) == "not_used"


def test_warmup_loads_and_runs_one_pair_in_the_background(monkeypatch) -> None:
    model = KeywordCountModel()
    loads: list[tuple] = []

    def fake_load(*args):
        loads.append(args)
        return model

    monkeypatch.setattr(reranker_module, "load_cross_encoder", fake_load)
    warmup = reranker_module.CrossEncoderWarmup()

    warmup.start(_settings(reranker_provider="cross_encoder", reranker_device="cpu"))

    assert warmup.wait(5) == "loaded"
    assert loads == [("BAAI/bge-reranker-v2-m3", 512, "cpu")]
    assert model.calls == [[("warm up", "warm up")]]


def test_failed_warmup_is_reported_not_raised(monkeypatch) -> None:
    def broken_load(*args):
        raise RerankerConfigurationError("model missing")

    monkeypatch.setattr(reranker_module, "load_cross_encoder", broken_load)
    warmup = reranker_module.CrossEncoderWarmup()

    warmup.start(_settings(reranker_provider="cross_encoder"))

    assert warmup.wait(5) == "failed"


def test_concurrent_loads_share_one_model(monkeypatch) -> None:
    import threading
    import time

    created: list[object] = []

    class SlowCrossEncoder:
        def __init__(self, *args, **kwargs) -> None:
            time.sleep(0.05)
            created.append(self)

    fake_module = type(sys)("sentence_transformers")
    fake_module.CrossEncoder = SlowCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    reranker_module._load_cross_encoder.cache_clear()
    results: list[object] = []
    threads = [
        threading.Thread(
            target=lambda: results.append(reranker_module.load_cross_encoder("m", 512, None))
        )
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    reranker_module._load_cross_encoder.cache_clear()

    assert len(created) == 1
    assert all(result is created[0] for result in results)


def test_fallbacks_are_counted_for_admin_stats() -> None:
    reranker_module.reranker_fallbacks.reset()
    reranker = FallbackJobReranker(
        FailingReranker(RerankerServiceError("timeout")),
        "llm",
        backup=ReversingReranker,
        backup_name="cross_encoder",
    )

    reranker.rerank("q", CANDIDATES)
    FallbackJobReranker(ReversingReranker(), "llm").rerank("q", CANDIDATES)

    assert reranker_module.reranker_fallbacks.snapshot() == {"cross_encoder": 1}
    reranker_module.reranker_fallbacks.reset()
