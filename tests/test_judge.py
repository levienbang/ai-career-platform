import json
import math

import pytest

from app.config import Settings
from evaluation.evaluate_judge import build_pool, build_report
from evaluation.judge import (
    PROMPT_VERSION,
    JudgedJob,
    JudgeResponseError,
    JudgmentCache,
    LangChainRelevanceJudge,
    build_human_review,
    cohen_kappa,
    human_agreement,
    human_review_started,
    judge_pool,
    judged_precision_at_k,
    ndcg_at_k,
    planned_calls,
    sample_pairs,
    split_batches,
    validate_judgments,
)


class FakeJudge:
    model_name = "fake-judge"

    def __init__(self, grades: dict[int, int], *, drop: set[int] = frozenset()) -> None:
        self.grades = grades
        self.drop = set(drop)
        self.calls: list[list[int]] = []

    def grade(self, query, documents):
        ids = [doc.job_id for doc in documents]
        self.calls.append(ids)
        return [
            JudgedJob(job_id=job_id, grade=self.grades[job_id], reason="fake")
            for job_id in ids
            if job_id not in self.drop
        ]


class FakeRunnable:
    def __init__(self, output) -> None:
        self.output = output
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self.output


def test_validate_judgments_rejects_missing_duplicate_and_extra_ids() -> None:
    items = [JudgedJob(job_id=1, grade=2), JudgedJob(job_id=2, grade=0)]
    validate_judgments([2, 1], items)
    with pytest.raises(JudgeResponseError, match="missing=\\[3\\]"):
        validate_judgments([1, 2, 3], items)
    with pytest.raises(JudgeResponseError, match="extra=\\[2\\]"):
        validate_judgments([1], items)
    with pytest.raises(JudgeResponseError, match="duplicate"):
        validate_judgments([1], [JudgedJob(job_id=1, grade=1), JudgedJob(job_id=1, grade=2)])


def test_langchain_judge_validates_returned_ids_and_marks_content_untrusted() -> None:
    from evaluation.judge import JudgeDocument

    documents = [JudgeDocument(1, "Title: AI Engineer"), JudgeDocument(2, "Title: Chef")]
    runnable = FakeRunnable({"items": [{"job_id": 1, "grade": 2, "reason": "match"}]})
    judge = LangChainRelevanceJudge(Settings(), runnable=runnable)
    with pytest.raises(JudgeResponseError, match="missing=\\[2\\]"):
        judge.grade("AI engineer", documents)
    prompt_text = "\n".join(message.content for message in runnable.messages.to_messages())
    assert "untrusted" in prompt_text
    assert "Title: Chef" in prompt_text

    runnable.output = {"items": [{"job_id": 2, "grade": 0}, {"job_id": 1, "grade": 2}]}
    assert {item.job_id: item.grade for item in judge.grade("AI engineer", documents)} == {
        1: 2,
        2: 0,
    }

    runnable.output = {"items": [{"job_id": 1, "grade": 3}, {"job_id": 2, "grade": 0}]}
    with pytest.raises(JudgeResponseError):
        judge.grade("AI engineer", documents)


def test_split_batches_is_even_and_minimal() -> None:
    assert split_batches(list(range(17)), 8) == [
        list(range(6)),
        list(range(6, 12)),
        list(range(12, 17)),
    ]
    assert split_batches([], 8) == []
    with pytest.raises(ValueError):
        split_batches([1], 0)


def test_judge_pool_caches_and_reuses_judgments(tmp_path) -> None:
    cache_file = tmp_path / "cache.json"
    pool = {"q1": [1, 2, 3], "q2": [3]}
    documents = {1: "doc one", 2: "doc two", 3: "doc three"}
    judge = FakeJudge({1: 2, 2: 1, 3: 0})
    cache = JudgmentCache(cache_file)

    assert planned_calls(cache, pool, batch_size=2) == 3
    stats = judge_pool(judge, cache, pool, documents, batch_size=2)
    assert stats.calls == 3 and stats.judged_pairs == 4 and not stats.failed_batches

    reloaded = JudgmentCache(cache_file)
    assert reloaded.grade("q1", 2) == 1
    assert reloaded.get("q1", 2)["prompt_version"] == PROMPT_VERSION
    assert planned_calls(reloaded, pool, batch_size=2, documents=documents) == 0
    rerun = FakeJudge({1: 2, 2: 1, 3: 0})
    assert judge_pool(rerun, reloaded, pool, documents, batch_size=2).calls == 0
    assert rerun.calls == []

    # A changed search document invalidates only that pair.
    changed = {**documents, 2: "doc two, updated skills"}
    judge_pool(rerun, reloaded, pool, changed, batch_size=2)
    assert rerun.calls == [[2]]


def test_judge_pool_records_failed_batches_without_caching_them(tmp_path) -> None:
    cache = JudgmentCache(tmp_path / "cache.json")
    judge = FakeJudge({1: 2, 2: 1}, drop={2})
    stats = judge_pool(judge, cache, {"q": [1, 2]}, {1: "a", 2: "b"}, batch_size=5)
    assert len(stats.failed_batches) == 1
    assert cache.grade("q", 1) is None
    assert not (tmp_path / "cache.json").exists()


def test_cache_ignores_judgments_from_another_prompt_version(tmp_path) -> None:
    path = tmp_path / "cache.json"
    path.write_text(
        json.dumps({"judgments": {"q": {"1": {"grade": 2, "prompt_version": "old"}}}}),
        encoding="utf-8",
    )
    assert JudgmentCache(path).grade("q", 1) is None


def test_cohen_kappa_and_ndcg_math() -> None:
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    assert cohen_kappa([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)
    assert cohen_kappa([1, 0], [0, 1]) == pytest.approx(-1.0)
    assert cohen_kappa([1, 1], [1, 1]) == 1.0
    assert cohen_kappa([], []) is None

    ideal = 3 + 1 / math.log2(3)
    assert ndcg_at_k([2, 0, 1], [2, 1, 0], 10) == pytest.approx(3.5 / ideal)
    assert ndcg_at_k([2, 1, 0], [0, 1, 2], 10) == pytest.approx(1.0)
    assert ndcg_at_k([0, 0], [0, 0], 10) == 0.0
    assert judged_precision_at_k([2, 1, 0, 2, 2, 2], 5) == pytest.approx(0.6)
    assert judged_precision_at_k([2, 1, 0, 2, 2], 5, threshold=1) == pytest.approx(0.8)


def _tiny_fixture(tmp_path):
    cases = {
        "retrieval": [{"id": "r1", "query": "ai", "relevant_job_ids": [1, 2]}],
        "retrieval_paraphrase": [{"id": "p1", "query": "máy học", "relevant_job_ids": [1]}],
    }
    results = {
        "retrieval": {
            "dense": {"precision_at_5": 0.4},
            "keyword": {"precision_at_5": 0.2},
            "cases": [
                {
                    "id": "r1",
                    "query": "ai",
                    "dense": {"top10": [1, 2, 3, 4, 5]},
                    "keyword": {"top10": [5, 4, 3, 2, 950]},
                }
            ],
        },
        "retrieval_paraphrase": {
            "dense": {"precision_at_5": 1.0},
            "cases": [{"id": "p1", "query": "máy học", "dense": {"top10": [1, 9]}}],
        },
    }
    cache = JudgmentCache(tmp_path / "cache.json")
    grades = {1: 2, 2: 2, 3: 1, 4: 0, 5: 0, 950: 2}
    for job_id, grade in grades.items():
        cache.put("ai", JudgedJob(job_id=job_id, grade=grade), doc_sha="x", model="fake")
    return cases, results, cache


def test_build_pool_takes_union_of_method_rankings(tmp_path) -> None:
    _, results, _ = _tiny_fixture(tmp_path)
    assert build_pool(results) == {"ai": [1, 2, 3, 4, 5, 950], "máy học": [1, 9]}


def test_build_report_scores_methods_and_silver_agreement(tmp_path) -> None:
    cases, results, cache = _tiny_fixture(tmp_path)
    report = build_report(cases, results, cache, labelled_max_job_id=941)

    dense = report["retrieval"]["dense"]
    assert dense["judged_precision_at_5"] == pytest.approx(0.4)
    assert dense["judged_precision_at_5_lenient"] == pytest.approx(0.6)
    ideal = 3 + 3 / math.log2(3) + 3 / 2 + 1 / math.log2(5)
    assert dense["ndcg_at_10"] == pytest.approx(round((3 + 3 / math.log2(3) + 1 / 2) / ideal, 3))
    assert dense["silver_precision_at_5"] == 0.4
    assert report["retrieval"]["keyword"]["judged_precision_at_5"] == pytest.approx(0.4)
    assert report["retrieval_paraphrase"]["incomplete_cases"] == ["p1"]

    agreement = report["silver_agreement"]
    assert agreement["unlabelled_pairs_skipped"] == 1
    strict = agreement["relevant_if_grade_ge_2"]
    assert strict["pairs"] == 5 and strict["raw_agreement"] == 1.0 and strict["cohen_kappa"] == 1.0
    lenient = agreement["relevant_if_grade_ge_1"]
    assert lenient["judge_relevant_silver_not"] == 1
    assert lenient["raw_agreement"] == pytest.approx(0.8)


def test_human_review_sheet_is_blind_and_agreement_uses_filled_pairs(tmp_path) -> None:
    _, _, cache = _tiny_fixture(tmp_path)
    pairs = [("ai", job_id) for job_id in (1, 2, 3, 4)]
    assert sample_pairs(pairs, 3, seed=7) == sample_pairs(list(reversed(pairs)), 3, seed=7)
    review = build_human_review(pairs, {i: f"doc {i}" for i in range(1, 5)}, {"ai": "r1"}, seed=7)
    assert all(pair["human_grade"] is None and "grade" not in pair for pair in review["pairs"])

    path = tmp_path / "review.json"
    path.write_text(json.dumps(review), encoding="utf-8")
    assert human_review_started(path) is False
    for pair, human in zip(review["pairs"], (2, 1, 1, None), strict=True):
        pair["human_grade"] = human
    path.write_text(json.dumps(review), encoding="utf-8")
    assert human_review_started(path) is True

    result = human_agreement(review, cache)
    assert result["graded_pairs"] == 3
    assert result["exact_0_1_2"]["raw_agreement"] == 0.667
    assert result["binary_grade_ge_1"]["raw_agreement"] == 1.0

    review["pairs"][3]["human_grade"] = 5
    with pytest.raises(ValueError):
        human_agreement(review, cache)
