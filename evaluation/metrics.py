"""Ranking and latency metrics shared by every evaluation script."""

import math
from collections.abc import Sequence


def _require_labels(relevant_ids: set[int]) -> None:
    if not relevant_ids:
        raise ValueError("Each evaluation case needs at least one relevant job")


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    _require_labels(relevant_ids)
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    return sum(job_id in relevant_ids for job_id in retrieved_ids[:k]) / k


def mrr_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    _require_labels(relevant_ids)
    for rank, job_id in enumerate(retrieved_ids[:k], start=1):
        if job_id in relevant_ids:
            return 1 / rank
    return 0.0


def judged_precision_at_k(grades: Sequence[int], k: int, *, threshold: int = 2) -> float:
    """Precision@k from 0/1/2 grades: a job counts when its grade reaches threshold."""
    return sum(grade >= threshold for grade in grades[:k]) / k


def dcg(grades: Sequence[int]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))


def ndcg_at_k(ranked_grades: Sequence[int], pool_grades: Sequence[int], k: int) -> float:
    """nDCG with gains 2^grade - 1; the ideal ranking is the best k of the judged pool."""
    ideal = dcg(sorted(pool_grades, reverse=True)[:k])
    return 0.0 if ideal == 0 else dcg(ranked_grades[:k]) / ideal


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        raise ValueError("At least one latency is required")
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * percentile_value), len(ordered) - 1)
    return ordered[index]
