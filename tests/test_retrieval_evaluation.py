import pytest

from evaluation.evaluate_retrieval import mrr_at_k, recall_at_k


def test_recall_at_k_counts_relevant_results() -> None:
    assert recall_at_k([3, 1, 7, 9], {1, 2}, 3) == 0.5


def test_recall_at_k_rejects_empty_labels() -> None:
    with pytest.raises(ValueError):
        recall_at_k([1], set(), 5)


def test_mrr_at_k_uses_first_relevant_rank() -> None:
    assert mrr_at_k([9, 3, 1, 2], {1, 2}, 10) == 1 / 3


def test_mrr_at_k_returns_zero_when_no_relevant_result() -> None:
    assert mrr_at_k([9, 3], {1, 2}, 10) == 0
