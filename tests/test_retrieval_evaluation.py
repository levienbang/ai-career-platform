import pytest

from evaluation.evaluate_retrieval import recall_at_k


def test_recall_at_k_counts_relevant_results() -> None:
    assert recall_at_k([3, 1, 7, 9], {1, 2}, 3) == 0.5


def test_recall_at_k_rejects_empty_labels() -> None:
    with pytest.raises(ValueError):
        recall_at_k([1], set(), 5)
