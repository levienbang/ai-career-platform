from dataclasses import replace

from app.retrieval.hybrid import HybridSearchService, reciprocal_rank_fusion
from app.retrieval.types import SearchHit


def _hit(job_id: int, score: float = 0.0) -> SearchHit:
    return SearchHit(
        job_id=job_id,
        score=score,
        title=f"Job {job_id}",
        company="Company",
        location=None,
        employment_type=None,
        experience_years_min=None,
        required_skills=[],
        preferred_skills=[],
    )


def test_rrf_fuses_ranks_and_removes_duplicate_job_ids() -> None:
    keyword = [_hit(1, 1000), _hit(2, 500)]
    dense = [_hit(2, 0.9), _hit(3, 0.8)]

    results = reciprocal_rank_fusion([keyword, dense], rrf_k=60)

    assert [hit.job_id for hit in results] == [2, 1, 3]
    assert len({hit.job_id for hit in results}) == 3
    assert results[0].score == 1 / 62 + 1 / 61


def test_rrf_ignores_raw_scores_and_breaks_ties_by_job_id() -> None:
    results = reciprocal_rank_fusion([[_hit(2, 9999)], [_hit(1, -9999)]], rrf_k=60)

    assert [hit.job_id for hit in results] == [1, 2]


def test_rrf_counts_a_duplicate_only_once_per_source() -> None:
    duplicate = _hit(1)
    results = reciprocal_rank_fusion([[duplicate, replace(duplicate, score=99)]], rrf_k=60)

    assert len(results) == 1
    assert results[0].score == 1 / 61


def test_hybrid_search_uses_configured_candidate_depth() -> None:
    class StubSearch:
        def __init__(self, hits):
            self.hits = hits
            self.limits = []

        def search(self, query: str, *, limit: int = 5):
            assert query == "python"
            self.limits.append(limit)
            return self.hits[:limit]

    keyword = StubSearch([_hit(1), _hit(2)])
    dense = StubSearch([_hit(2), _hit(3)])

    results = HybridSearchService(keyword, dense, candidate_limit=20).search("python", limit=2)

    assert keyword.limits == [20]
    assert dense.limits == [20]
    assert [hit.job_id for hit in results] == [2, 1]
