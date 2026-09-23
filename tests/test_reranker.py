from dataclasses import replace

import pytest

from app.retrieval.reranker import (
    RerankedSearchService,
    RerankerInputError,
    RerankerResponseError,
)
from app.retrieval.types import SearchHit


def _hit(job_id: int) -> SearchHit:
    return SearchHit(
        job_id=job_id,
        score=0.1,
        title=f"Job {job_id}",
        company="Company",
        location=None,
        employment_type=None,
        experience_years_min=None,
        required_skills=[],
        preferred_skills=[],
    )


class StubHybrid:
    def __init__(self, candidates: list[SearchHit]) -> None:
        self.candidates = candidates

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        del query
        return self.candidates[:limit]


def test_fake_reranker_can_only_reorder_candidate_set(fake_reranker) -> None:
    candidates = [_hit(1), _hit(2), _hit(3)]
    service = RerankedSearchService(StubHybrid(candidates), fake_reranker, candidate_limit=3)

    results = service.search("query", limit=2)

    assert [hit.job_id for hit in results] == [3, 2]
    assert set(hit.job_id for hit in results).issubset({1, 2, 3})


def test_reranker_cannot_add_a_job_outside_candidates() -> None:
    class BadReranker:
        def rerank(self, query, candidates):
            del query
            return [*candidates, replace(candidates[0], job_id=999)]

    service = RerankedSearchService(
        StubHybrid([_hit(1), _hit(2)]), BadReranker(), candidate_limit=2
    )

    with pytest.raises(RerankerResponseError):
        service.search("query", limit=2)


def test_reranker_rejects_limit_above_candidate_pool(fake_reranker) -> None:
    service = RerankedSearchService(
        StubHybrid([_hit(1), _hit(2)]), fake_reranker, candidate_limit=2
    )

    with pytest.raises(RerankerInputError):
        service.search("query", limit=3)
