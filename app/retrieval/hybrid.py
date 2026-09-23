from dataclasses import replace
from typing import Protocol

from app.retrieval.types import SearchHit


class RankedSearch(Protocol):
    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]: ...


def reciprocal_rank_fusion(
    rankings: list[list[SearchHit]], *, rrf_k: int = 60, limit: int | None = None
) -> list[SearchHit]:
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")

    scores: dict[int, float] = {}
    best_ranks: dict[int, int] = {}
    hits: dict[int, SearchHit] = {}

    for ranking in rankings:
        seen_in_ranking: set[int] = set()
        for rank, hit in enumerate(ranking, start=1):
            if hit.job_id in seen_in_ranking:
                continue
            seen_in_ranking.add(hit.job_id)
            scores[hit.job_id] = scores.get(hit.job_id, 0.0) + 1 / (rrf_k + rank)
            best_ranks[hit.job_id] = min(best_ranks.get(hit.job_id, rank), rank)
            hits.setdefault(hit.job_id, hit)

    ordered_ids = sorted(
        hits,
        key=lambda job_id: (-scores[job_id], best_ranks[job_id], job_id),
    )
    if limit is not None:
        ordered_ids = ordered_ids[:limit]
    return [replace(hits[job_id], score=scores[job_id]) for job_id in ordered_ids]


class HybridSearchService:
    def __init__(
        self,
        keyword: RankedSearch,
        dense: RankedSearch,
        *,
        candidate_limit: int = 20,
        rrf_k: int = 60,
    ) -> None:
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        self.keyword = keyword
        self.dense = dense
        self.candidate_limit = candidate_limit
        self.rrf_k = rrf_k

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        source_limit = max(limit, self.candidate_limit)
        keyword_hits = self.keyword.search(query, limit=source_limit)
        dense_hits = self.dense.search(query, limit=source_limit)
        return reciprocal_rank_fusion([keyword_hits, dense_hits], rrf_k=self.rrf_k, limit=limit)
