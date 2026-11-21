import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from qdrant_client import QdrantClient

from app.config import Settings
from app.db.repositories import JobRepository
from app.db.session import SessionLocal
from app.retrieval.dense import DenseSearchService
from app.retrieval.embeddings import build_embedding_provider
from app.retrieval.hybrid import HybridSearchService, RankedSearch
from app.retrieval.keyword import KeywordSearchService
from app.retrieval.reranker import RerankedSearchService, build_reranker
from evaluation.metrics import mrr_at_k, recall_at_k

CASES_FILE = Path(__file__).with_name("retrieval_cases.json")
RESULTS_FILE = Path(__file__).with_name("retrieval_results.json")


@dataclass
class MethodScores:
    recalls: list[float] = field(default_factory=list)
    reciprocal_ranks: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)

    def mean_quality(self) -> dict[str, float]:
        return {
            "recall_at_5": mean(self.recalls),
            "mrr_at_10": mean(self.reciprocal_ranks),
        }


def _resolve_relevant_ids(repository: JobRepository, source_urls: list[str]) -> set[int]:
    jobs = repository.get_by_source_urls(source_urls)
    resolved = {job.source_url: job.id for job in jobs}
    missing = sorted(set(source_urls) - set(resolved))
    if missing:
        raise RuntimeError(
            "Evaluation jobs are missing from PostgreSQL; run python -m scripts.load_demo_jobs. "
            f"Missing: {missing}"
        )
    return {resolved[source_url] for source_url in source_urls}


def score_retrieval_cases(
    repository: JobRepository,
    cases: list[dict],
    methods: Mapping[str, RankedSearch],
) -> tuple[dict[str, MethodScores], list[dict[str, object]]]:
    scores = {name: MethodScores() for name in methods}
    case_results: list[dict[str, object]] = []

    for case in cases:
        query = case["query"]
        relevant_ids = _resolve_relevant_ids(repository, case["relevant_source_urls"])
        result: dict[str, object] = {
            "query": query,
            "relevant_job_ids": sorted(relevant_ids),
        }
        for name, method in methods.items():
            started = perf_counter()
            hits = method.search(query, limit=10)
            latency = (perf_counter() - started) * 1000
            job_ids = [hit.job_id for hit in hits]
            recall = recall_at_k(job_ids, relevant_ids, 5)
            reciprocal_rank = mrr_at_k(job_ids, relevant_ids, 10)
            scores[name].recalls.append(recall)
            scores[name].reciprocal_ranks.append(reciprocal_rank)
            scores[name].latencies_ms.append(latency)
            result[f"{name}_job_ids"] = job_ids
            result[f"{name}_recall_at_5"] = recall
            result[f"{name}_mrr_at_10"] = reciprocal_rank
        case_results.append(result)

    return scores, case_results


def evaluate() -> dict[str, object]:
    settings = Settings()
    cases = json.loads(CASES_FILE.read_text())
    embedder = build_embedding_provider(settings)
    reranker = build_reranker(settings)
    client = QdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_seconds)

    with SessionLocal() as session:
        keyword = KeywordSearchService(session)
        dense = DenseSearchService(client, embedder, settings.qdrant_collection)
        hybrid = HybridSearchService(
            keyword,
            dense,
            candidate_limit=settings.hybrid_candidate_limit,
            rrf_k=settings.rrf_k,
        )
        reranked = RerankedSearchService(
            hybrid,
            reranker,
            candidate_limit=settings.reranker_max_candidates,
        )
        methods = {
            "keyword": keyword,
            "dense": dense,
            "hybrid": hybrid,
            "reranked": reranked,
        }
        scores, case_results = score_retrieval_cases(JobRepository(session), cases, methods)

    metrics: dict[str, object] = {
        name: {
            **score.mean_quality(),
            "mean_latency_ms": mean(score.latencies_ms),
            "median_latency_ms": median(score.latencies_ms),
        }
        for name, score in scores.items()
    }
    metrics["case_count"] = len(cases)
    metrics["cases"] = case_results
    RESULTS_FILE.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
