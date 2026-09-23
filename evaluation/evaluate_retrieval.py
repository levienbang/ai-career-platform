import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from qdrant_client import QdrantClient

from app.config import Settings
from app.db.repositories import JobRepository
from app.db.session import SessionLocal
from app.retrieval.dense import DenseSearchService
from app.retrieval.embeddings import build_embedding_provider
from app.retrieval.hybrid import HybridSearchService
from app.retrieval.keyword import KeywordSearchService
from app.retrieval.reranker import RerankedSearchService, build_reranker

CASES_FILE = Path(__file__).with_name("retrieval_cases.json")
RESULTS_FILE = Path(__file__).with_name("retrieval_results.json")


@dataclass(frozen=True)
class MethodMetrics:
    recall_at_5: float
    mrr_at_10: float
    mean_latency_ms: float
    median_latency_ms: float


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        raise ValueError("Each evaluation case needs at least one relevant job")
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def mrr_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        raise ValueError("Each evaluation case needs at least one relevant job")
    for rank, job_id in enumerate(retrieved_ids[:k], start=1):
        if job_id in relevant_ids:
            return 1 / rank
    return 0.0


def _resolve_relevant_ids(session, source_urls: list[str]) -> set[int]:
    jobs = JobRepository(session).get_by_source_urls(source_urls)
    resolved = {job.source_url: job.id for job in jobs}
    missing = sorted(set(source_urls) - set(resolved))
    if missing:
        raise RuntimeError(
            "Evaluation jobs are missing from PostgreSQL; run python -m scripts.seed. "
            f"Missing: {missing}"
        )
    return {resolved[source_url] for source_url in source_urls}


def evaluate() -> dict[str, object]:
    settings = Settings()
    cases = json.loads(CASES_FILE.read_text())
    embedder = build_embedding_provider(settings)
    reranker = build_reranker(settings)
    client = QdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_seconds)

    method_names = ("keyword", "dense", "hybrid", "reranked")
    recalls: dict[str, list[float]] = {name: [] for name in method_names}
    reciprocal_ranks: dict[str, list[float]] = {name: [] for name in method_names}
    latencies: dict[str, list[float]] = {name: [] for name in method_names}
    case_results: list[dict[str, object]] = []

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
        for case in cases:
            relevant_ids = _resolve_relevant_ids(session, case["relevant_source_urls"])
            result: dict[str, object] = {
                "query": case["query"],
                "relevant_job_ids": sorted(relevant_ids),
            }
            for method_name, method in methods.items():
                started = perf_counter()
                hits = method.search(case["query"], limit=10)
                latency = (perf_counter() - started) * 1000
                job_ids = [hit.job_id for hit in hits]
                recall = recall_at_k(job_ids, relevant_ids, 5)
                reciprocal_rank = mrr_at_k(job_ids, relevant_ids, 10)
                recalls[method_name].append(recall)
                reciprocal_ranks[method_name].append(reciprocal_rank)
                latencies[method_name].append(latency)
                result[f"{method_name}_job_ids"] = job_ids
                result[f"{method_name}_recall_at_5"] = recall
                result[f"{method_name}_mrr_at_10"] = reciprocal_rank
            case_results.append(result)

    metrics: dict[str, object] = {
        method_name: asdict(
            MethodMetrics(
                recall_at_5=mean(recalls[method_name]),
                mrr_at_10=mean(reciprocal_ranks[method_name]),
                mean_latency_ms=mean(latencies[method_name]),
                median_latency_ms=median(latencies[method_name]),
            )
        )
        for method_name in method_names
    }
    metrics["case_count"] = len(cases)
    metrics["cases"] = case_results
    RESULTS_FILE.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
