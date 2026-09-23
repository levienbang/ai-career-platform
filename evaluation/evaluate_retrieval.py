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
from app.retrieval.keyword import KeywordSearchService

CASES_FILE = Path(__file__).with_name("retrieval_cases.json")
RESULTS_FILE = Path(__file__).with_name("retrieval_results.json")


@dataclass(frozen=True)
class MethodMetrics:
    recall_at_5: float
    mean_latency_ms: float
    median_latency_ms: float


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        raise ValueError("Each evaluation case needs at least one relevant job")
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


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
    client = QdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_seconds)

    keyword_recalls: list[float] = []
    keyword_latencies: list[float] = []
    dense_recalls: list[float] = []
    dense_latencies: list[float] = []
    case_results: list[dict[str, object]] = []

    with SessionLocal() as session:
        keyword = KeywordSearchService(session)
        dense = DenseSearchService(client, embedder, settings.qdrant_collection)
        for case in cases:
            relevant_ids = _resolve_relevant_ids(session, case["relevant_source_urls"])

            started = perf_counter()
            keyword_hits = keyword.search(case["query"], limit=5)
            keyword_latency = (perf_counter() - started) * 1000

            started = perf_counter()
            dense_hits = dense.search(case["query"], limit=5)
            dense_latency = (perf_counter() - started) * 1000

            keyword_ids = [hit.job_id for hit in keyword_hits]
            dense_ids = [hit.job_id for hit in dense_hits]
            keyword_recall = recall_at_k(keyword_ids, relevant_ids, 5)
            dense_recall = recall_at_k(dense_ids, relevant_ids, 5)
            keyword_recalls.append(keyword_recall)
            keyword_latencies.append(keyword_latency)
            dense_recalls.append(dense_recall)
            dense_latencies.append(dense_latency)
            case_results.append(
                {
                    "query": case["query"],
                    "relevant_job_ids": sorted(relevant_ids),
                    "keyword_job_ids": keyword_ids,
                    "dense_job_ids": dense_ids,
                    "keyword_recall_at_5": keyword_recall,
                    "dense_recall_at_5": dense_recall,
                }
            )

    metrics = {
        "keyword": asdict(
            MethodMetrics(
                recall_at_5=mean(keyword_recalls),
                mean_latency_ms=mean(keyword_latencies),
                median_latency_ms=median(keyword_latencies),
            )
        ),
        "dense": asdict(
            MethodMetrics(
                recall_at_5=mean(dense_recalls),
                mean_latency_ms=mean(dense_latencies),
                median_latency_ms=median(dense_latencies),
            )
        ),
        "case_count": len(cases),
        "cases": case_results,
    }
    RESULTS_FILE.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
