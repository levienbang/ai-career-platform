"""Evaluate routing, SQL answers and retrieval on the real job corpus.

Dry-run by default: prints the planned paid calls and exits. Pass --run to call
DeepSeek (routing + SQL generation) and Google embedding (one call per query).
--only limits the run to some sections; other sections keep their saved results.
--rescore re-grades saved SQL answers after label fixes, with no API calls.
"""

import argparse
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from qdrant_client import QdrantClient

from app.config import Settings
from app.db.session import SessionLocal
from app.graph.router import build_question_router
from app.retrieval.dense import DenseSearchService
from app.retrieval.embeddings import build_embedding_provider
from app.retrieval.hybrid import HybridSearchService
from app.retrieval.keyword import KeywordSearchService
from app.tools.sql_tool import build_sql_tool

CASES_FILE = Path(__file__).with_name("real_cases.json")
RESULTS_FILE = Path(__file__).with_name("real_results.json")


class CachedQueryEmbedder:
    """Dense and hybrid search embed the same query; pay for it once."""

    def __init__(self, provider) -> None:
        self.provider = provider
        self.calls = 0
        self._cache: dict[str, list[float]] = {}

    @property
    def dimensions(self) -> int:
        return self.provider.dimensions

    def embed_documents(self, texts, titles):
        raise RuntimeError("Evaluation never embeds documents")

    def embed_query(self, text: str) -> list[float]:
        if text not in self._cache:
            self.calls += 1
            self._cache[text] = self.provider.embed_query(text)
        return self._cache[text]


def precision_at_k(ids: list[int], relevant: set[int], k: int) -> float:
    return len([job_id for job_id in ids[:k] if job_id in relevant]) / k


def mrr_at_k(ids: list[int], relevant: set[int], k: int) -> float:
    for rank, job_id in enumerate(ids[:k], start=1):
        if job_id in relevant:
            return 1 / rank
    return 0.0


def evaluate_routing(settings: Settings, cases: list[dict]) -> dict:
    router = build_question_router(settings)
    rows = []
    for case in cases:
        started = perf_counter()
        try:
            route, error = router.decide(case["question"]).route, None
        except Exception as exc:  # report, do not stop the whole evaluation
            route, error = None, type(exc).__name__
        rows.append(
            {
                "question": case["question"],
                "expected_route": case["expected_route"],
                "actual_route": route,
                "correct": route == case["expected_route"],
                "latency_ms": round((perf_counter() - started) * 1000),
                "error": error,
            }
        )
    latencies = [row["latency_ms"] for row in rows]
    return {
        "accuracy": mean(row["correct"] for row in rows),
        "median_latency_ms": median(latencies),
        "cases": rows,
    }


def _sql_correct(case: dict, rows: list[list]) -> bool:
    if not rows:
        return False
    if "accepted_text" in case:
        return case["accepted_text"].casefold() in str(rows[0]).casefold()
    return any(value in case["accepted"] for value in rows[0] if isinstance(value, int))


def evaluate_sql(settings: Settings, cases: list[dict]) -> dict:
    tool = build_sql_tool(settings)
    rows = []
    for case in cases:
        started = perf_counter()
        try:
            result = tool.invoke(case["question"])
            sql, answer, error = result.sql, result.rows[:3], None
            correct = _sql_correct(case, result.rows)
        except Exception as exc:
            sql, answer, error, correct = None, None, type(exc).__name__, False
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "sql": sql,
                "answer": answer,
                "correct": correct,
                "latency_ms": round((perf_counter() - started) * 1000),
                "error": error,
            }
        )
    return {"accuracy": mean(row["correct"] for row in rows), "cases": rows}


def evaluate_retrieval(settings: Settings, cases: list[dict], embedder) -> dict:
    client = QdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_seconds)
    names = ("keyword", "dense", "hybrid")
    scores = {name: {"precision_at_5": [], "mrr_at_10": [], "latency_ms": []} for name in names}
    rows = []
    with SessionLocal() as session:
        keyword = KeywordSearchService(session)
        dense = DenseSearchService(client, embedder, settings.qdrant_collection)
        hybrid = HybridSearchService(
            keyword, dense, candidate_limit=settings.hybrid_candidate_limit, rrf_k=settings.rrf_k
        )
        methods = {"keyword": keyword, "dense": dense, "hybrid": hybrid}
        for case in cases:
            relevant = set(case["relevant_job_ids"])
            row = {"id": case["id"], "query": case["query"], "relevant_count": len(relevant)}
            for name, method in methods.items():
                started = perf_counter()
                ids = [hit.job_id for hit in method.search(case["query"], limit=10)]
                scores[name]["latency_ms"].append((perf_counter() - started) * 1000)
                scores[name]["precision_at_5"].append(precision_at_k(ids, relevant, 5))
                scores[name]["mrr_at_10"].append(mrr_at_k(ids, relevant, 10))
                row[name] = {
                    "top10": ids,
                    "precision_at_5": scores[name]["precision_at_5"][-1],
                    "mrr_at_10": round(scores[name]["mrr_at_10"][-1], 3),
                }
            rows.append(row)
    summary = {
        name: {
            "precision_at_5": round(mean(values["precision_at_5"]), 3),
            "mrr_at_10": round(mean(values["mrr_at_10"]), 3),
            "median_latency_ms": round(median(values["latency_ms"]), 1),
        }
        for name, values in scores.items()
    }
    return {**summary, "cases": rows}


def rescore_sql(cases: list[dict]) -> None:
    results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    by_id = {case["id"]: case for case in cases}
    for row in results["sql"]["cases"]:
        row["correct"] = _sql_correct(by_id[row["id"]], row["answer"] or [])
    results["sql"]["accuracy"] = mean(row["correct"] for row in results["sql"]["cases"])
    RESULTS_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(f"SQL accuracy after re-grading: {results['sql']['accuracy']}")


SECTIONS = ("routing", "sql", "retrieval", "retrieval_paraphrase")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Make the paid API calls")
    parser.add_argument("--only", nargs="+", choices=SECTIONS, default=list(SECTIONS))
    parser.add_argument("--rescore", action="store_true", help="Re-grade saved SQL answers")
    args = parser.parse_args()
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    if args.rescore:
        rescore_sql(cases["sql"])
        return
    planned = {
        "deepseek_calls": sum(len(cases[name]) for name in ("routing", "sql") if name in args.only),
        "embedding_calls": sum(
            len(cases[name]) for name in ("retrieval", "retrieval_paraphrase") if name in args.only
        ),
    }
    print(f"Planned paid calls (before retries): {planned}")
    if not args.run:
        print("Dry-run only; pass --run to evaluate.")
        return
    settings = Settings()
    embedder = CachedQueryEmbedder(build_embedding_provider(settings))
    results = json.loads(RESULTS_FILE.read_text(encoding="utf-8")) if RESULTS_FILE.exists() else {}
    results.update(corpus=cases["corpus"], model=settings.deepseek_model)
    if "routing" in args.only:
        results["routing"] = evaluate_routing(settings, cases["routing"])
    if "sql" in args.only:
        results["sql"] = evaluate_sql(settings, cases["sql"])
    for name in ("retrieval", "retrieval_paraphrase"):
        if name in args.only:
            results[name] = evaluate_retrieval(settings, cases[name], embedder)
    RESULTS_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    summary = {
        "routing_accuracy": results.get("routing", {}).get("accuracy"),
        "sql_accuracy": results.get("sql", {}).get("accuracy"),
        **{
            name: {method: results[name][method] for method in ("keyword", "dense", "hybrid")}
            for name in ("retrieval", "retrieval_paraphrase")
            if name in results
        },
        "embedding_calls_this_run": embedder.calls,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved {RESULTS_FILE.name}")


if __name__ == "__main__":
    main()
