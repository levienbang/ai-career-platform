"""Compare LLM rerankers on the saved paraphrase pools with the manual 0/1/2 grades.

Every reranker sees the same 10 hybrid candidates per query (rerank_pools.json), so a
run costs one chat call per query and no embedding calls. The chat model comes from
the settings: DeepSeek by default, or a Bedrock model with
LLM_PROVIDER=bedrock BEDROCK_MODEL=us.amazon.nova-pro-v1:0.

Dry-run by default: prints the planned calls and exits. Pass --run to call the model.
Results are merged into rerank_pool_results.json under the model name, next to the
hybrid order. A job a reranker puts in its top 5 that has no grade yet counts as 0 and
is listed under "ungraded", so it can be graded and the run rescored with --rescore.
"""

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from app.config import Settings
from app.retrieval.reranker import LangChainJobReranker
from app.retrieval.types import SearchHit
from evaluation.metrics import judged_precision_at_k, ndcg_at_k

HERE = Path(__file__).parent
POOLS_FILE = HERE / "rerank_pools.json"
GRADES_FILE = HERE / "manual_grades_paraphrase.json"
RESULTS_FILE = HERE / "rerank_pool_results.json"
K = 5


def score_ranking(ranked: list[int], grades: dict[str, int]) -> dict[str, object]:
    top = ranked[:K]
    top_grades = [grades.get(str(job_id), 0) for job_id in top]
    return {
        "top5": top,
        "grades": top_grades,
        "ndcg_at_5": round(ndcg_at_k(top_grades, list(grades.values()), K), 3),
        "p_at_5_relevant": judged_precision_at_k(top_grades, K, threshold=2),
        "p_at_5_partial": judged_precision_at_k(top_grades, K, threshold=1),
        "ungraded": [job_id for job_id in top if str(job_id) not in grades],
    }


def summarize(rows: list[dict]) -> dict[str, object]:
    summary = {
        name: round(mean(row[name] for row in rows), 3)
        for name in ("ndcg_at_5", "p_at_5_relevant", "p_at_5_partial")
    }
    latencies = [row["latency_ms"] for row in rows if row.get("latency_ms") is not None]
    if latencies:
        summary["median_latency_ms"] = sorted(latencies)[len(latencies) // 2]
    summary["errors"] = sum(bool(row.get("error")) for row in rows)
    summary["ungraded"] = sum(len(row["ungraded"]) for row in rows)
    return summary


def rerank_rows(reranker, pools: list[dict], grades: dict) -> list[dict]:
    rows = []
    for pool in pools:
        candidates = [SearchHit(**candidate) for candidate in pool["candidates"]]
        started = perf_counter()
        error = None
        try:
            ranked = [hit.job_id for hit in reranker.rerank(pool["query"], candidates)]
        except Exception as exc:  # record and keep hybrid order, as the API would
            ranked, error = [hit.job_id for hit in candidates], type(exc).__name__
        rows.append(
            {
                "id": pool["id"],
                **score_ranking(ranked, grades[pool["id"]]),
                "latency_ms": round((perf_counter() - started) * 1000),
                "error": error,
            }
        )
    return rows


def rescore(results: dict, grades: dict) -> None:
    for method in results.values():
        for row in method["cases"]:
            latency, error = row.get("latency_ms"), row.get("error")
            row.update(score_ranking(row["top5"], grades[row["id"]]))
            row["latency_ms"], row["error"] = latency, error
        method["summary"] = summarize(method["cases"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Make the paid chat calls")
    parser.add_argument("--rescore", action="store_true", help="Re-grade saved rankings")
    args = parser.parse_args()
    pools = json.loads(POOLS_FILE.read_text(encoding="utf-8"))
    grades = json.loads(GRADES_FILE.read_text(encoding="utf-8"))
    results = json.loads(RESULTS_FILE.read_text(encoding="utf-8")) if RESULTS_FILE.exists() else {}

    if args.rescore:
        rescore(results, grades)
    else:
        settings = Settings()
        name = (
            settings.bedrock_model
            if settings.llm_provider == "bedrock"
            else settings.deepseek_model
        )
        print(f"Planned paid calls: {len(pools)} to {name}")
        if not args.run:
            print("Dry-run only; pass --run to evaluate.")
            return
        hybrid = [
            {
                "id": pool["id"],
                **score_ranking([c["job_id"] for c in pool["candidates"]], grades[pool["id"]]),
            }
            for pool in pools
        ]
        results["hybrid"] = {"summary": summarize(hybrid), "cases": hybrid}
        rows = rerank_rows(LangChainJobReranker(settings), pools, grades)
        results[name] = {"summary": summarize(rows), "cases": rows}

    RESULTS_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: value["summary"] for name, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
