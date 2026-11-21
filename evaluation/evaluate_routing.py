"""Routing accuracy of the LLM router alone (no tools), on agent_cases.json.

score_routing is shared with evaluate_real_data, which runs it on real_cases.json.
"""

import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from app.config import Settings
from app.graph.router import build_question_router
from evaluation.evaluate_agent import CASES_FILE
from evaluation.metrics import percentile

RESULTS_FILE = Path(__file__).with_name("agent_routing_results.json")


def score_routing(router, cases: list[dict]) -> dict[str, object]:
    """One router.decide call per case; a failure is recorded, not raised."""
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
        "case_count": len(rows),
        "accuracy": mean(row["correct"] for row in rows),
        "success_rate": mean(row["actual_route"] is not None for row in rows),
        "median_latency_ms": median(latencies),
        "p95_latency_ms": percentile(latencies, 0.95),
        "cases": rows,
    }


def evaluate() -> dict[str, object]:
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    metrics = score_routing(build_question_router(Settings()), cases)
    RESULTS_FILE.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
