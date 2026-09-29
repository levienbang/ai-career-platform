import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from app.config import Settings
from app.graph.router import build_question_router
from evaluation.evaluate_agent import CASES_FILE, percentile

RESULTS_FILE = Path(__file__).with_name("routing_results.json")


def evaluate() -> dict[str, object]:
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    router = build_question_router(Settings())
    correct = 0
    latencies: list[float] = []
    case_results: list[dict[str, object]] = []

    for case in cases:
        started_at = perf_counter()
        actual_route = None
        error_message = None
        try:
            actual_route = router.decide(case["question"]).route
        except Exception as error:
            error_message = str(error)
        latency_ms = (perf_counter() - started_at) * 1000
        route_correct = actual_route == case["expected_route"]
        correct += int(route_correct)
        latencies.append(latency_ms)
        case_results.append(
            {
                "question": case["question"],
                "expected_route": case["expected_route"],
                "actual_route": actual_route,
                "route_correct": route_correct,
                "latency_ms": latency_ms,
                "error": error_message,
            }
        )

    metrics: dict[str, object] = {
        "case_count": len(cases),
        "routing_accuracy": correct / len(cases),
        "routing_success_rate": sum(case["actual_route"] is not None for case in case_results)
        / len(cases),
        "mean_latency_ms": mean(latencies),
        "median_latency_ms": median(latencies),
        "p95_latency_ms": percentile(latencies, 0.95),
        "cases": case_results,
    }
    RESULTS_FILE.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
