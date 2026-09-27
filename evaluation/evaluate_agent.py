import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from qdrant_client import QdrantClient

from app.config import Settings
from app.db.session import SessionLocal
from app.graph.builder import AgentRunResult, CareerAgent
from app.graph.router import build_question_router
from app.retrieval.embeddings import build_embedding_provider
from app.retrieval.factory import build_reranked_search_service
from app.retrieval.reranker import build_reranker
from app.tools.search_tool import HybridJobSearchTool
from app.tools.sql_tool import build_sql_tool

CASES_FILE = Path(__file__).with_name("agent_cases.json")
RESULTS_FILE = Path(__file__).with_name("agent_results.json")


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        raise ValueError("At least one latency is required")
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * percentile_value), len(ordered) - 1)
    return ordered[index]


def required_tools_succeeded(result: AgentRunResult, expected_route: str) -> bool:
    if result.errors:
        return False
    if expected_route in {"sql", "both"} and result.sql_result is None:
        return False
    if expected_route in {"search", "both"} and result.search_result is None:
        return False
    return True


def evaluate() -> dict[str, object]:
    settings = Settings()
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    latencies: list[float] = []
    correct_routes = 0
    successful_tools = 0
    case_results: list[dict[str, object]] = []

    router = build_question_router(settings)
    sql_tool = build_sql_tool(settings)
    embedder = build_embedding_provider(settings)
    reranker = build_reranker(settings)
    client = QdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_seconds)

    with SessionLocal() as session:
        search_tool = HybridJobSearchTool(
            build_reranked_search_service(session, embedder, client, reranker, settings)
        )
        agent = CareerAgent(
            router,
            sql_tool,
            search_tool,
            search_limit=settings.agent_search_limit,
            tool_max_retries=settings.agent_tool_max_retries,
        )
        for case in cases:
            started_at = perf_counter()
            try:
                result = agent.invoke(case["question"])
            except Exception as error:
                result = None
                latency_ms = (perf_counter() - started_at) * 1000
                actual_route = None
                errors = [str(error)]
                route_correct = False
                tool_success = False
            else:
                latency_ms = result.latency_ms
                actual_route = result.route
                errors = result.errors
                route_correct = result.route == case["expected_route"]
                tool_success = required_tools_succeeded(result, case["expected_route"])
            correct_routes += int(route_correct)
            successful_tools += int(tool_success)
            latencies.append(latency_ms)
            case_results.append(
                {
                    "question": case["question"],
                    "expected_route": case["expected_route"],
                    "actual_route": actual_route,
                    "route_correct": route_correct,
                    "tool_success": tool_success,
                    "latency_ms": latency_ms,
                    "errors": errors,
                }
            )

    metrics: dict[str, object] = {
        "case_count": len(cases),
        "routing_accuracy": correct_routes / len(cases),
        "tool_success_rate": successful_tools / len(cases),
        "mean_latency_ms": mean(latencies),
        "median_latency_ms": median(latencies),
        "p95_latency_ms": percentile(latencies, 0.95),
        "cases": case_results,
    }
    RESULTS_FILE.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
