import json

from app.graph.builder import AgentRunResult
from app.graph.router import RouteDecision
from app.tools.sql_tool import SQLToolResult
from evaluation import evaluate_routing
from evaluation.evaluate_agent import required_tools_succeeded
from evaluation.metrics import percentile


def test_agent_evaluation_metrics() -> None:
    result = AgentRunResult(
        question="count",
        route="sql",
        answer="SQL result",
        sql_result=SQLToolResult(
            sql="SELECT COUNT(*) FROM jobs LIMIT 100",
            columns=["count"],
            rows=[[1]],
            row_count=1,
        ),
        latency_ms=12,
    )

    assert required_tools_succeeded(result, "sql") is True
    assert required_tools_succeeded(result, "both") is False
    assert percentile([10, 20, 30, 40], 0.95) == 40


def test_routing_evaluation_uses_decide(monkeypatch, tmp_path) -> None:
    cases_file = tmp_path / "cases.json"
    results_file = tmp_path / "results.json"
    cases_file.write_text(
        json.dumps([{"question": "find jobs", "expected_route": "search"}]), encoding="utf-8"
    )

    class FakeRouter:
        def decide(self, question: str) -> RouteDecision:
            assert question == "find jobs"
            return RouteDecision(route="search", sql_scope="all")

    monkeypatch.setattr(evaluate_routing, "CASES_FILE", cases_file)
    monkeypatch.setattr(evaluate_routing, "RESULTS_FILE", results_file)
    monkeypatch.setattr(evaluate_routing, "build_question_router", lambda settings: FakeRouter())

    metrics = evaluate_routing.evaluate()

    assert metrics["accuracy"] == 1.0
    assert metrics["success_rate"] == 1.0
    assert json.loads(results_file.read_text(encoding="utf-8"))["case_count"] == 1


def test_routing_evaluation_records_a_router_failure() -> None:
    class BrokenRouter:
        def decide(self, question: str) -> RouteDecision:
            raise TimeoutError("provider down")

    metrics = evaluate_routing.score_routing(
        BrokenRouter(), [{"question": "q", "expected_route": "sql"}]
    )

    assert metrics["accuracy"] == 0
    assert metrics["success_rate"] == 0
    assert metrics["cases"][0]["error"] == "TimeoutError"
