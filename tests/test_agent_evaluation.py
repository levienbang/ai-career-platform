from app.graph.builder import AgentRunResult
from app.tools.sql_tool import SQLToolResult
from evaluation.evaluate_agent import percentile, required_tools_succeeded


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
