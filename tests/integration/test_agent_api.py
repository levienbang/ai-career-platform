from app.api.agent import get_career_agent, get_sql_analytics_tool
from app.graph.builder import AgentRunResult
from app.tools.sql_tool import SQLToolResult, SQLValidationError


class FakeSQLTool:
    def invoke(self, question: str) -> SQLToolResult:
        return SQLToolResult(
            sql="SELECT COUNT(*) AS count FROM jobs LIMIT 100",
            columns=["count"],
            rows=[[4]],
            row_count=1,
        )


class UnsafeSQLTool:
    def invoke(self, question: str) -> SQLToolResult:
        raise SQLValidationError("Only SELECT or WITH ... SELECT is allowed")


class FakeAgent:
    def invoke(self, question: str) -> AgentRunResult:
        return AgentRunResult(
            question=question,
            route="sql",
            answer='SQL result: {"columns": ["count"], "rows": [[4]]}',
            sql_result=FakeSQLTool().invoke(question),
            latency_ms=1,
        )


def test_analytics_api_returns_structured_sql_result(client) -> None:
    client.app.dependency_overrides[get_sql_analytics_tool] = lambda: FakeSQLTool()

    response = client.post("/analytics/query", json={"question": "Có bao nhiêu job?"})

    assert response.status_code == 200
    assert response.json()["result"]["rows"] == [[4]]


def test_analytics_api_rejects_unsafe_generated_sql(client) -> None:
    client.app.dependency_overrides[get_sql_analytics_tool] = lambda: UnsafeSQLTool()

    response = client.post("/analytics/query", json={"question": "delete everything"})

    assert response.status_code == 422
    assert "Only SELECT" in response.json()["detail"]


def test_agent_api_returns_graph_result(client) -> None:
    client.app.dependency_overrides[get_career_agent] = lambda: FakeAgent()

    response = client.post("/agent/query", json={"question": "Có bao nhiêu job?"})

    assert response.status_code == 200
    assert response.json()["route"] == "sql"
    assert response.json()["sql_result"]["rows"] == [[4]]


def test_agent_api_validates_empty_question(client) -> None:
    client.app.dependency_overrides[get_career_agent] = lambda: FakeAgent()

    response = client.post("/agent/query", json={"question": ""})

    assert response.status_code == 422
