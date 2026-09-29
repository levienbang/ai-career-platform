from langchain_core.callbacks import BaseCallbackHandler
from pydantic import SecretStr

from app.config import Settings
from app.graph.builder import CareerAgent
from app.graph.router import RouteDecision
from app.tools.sql_tool import SQLToolResult
from app.tracing import build_agent_callbacks


class FakeRouter:
    def decide(self, question):
        return RouteDecision(route="sql", sql_scope="all")


class FakeSQL:
    def invoke(self, question, *, job_ids=None):
        return SQLToolResult(
            sql="SELECT COUNT(*) FROM jobs", columns=["count"], rows=[[1]], row_count=1
        )


class UnusedSearch:
    def invoke(self, query, limit=5):
        raise AssertionError("Search should not run")


class RecordingCallback(BaseCallbackHandler):
    def __init__(self):
        self.starts = 0
        self.ends = 0

    def on_chain_start(self, serialized, inputs, **kwargs):
        self.starts += 1

    def on_chain_end(self, outputs, **kwargs):
        self.ends += 1


def test_tracing_is_disabled_without_key():
    settings = Settings(langsmith_tracing=True, langsmith_api_key=None)
    assert build_agent_callbacks(settings) == []


def test_tracing_builds_langsmith_callback_when_enabled(monkeypatch):
    from app import tracing

    monkeypatch.setattr(tracing, "Client", lambda **kwargs: object())
    monkeypatch.setattr(tracing, "LangChainTracer", lambda **kwargs: "fake-tracer")
    settings = Settings(langsmith_tracing=True, langsmith_api_key=SecretStr("test-key"))
    assert build_agent_callbacks(settings) == ["fake-tracer"]


def test_agent_emits_full_graph_trace_without_network():
    callback = RecordingCallback()
    result = CareerAgent(FakeRouter(), FakeSQL(), UnusedSearch(), callbacks=[callback]).invoke(
        "Count jobs"
    )
    assert result.route == "sql"
    assert callback.starts >= 3
    assert callback.ends >= 3
