import pytest

from app.config import Settings
from app.graph import factory
from app.graph.factory import LazyRerankedSearchTool, LazySQLTool, build_career_agent
from app.graph.router import RouteDecision


class FakeRouter:
    def decide(self, question: str) -> RouteDecision:
        return RouteDecision(route="search", sql_scope="all")


class FakeClient:
    closed = False

    def close(self) -> None:
        self.closed = True


def test_build_career_agent_wires_lazy_tools() -> None:
    settings = Settings(agent_search_limit=3, agent_tool_max_retries=0)

    agent = build_career_agent(None, settings, FakeRouter())

    assert isinstance(agent.sql_tool, LazySQLTool)
    assert isinstance(agent.search_tool, LazyRerankedSearchTool)
    assert agent.search_limit == 3
    assert agent.tool_max_retries == 0


def test_lazy_search_tool_closes_qdrant_client_on_failure(monkeypatch) -> None:
    client = FakeClient()
    monkeypatch.setattr(factory, "build_embedding_provider", lambda settings: object())
    monkeypatch.setattr(factory, "build_qdrant_client", lambda url, timeout: client)

    def fail(settings):
        raise RuntimeError("reranker unavailable")

    monkeypatch.setattr(factory, "build_reranker", fail)

    with pytest.raises(RuntimeError):
        LazyRerankedSearchTool(None, Settings()).invoke("python jobs")

    assert client.closed
