import pytest

from app.config import Settings
from app.graph.router import AgentRouterConfigurationError, build_question_router
from app.tools.sql_tool import SQLToolConfigurationError, build_sql_tool


def test_router_requires_agent_or_llm_api_key() -> None:
    settings = Settings(agent_api_key=None, llm_api_key=None, _env_file=None)

    with pytest.raises(AgentRouterConfigurationError, match="Gemini API key or OLLAMA_BASE_URL"):
        build_question_router(settings)


def test_sql_tool_requires_readonly_database_url() -> None:
    settings = Settings(readonly_database_url=None, _env_file=None)

    with pytest.raises(SQLToolConfigurationError, match="READONLY_DATABASE_URL"):
        build_sql_tool(settings)
