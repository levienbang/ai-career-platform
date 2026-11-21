import pytest

from app.config import Settings
from app.graph.router import AgentRouterConfigurationError, build_question_router
from app.tools.sql_tool import SQLToolConfigurationError, build_sql_tool


def test_router_requires_agent_or_deepseek_api_key() -> None:
    settings = Settings(deepseek_api_key=None, _env_file=None)

    with pytest.raises(AgentRouterConfigurationError, match="DEEPSEEK_API_KEY"):
        build_question_router(settings)


def test_sql_tool_requires_readonly_database_url() -> None:
    settings = Settings(readonly_database_url=None, _env_file=None)

    with pytest.raises(SQLToolConfigurationError, match="READONLY_DATABASE_URL"):
        build_sql_tool(settings)
