from typing import Literal, TypedDict

from app.tools.search_tool import SearchToolResult
from app.tools.sql_tool import SQLToolResult

AgentRoute = Literal["search", "sql", "both"]
SQLScope = Literal["all", "search_results"]


class AgentState(TypedDict, total=False):
    question: str
    route: AgentRoute
    sql_scope: SQLScope
    sql_result: SQLToolResult
    search_result: SearchToolResult
    answer: str
    errors: list[str]
    retry_count: int
