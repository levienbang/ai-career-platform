from pydantic import BaseModel, Field

from app.graph.builder import AgentRunResult
from app.tools.sql_tool import SQLToolResult


class NaturalLanguageQuery(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class AnalyticsResponse(BaseModel):
    question: str
    result: SQLToolResult
    latency_ms: float = Field(ge=0)


class AgentResponse(AgentRunResult):
    pass
