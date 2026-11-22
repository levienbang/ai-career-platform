from pydantic import BaseModel, Field

from app.graph.builder import AgentRunResult
from app.graph.chatbot import ChatRunResult, ChatTurn
from app.tools.sql_tool import SQLToolResult


class NaturalLanguageQuery(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class AnalyticsResponse(BaseModel):
    question: str
    result: SQLToolResult
    latency_ms: float = Field(ge=0)


class AgentResponse(AgentRunResult):
    pass


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    # The token budget, not this count, decides how much history the model reads.
    history: list[ChatTurn] = Field(default_factory=list, max_length=400)


class ChatResponse(ChatRunResult):
    pass
