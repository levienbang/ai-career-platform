from time import perf_counter
from typing import Annotated, NoReturn

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.session import get_db
from app.graph.builder import CareerAgent
from app.graph.router import (
    AgentRouterConfigurationError,
    AgentRoutingError,
    QuestionRouter,
    build_question_router,
)
from app.retrieval.embeddings import build_embedding_provider
from app.retrieval.factory import build_reranked_search_service
from app.retrieval.qdrant import build_qdrant_client
from app.retrieval.reranker import build_reranker
from app.schemas.agent import AgentResponse, AnalyticsResponse, NaturalLanguageQuery
from app.tools.search_tool import RerankedJobSearchTool
from app.tools.sql_tool import (
    SQLAnalyticsTool,
    SQLExecutionError,
    SQLGenerationError,
    SQLToolConfigurationError,
    SQLValidationError,
    build_sql_tool,
)
from app.tracing import build_agent_callbacks

router = APIRouter(tags=["agent"])


def get_sql_analytics_tool(
    settings: Annotated[Settings, Depends(get_settings)],
) -> SQLAnalyticsTool:
    try:
        return build_sql_tool(settings)
    except SQLToolConfigurationError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error


def get_question_router(
    settings: Annotated[Settings, Depends(get_settings)],
) -> QuestionRouter:
    try:
        return build_question_router(settings)
    except AgentRouterConfigurationError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error


def _raise_sql_http_error(error: Exception) -> NoReturn:
    if isinstance(error, SQLValidationError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    if isinstance(error, (SQLToolConfigurationError, SQLGenerationError, SQLExecutionError)):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error
    raise error


class _LazySQLTool:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def invoke(self, question: str, *, job_ids: list[int] | None = None):
        return build_sql_tool(self.settings).invoke(question, job_ids=job_ids)


class _LazyRerankedSearchTool:
    def __init__(self, db: Session, settings: Settings) -> None:
        self.db = db
        self.settings = settings

    def invoke(self, query: str, limit: int = 5):
        embedder = build_embedding_provider(self.settings)
        client = build_qdrant_client(self.settings.qdrant_url, self.settings.qdrant_timeout_seconds)
        reranker = build_reranker(self.settings)
        backend = build_reranked_search_service(self.db, embedder, client, reranker, self.settings)
        return RerankedJobSearchTool(backend).invoke(query, limit)


def get_career_agent(
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    question_router: Annotated[QuestionRouter, Depends(get_question_router)],
) -> CareerAgent:
    return CareerAgent(
        question_router,
        _LazySQLTool(settings),
        _LazyRerankedSearchTool(db, settings),
        search_limit=settings.agent_search_limit,
        tool_max_retries=settings.agent_tool_max_retries,
        callbacks=build_agent_callbacks(settings),
    )


@router.post("/analytics/query", response_model=AnalyticsResponse)
def analytics_query(
    request: NaturalLanguageQuery,
    sql_tool: Annotated[SQLAnalyticsTool, Depends(get_sql_analytics_tool)],
) -> AnalyticsResponse:
    started_at = perf_counter()
    try:
        result = sql_tool.invoke(request.question)
    except Exception as error:
        _raise_sql_http_error(error)
    return AnalyticsResponse(
        question=request.question,
        result=result,
        latency_ms=(perf_counter() - started_at) * 1000,
    )


@router.post("/agent/query", response_model=AgentResponse)
def agent_query(
    request: NaturalLanguageQuery,
    agent: Annotated[CareerAgent, Depends(get_career_agent)],
) -> AgentResponse:
    try:
        return AgentResponse(**agent.invoke(request.question).model_dump())
    except AgentRoutingError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error
