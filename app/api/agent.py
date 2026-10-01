from time import perf_counter
from typing import Annotated, NoReturn

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.session import get_db
from app.graph.builder import CareerAgent
from app.graph.factory import build_career_agent
from app.graph.router import (
    AgentRouterConfigurationError,
    AgentRoutingError,
    QuestionRouter,
    build_question_router,
)
from app.schemas.agent import AgentResponse, AnalyticsResponse, NaturalLanguageQuery
from app.tools.sql_tool import (
    SQLAnalyticsTool,
    SQLExecutionError,
    SQLGenerationError,
    SQLToolConfigurationError,
    SQLValidationError,
    build_sql_tool,
)

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


def get_career_agent(
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    question_router: Annotated[QuestionRouter, Depends(get_question_router)],
) -> CareerAgent:
    return build_career_agent(db, settings, question_router)


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
