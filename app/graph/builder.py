import json
from collections.abc import Callable
from time import perf_counter
from typing import Any, Literal, Protocol

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.graph.router import QuestionRouter, RouteDecision
from app.graph.state import AgentRoute, AgentState, SQLScope
from app.tools.search_tool import SearchToolResult
from app.tools.sql_tool import SQLToolResult, SQLValidationError


class SQLToolLike(Protocol):
    def invoke(self, question: str, *, job_ids: list[int] | None = None) -> SQLToolResult: ...


class SearchToolLike(Protocol):
    def invoke(self, query: str, limit: int = 5) -> SearchToolResult: ...


class AgentRunResult(BaseModel):
    question: str
    route: AgentRoute
    sql_scope: SQLScope | None = None
    answer: str
    sql_result: SQLToolResult | None = None
    search_result: SearchToolResult | None = None
    errors: list[str] = Field(default_factory=list)
    latency_ms: float = Field(ge=0)


def _invoke_with_retry[ResultT: (SQLToolResult, SearchToolResult)](
    operation: Callable[[], ResultT],
    *,
    max_retries: int,
    non_retryable: tuple[type[Exception], ...] = (),
) -> tuple[ResultT | None, Exception | None, int]:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return operation(), None, attempt
        except non_retryable as error:
            return None, error, attempt
        except Exception as error:
            last_error = error
    return None, last_error, max_retries


def _render_grounded_answer(state: AgentState) -> str:
    if state.get("errors"):
        return "Unable to answer because a required tool failed."
    if (
        state.get("route") == "both"
        and state.get("sql_scope") == "search_results"
        and (
            not state.get("search_result")
            or not state["search_result"].jobs
            or not state.get("sql_result")
            or not state["sql_result"].rows
        )
    ):
        return "No relevant data was found; no answer can be inferred."
    evidence: list[str] = []
    sql_result = state.get("sql_result")
    if sql_result and sql_result.rows:
        evidence.append(
            "SQL result: "
            + json.dumps(
                {"columns": sql_result.columns, "rows": sql_result.rows}, ensure_ascii=False
            )
        )
    search_result = state.get("search_result")
    if search_result and search_result.jobs:
        jobs = [
            {
                "job_id": job.job_id,
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "employment_type": job.employment_type,
                "experience_years_min": job.experience_years_min,
                "required_skills": job.required_skills,
                "preferred_skills": job.preferred_skills,
            }
            for job in search_result.jobs
        ]
        evidence.append("Search result: " + json.dumps(jobs, ensure_ascii=False))
    elif state.get("route") == "both" and state.get("sql_scope") == "all":
        evidence.append("Search result: []")
    if evidence:
        return "\n".join(evidence)
    return "No relevant data was found; no answer can be inferred."


class CareerAgent:
    def __init__(
        self,
        router: QuestionRouter,
        sql_tool: SQLToolLike,
        search_tool: SearchToolLike,
        *,
        search_limit: int = 5,
        tool_max_retries: int = 1,
        callbacks: list[BaseCallbackHandler] | None = None,
    ) -> None:
        self.router = router
        self.sql_tool = sql_tool
        self.search_tool = search_tool
        self.search_limit = search_limit
        self.tool_max_retries = tool_max_retries
        self.callbacks = callbacks or []
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)

        def route_node(state: AgentState) -> AgentState:
            decision = RouteDecision.model_validate(self.router.decide(state["question"]))
            return {
                "route": decision.route,
                "sql_scope": decision.sql_scope,
                "errors": [],
            }

        def sql_node(state: AgentState) -> AgentState:
            job_ids = None
            if state["route"] == "both" and state["sql_scope"] == "search_results":
                job_ids = [job.job_id for job in state["search_result"].jobs]
            result, error, retries = _invoke_with_retry(
                lambda: (
                    self.sql_tool.invoke(state["question"], job_ids=job_ids)
                    if job_ids is not None
                    else self.sql_tool.invoke(state["question"])
                ),
                max_retries=self.tool_max_retries,
                non_retryable=(SQLValidationError,),
            )
            update: AgentState = {"retry_count": state.get("retry_count", 0) + retries}
            if result is not None:
                update["sql_result"] = result
            if error is not None:
                update["errors"] = [*state.get("errors", []), f"sql: {error}"]
            return update

        def search_node(state: AgentState) -> AgentState:
            result, error, retries = _invoke_with_retry(
                lambda: self.search_tool.invoke(state["question"], limit=self.search_limit),
                max_retries=self.tool_max_retries,
            )
            update: AgentState = {"retry_count": state.get("retry_count", 0) + retries}
            if result is not None:
                update["search_result"] = result
            if error is not None:
                update["errors"] = [*state.get("errors", []), f"search: {error}"]
            return update

        def answer_node(state: AgentState) -> AgentState:
            return {"answer": _render_grounded_answer(state)}

        def after_route(state: AgentState) -> Literal["sql", "search"]:
            return "sql" if state["route"] == "sql" else "search"

        def after_search(state: AgentState) -> Literal["sql", "answer"]:
            search_result = state.get("search_result")
            if state["route"] == "both" and (
                state["sql_scope"] == "all" or search_result and search_result.jobs
            ):
                return "sql"
            return "answer"

        builder.add_node("route", route_node)
        builder.add_node("sql", sql_node)
        builder.add_node("search", search_node)
        builder.add_node("answer", answer_node)
        builder.add_edge(START, "route")
        builder.add_conditional_edges("route", after_route)
        builder.add_edge("sql", "answer")
        builder.add_conditional_edges("search", after_search)
        builder.add_edge("answer", END)
        return builder.compile()

    def invoke(self, question: str) -> AgentRunResult:
        started_at = perf_counter()
        state: dict[str, Any] = self.graph.invoke(
            {"question": question, "errors": [], "retry_count": 0},
            config={"recursion_limit": 8, "callbacks": self.callbacks},
        )
        return AgentRunResult(
            question=question,
            route=state["route"],
            sql_scope=state["sql_scope"],
            answer=state["answer"],
            sql_result=state.get("sql_result"),
            search_result=state.get("search_result"),
            errors=state.get("errors", []),
            latency_ms=(perf_counter() - started_at) * 1000,
        )
