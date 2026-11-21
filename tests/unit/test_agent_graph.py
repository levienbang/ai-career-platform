from app.graph.builder import CareerAgent
from app.graph.router import RouteDecision
from app.graph.state import AgentRoute, SQLScope
from app.retrieval.embeddings import EmbeddingConfigurationError
from app.tools.search_tool import SearchToolJob, SearchToolResult
from app.tools.sql_tool import SQLToolConfigurationError, SQLToolResult


class FakeRouter:
    def __init__(self, route: AgentRoute) -> None:
        self.selected_route = route

    def decide(self, question: str) -> RouteDecision:
        assert question
        return RouteDecision(
            route=self.selected_route,
            sql_scope="search_results" if self.selected_route == "both" else "all",
        )


class ScopedFakeRouter(FakeRouter):
    def __init__(self, scope: SQLScope) -> None:
        super().__init__("both")
        self.scope = scope

    def decide(self, question: str) -> RouteDecision:
        assert question
        return RouteDecision(route="both", sql_scope=self.scope)


class FakeSQLTool:
    def __init__(self, *, fail: bool = False, empty: bool = False) -> None:
        self.calls = 0
        self.fail = fail
        self.empty = empty
        self.job_ids: list[int] | None = None

    def invoke(self, question: str, *, job_ids: list[int] | None = None) -> SQLToolResult:
        self.calls += 1
        self.job_ids = job_ids
        if self.fail:
            raise RuntimeError("database unavailable")
        rows = [] if self.empty else [[3]]
        return SQLToolResult(
            sql="SELECT COUNT(*) AS count FROM jobs LIMIT 100",
            columns=["count"],
            rows=rows,
            row_count=len(rows),
        )


class FakeSearchTool:
    def __init__(self, *, fail: bool = False, empty: bool = False) -> None:
        self.calls = 0
        self.fail = fail
        self.empty = empty

    def invoke(self, query: str, limit: int = 5) -> SearchToolResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("qdrant unavailable")
        jobs = []
        if not self.empty:
            jobs = [
                SearchToolJob(
                    job_id=1,
                    score=0.8,
                    title="AI Intern",
                    company="Example",
                    location="Hanoi",
                    employment_type="Internship",
                    experience_years_min=0,
                    required_skills=["Python"],
                    preferred_skills=[],
                    evidence="Title: AI Intern; required skills: Python",
                )
            ]
        return SearchToolResult(query=query, jobs=jobs)


def test_graph_routes_to_search_only() -> None:
    sql_tool = FakeSQLTool()
    search_tool = FakeSearchTool()

    result = CareerAgent(FakeRouter("search"), sql_tool, search_tool).invoke("find jobs")

    assert result.route == "search"
    assert sql_tool.calls == 0
    assert search_tool.calls == 1
    assert "AI Intern" in result.answer


def test_graph_routes_to_sql_only() -> None:
    sql_tool = FakeSQLTool()
    search_tool = FakeSearchTool()

    result = CareerAgent(FakeRouter("sql"), sql_tool, search_tool).invoke("count jobs")

    assert result.route == "sql"
    assert sql_tool.calls == 1
    assert search_tool.calls == 0
    assert '"rows": [[3]]' in result.answer


def test_graph_routes_to_both_tools() -> None:
    sql_tool = FakeSQLTool()
    search_tool = FakeSearchTool()

    result = CareerAgent(FakeRouter("both"), sql_tool, search_tool).invoke("find and count")

    assert sql_tool.calls == 1
    assert sql_tool.job_ids == [1]
    assert search_tool.calls == 1
    assert result.sql_result is not None
    assert result.search_result is not None


def test_combined_route_does_not_query_all_jobs_when_search_is_empty() -> None:
    sql_tool = FakeSQLTool()
    result = CareerAgent(FakeRouter("both"), sql_tool, FakeSearchTool(empty=True)).invoke(
        "find and count"
    )

    assert sql_tool.calls == 0
    assert result.answer == "No relevant data was found; no answer can be inferred."


def test_combined_global_count_is_not_limited_to_search_hits() -> None:
    sql_tool = FakeSQLTool()
    result = CareerAgent(ScopedFakeRouter("all"), sql_tool, FakeSearchTool(empty=True)).invoke(
        "find remote jobs and count all remote jobs in the dataset"
    )

    assert sql_tool.calls == 1
    assert sql_tool.job_ids is None
    assert result.sql_scope == "all"
    assert "Search result: []" in result.answer
    assert '"rows": [[3]]' in result.answer


def test_combined_route_does_not_return_partial_answer_on_sql_error() -> None:
    result = CareerAgent(FakeRouter("both"), FakeSQLTool(fail=True), FakeSearchTool()).invoke(
        "find and count"
    )

    assert result.search_result is not None
    assert "AI Intern" not in result.answer
    assert result.errors


def test_tool_error_is_bounded_and_answer_does_not_invent_data() -> None:
    result = CareerAgent(
        FakeRouter("search"), FakeSQLTool(), FakeSearchTool(fail=True), tool_max_retries=1
    ).invoke("find jobs")

    assert result.errors == ["search: qdrant unavailable"]
    assert result.search_result is None
    assert result.answer == "Unable to answer because a required tool failed."


def test_no_data_returns_explicit_grounded_fallback() -> None:
    result = CareerAgent(FakeRouter("sql"), FakeSQLTool(empty=True), FakeSearchTool()).invoke(
        "count missing"
    )

    assert result.errors == []
    assert result.answer == "No relevant data was found; no answer can be inferred."


class ConfigErrorSQLTool(FakeSQLTool):
    def invoke(self, question: str, *, job_ids: list[int] | None = None) -> SQLToolResult:
        self.calls += 1
        raise SQLToolConfigurationError("READONLY_DATABASE_URL must be configured")


class ConfigErrorSearchTool(FakeSearchTool):
    def invoke(self, query: str, limit: int = 5) -> SearchToolResult:
        self.calls += 1
        raise EmbeddingConfigurationError("EMBEDDING_API_KEY must be configured")


def test_single_tool_route_ignores_search_result_scope() -> None:
    sql_tool = FakeSQLTool()
    router = FakeRouter("sql")
    router.decide = lambda question: RouteDecision(route="sql", sql_scope="search_results")

    result = CareerAgent(router, sql_tool, FakeSearchTool()).invoke("count jobs")

    assert result.sql_scope == "all"
    assert sql_tool.job_ids is None


def test_configuration_errors_are_not_retried() -> None:
    sql_tool = ConfigErrorSQLTool()
    search_tool = ConfigErrorSearchTool()

    sql_result = CareerAgent(
        FakeRouter("sql"), sql_tool, FakeSearchTool(), tool_max_retries=2
    ).invoke("count jobs")
    search_result = CareerAgent(
        FakeRouter("search"), FakeSQLTool(), search_tool, tool_max_retries=2
    ).invoke("find jobs")

    assert sql_tool.calls == 1
    assert search_tool.calls == 1
    assert sql_result.errors and search_result.errors
