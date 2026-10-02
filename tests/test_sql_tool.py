import pytest

from app.tools.sql_tool import (
    SQLAnalyticsTool,
    SQLSafetyValidator,
    SQLToolResult,
    SQLValidationError,
)


class FakeSQLGenerator:
    def __init__(self, sql: str) -> None:
        self.sql = sql

    def generate(self, question: str) -> str:
        assert question
        return self.sql


class RecordingExecutor:
    def __init__(self) -> None:
        self.sql = ""

    def execute(self, sql: str) -> SQLToolResult:
        self.sql = sql
        return SQLToolResult(sql=sql, columns=["count"], rows=[[2]], row_count=1)


def test_sql_tool_validates_before_execution() -> None:
    executor = RecordingExecutor()
    tool = SQLAnalyticsTool(
        FakeSQLGenerator("SELECT COUNT(*) AS count FROM jobs"),
        SQLSafetyValidator(max_rows=25),
        executor,
    )

    result = tool.invoke("Có bao nhiêu job?")

    assert result.rows == [[2]]
    assert executor.sql == "SELECT COUNT(*) AS count FROM jobs LIMIT 25"


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM jobs",
        "UPDATE jobs SET title = 'x'",
        "DROP TABLE jobs",
        "SELECT id FROM jobs; SELECT id FROM jobs",
        "SELECT id FROM pg_catalog.pg_tables",
        "WITH jobs AS (SELECT id FROM jobs) SELECT id FROM pg_catalog.jobs",
        "WITH jobs AS (SELECT id FROM jobs) SELECT skills.id FROM jobs JOIN skills ON true",
        "SELECT description FROM jobs",
        "SELECT pg_sleep(10) FROM jobs",
        "SELECT * FROM jobs",
        "SELECT id FROM jobs FOR UPDATE",
        "WITH removed AS (DELETE FROM jobs RETURNING id) SELECT id FROM removed",
    ],
)
def test_sql_validator_rejects_unsafe_queries(sql: str) -> None:
    with pytest.raises(SQLValidationError):
        SQLSafetyValidator().validate(sql)


def test_sql_validator_allows_join_aggregate_and_caps_limit() -> None:
    sql = """
    SELECT s.canonical_name, COUNT(js.job_id) AS job_count
    FROM skills AS s
    JOIN job_skills AS js ON js.skill_id = s.id
    GROUP BY s.canonical_name
    ORDER BY job_count DESC
    LIMIT 999
    """

    validated = SQLSafetyValidator(max_rows=10).validate(sql)

    assert validated.endswith("LIMIT 10")
    assert "skills AS s" in validated


def test_sql_validator_allows_left_join_for_jobs_without_company() -> None:
    sql = (
        "SELECT COUNT(j.id) AS count FROM jobs AS j "
        "LEFT JOIN companies AS c ON c.id = j.company_id WHERE c.id IS NULL"
    )
    validated = SQLSafetyValidator().validate(sql)
    assert "LEFT JOIN companies AS c" in validated


def test_sql_validator_allows_read_only_cte_and_safe_date_function() -> None:
    sql = """
    WITH monthly AS (
        SELECT DATE_TRUNC('month', posted_at) AS month, COUNT(*) AS count
        FROM jobs
        GROUP BY DATE_TRUNC('month', posted_at)
    )
    SELECT month, count FROM monthly ORDER BY month
    """

    validated = SQLSafetyValidator(max_rows=20).validate(sql)

    assert validated.endswith("LIMIT 20")


def test_sql_validator_rejects_column_from_wrong_table() -> None:
    with pytest.raises(SQLValidationError):
        SQLSafetyValidator().validate("SELECT title FROM skills")


def test_sql_validator_rejects_ambiguous_column() -> None:
    with pytest.raises(SQLValidationError):
        SQLSafetyValidator().validate("SELECT id FROM jobs JOIN skills ON jobs.id = skills.id")


def test_sql_tool_scopes_both_route_to_retrieved_job_ids() -> None:
    executor = RecordingExecutor()
    tool = SQLAnalyticsTool(
        FakeSQLGenerator(
            "SELECT s.canonical_name, COUNT(js.job_id) AS count "
            "FROM skills s JOIN job_skills js ON js.skill_id = s.id "
            "GROUP BY s.canonical_name"
        ),
        SQLSafetyValidator(max_rows=25),
        executor,
    )

    tool.invoke("most common skills", job_ids=[7, 2, 7])

    assert "job_id IN (2, 7)" in executor.sql
    assert "SELECT id, canonical_name" not in executor.sql
    assert "id IN (SELECT skill_id FROM job_skills WHERE job_id IN (2, 7))" in executor.sql


@pytest.mark.parametrize("job_ids", [[], [0], [-1], [True]])
def test_sql_validator_rejects_invalid_search_job_ids(job_ids: list[int]) -> None:
    with pytest.raises(SQLValidationError):
        SQLSafetyValidator().validate("SELECT COUNT(*) FROM jobs", job_ids=job_ids)


def test_sql_langchain_wrapper_invokes_same_tool() -> None:
    tool = SQLAnalyticsTool(
        FakeSQLGenerator("SELECT COUNT(*) AS count FROM jobs"),
        SQLSafetyValidator(),
        RecordingExecutor(),
    ).as_langchain_tool()

    result = tool.invoke({"question": "count jobs"})

    assert result.row_count == 1


def test_sql_validator_allows_and_or_conditions() -> None:
    sql = (
        "SELECT count(j.id) FROM jobs AS j "
        "WHERE (j.location ILIKE '%hồ chí minh%' AND j.title ILIKE '%software%') "
        "OR j.title ILIKE '%developer%'"
    )
    validated = SQLSafetyValidator().validate(sql)
    assert " AND " in validated and " OR " in validated


def test_sql_validator_still_rejects_unsafe_function_inside_and() -> None:
    with pytest.raises(SQLValidationError, match="is not allowed"):
        SQLSafetyValidator().validate(
            "SELECT j.id FROM jobs AS j WHERE j.id > 0 AND pg_sleep(1) IS NULL"
        )
