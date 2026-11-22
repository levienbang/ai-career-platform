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
        "SELECT set_config('statement_timeout', '0', false) FROM jobs",
        "SELECT pg_read_file('/etc/passwd') FROM jobs",
        "SELECT version() FROM jobs",
        "SELECT COUNT(*) FROM jobs, generate_series(1, 100000000)",
        "SELECT id FROM jobs UNION SELECT id FROM pg_catalog.pg_tables",
        "SELECT id FROM jobs UNION SELECT pg_sleep(1) FROM jobs",
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
        SELECT DATE_TRUNC('month', ingested_at) AS month, COUNT(*) AS count
        FROM jobs
        GROUP BY DATE_TRUNC('month', ingested_at)
    )
    SELECT month, count FROM monthly ORDER BY month
    """

    validated = SQLSafetyValidator(max_rows=20).validate(sql)

    assert validated.endswith("LIMIT 20")


def test_sql_validator_rejects_column_from_wrong_table() -> None:
    with pytest.raises(SQLValidationError):
        SQLSafetyValidator().validate("SELECT title FROM skills")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) AS count FROM jobs WHERE posted_at >= CURRENT_DATE",
        "SELECT name FROM companies WHERE location = 'Ha Noi'",
    ],
)
def test_sql_validator_rejects_unpopulated_columns(sql: str) -> None:
    with pytest.raises(SQLValidationError):
        SQLSafetyValidator().validate(sql)


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


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(CASE WHEN lower(location) LIKE '%remote%' THEN 1 END) FROM jobs",
        "SELECT COUNT(*) FROM jobs AS j WHERE EXISTS "
        "(SELECT 1 FROM job_skills AS js WHERE js.job_id = j.id)",
        "SELECT trim(location), COUNT(*) FROM jobs GROUP BY trim(location)",
        "SELECT AVG(length(title)) FROM jobs",
        "SELECT GREATEST(experience_years_min, 0), ABS(experience_years_min) FROM jobs",
        "SELECT location, COUNT(*), RANK() OVER (ORDER BY COUNT(*) DESC) FROM jobs "
        "GROUP BY location",
        "SELECT title, ROW_NUMBER() OVER (PARTITION BY location ORDER BY ingested_at DESC) "
        "FROM jobs",
        "SELECT location, string_agg(title, ', ') FROM jobs GROUP BY location",
        "SELECT COUNT(*) FROM jobs WHERE ingested_at > now() - interval '7 days'",
        "SELECT COUNT(*) FROM jobs WHERE title ~* 'python'",
        "SELECT split_part(location, ',', 1), COUNT(*) FROM jobs GROUP BY 1",
        "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY experience_years_min) FROM jobs",
        "SELECT to_char(ingested_at, 'YYYY-MM'), COUNT(*) FROM jobs GROUP BY 1",
        "SELECT 'remote' AS k, COUNT(*) FROM jobs WHERE location ILIKE '%remote%' "
        "UNION ALL SELECT 'all', COUNT(*) FROM jobs",
    ],
)
def test_sql_validator_allows_common_read_only_analytics(sql: str) -> None:
    assert SQLSafetyValidator().validate(sql).endswith("LIMIT 100")


def test_union_is_capped_and_scoped_to_search_results() -> None:
    validated = SQLSafetyValidator(max_rows=10).validate(
        "SELECT id FROM jobs UNION SELECT job_id FROM job_skills LIMIT 500", job_ids=[3, 4]
    )

    assert validated.endswith("LIMIT 10")
    assert validated.count("IN (3, 4)") == 2
