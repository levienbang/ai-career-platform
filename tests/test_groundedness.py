import pytest

from app.config import Settings
from app.graph.builder import AgentRunResult, _render_grounded_answer
from app.tools.search_tool import SearchToolJob, SearchToolResult
from app.tools.sql_tool import SQLToolResult
from evaluation.evaluate_agent import grounded_rate
from evaluation.groundedness import (
    LLMFaithfulnessJudge,
    check_agent_result,
    check_groundedness,
    parse_number,
)


def _job(job_id: int, title: str, company: str | None, score: float = 0.8734) -> SearchToolJob:
    return SearchToolJob(
        job_id=job_id,
        score=score,
        title=title,
        company=company,
        location="Ho Chi Minh City",
        employment_type="Full-time",
        experience_years_min=2,
        required_skills=["Python", "PyTorch"],
        preferred_skills=["Docker"],
        evidence=f"Title: {title}\nDescription: 2 years of deep learning.",
    )


@pytest.fixture
def tool_results() -> tuple[SQLToolResult, SearchToolResult]:
    sql = SQLToolResult(
        sql="SELECT COUNT(*) FROM jobs LIMIT 100", columns=["count"], rows=[[1081]], row_count=1
    )
    search = SearchToolResult(
        query="AI jobs",
        jobs=[_job(17, "AI Engineer", "VinAI"), _job(31, "Computer Vision Engineer", "FPT")],
    )
    return sql, search


def test_rendered_agent_answer_is_grounded(tool_results) -> None:
    sql, search = tool_results
    state = {"route": "both", "sql_scope": "all", "sql_result": sql, "search_result": search}
    result = AgentRunResult(
        question="Tìm job AI và đếm tổng số job",
        route="both",
        sql_scope="all",
        answer=_render_grounded_answer(state),
        sql_result=sql,
        search_result=search,
        latency_ms=5,
    )

    report = check_agent_result(result, known_titles=["AI Engineer"], known_companies=["VinAI"])

    assert report.grounded, report.unsupported
    assert report.checked["job_id"] == 2
    assert report.checked["title"] >= 2
    assert report.checked["number"] > 0


def test_free_text_answer_with_rounding_and_separators_is_grounded(tool_results) -> None:
    sql, search = tool_results
    answer = (
        "Top 2 results:\n1. AI Engineer at VinAI (job 17, score 0.87, 2 years)\n"
        "2. Computer Vision Engineer at FPT (job_id: 31)\nTotal: 1,081 jobs."
    )
    report = check_groundedness(
        answer,
        question="Show the top 2 AI jobs",
        sql_result=sql,
        search_result=search,
        known_titles=["AI Engineer", "Computer Vision Engineer"],
        known_companies=["VinAI", "FPT Software", "FPT"],
    )
    assert report.grounded, report.unsupported


def test_fabricated_answer_is_flagged(tool_results) -> None:
    sql, search = tool_results
    answer = (
        "Job 999: Senior Rust Engineer at Acme Corp pays 5000 USD. "
        "There are 42 jobs in total. Also job 17 AI Engineer at VinAI."
    )
    report = check_groundedness(
        answer,
        question="Find AI jobs",
        sql_result=sql,
        search_result=search,
        known_titles=["Senior Rust Engineer", "AI Engineer"],
        known_companies=["Acme Corp", "VinAI"],
    )

    flagged = {(claim.kind, claim.value) for claim in report.unsupported}
    assert not report.grounded
    assert ("job_id", "999") in flagged
    assert ("title", "Senior Rust Engineer") in flagged
    assert ("company", "Acme Corp") in flagged
    assert ("number", "5000") in flagged
    assert ("number", "42") in flagged
    assert ("job_id", "17") not in flagged
    assert ("title", "AI Engineer") not in flagged


def test_json_job_with_swapped_attributes_or_unknown_job_is_flagged(tool_results) -> None:
    _, search = tool_results
    answer = (
        'Search result: [{"job_id": 17, "title": "Computer Vision Engineer", "company": "VinAI"},'
        ' {"job_id": 555, "title": "Data Scientist", "company": "Tiki"}]'
    )
    report = check_groundedness(answer, search_result=search)

    kinds = {(claim.kind, claim.value) for claim in report.unsupported}
    assert ("job_attribute", "17.title=Computer Vision Engineer") in kinds
    assert ("job_id", "555") in kinds
    assert ("title", "Data Scientist") in kinds
    assert ("company", "Tiki") in kinds
    assert ("number", "555") in kinds


def test_no_data_answers_have_nothing_to_flag() -> None:
    report = check_groundedness("No relevant data was found; no answer can be inferred.")
    assert report.grounded
    assert sum(report.checked.values()) == 0


def test_parse_number_keeps_both_readings_of_separators() -> None:
    assert [str(value) for value in parse_number("1,081")] == ["1081", "1.081"]
    assert [str(value) for value in parse_number("1.234.567")] == ["1234567"]
    assert [str(value) for value in parse_number("0,5")] == ["0.5"]


def test_grounded_rate_ignores_failed_cases() -> None:
    cases = [
        {"groundedness": {"grounded": True}},
        {"groundedness": {"grounded": False}},
        {"groundedness": None},
    ]
    assert grounded_rate(cases) == 0.5
    assert grounded_rate([{"groundedness": None}]) is None


class FakeRunnable:
    def __init__(self, output) -> None:
        self.output = output
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self.output


def test_llm_faithfulness_judge_scores_claims(tool_results) -> None:
    sql, search = tool_results
    result = AgentRunResult(
        question="AI jobs",
        route="search",
        answer="AI Engineer at VinAI; Rust job at Acme; 1081 jobs",
        sql_result=sql,
        search_result=search,
        latency_ms=1,
    )
    runnable = FakeRunnable(
        {
            "claims": [
                {"claim": "AI Engineer at VinAI", "supported": True},
                {"claim": "Rust job at Acme", "supported": False, "reason": "not in evidence"},
                {"claim": "1081 jobs", "supported": True},
            ]
        }
    )
    report = LLMFaithfulnessJudge(Settings(), runnable=runnable).evaluate(result)
    assert report.faithfulness == pytest.approx(2 / 3)
    prompt_text = "\n".join(message.content for message in runnable.messages.to_messages())
    assert "untrusted" in prompt_text and "VinAI" in prompt_text

    runnable.output = {"claims": []}
    assert LLMFaithfulnessJudge(Settings(), runnable=runnable).evaluate(result).faithfulness == 1.0
