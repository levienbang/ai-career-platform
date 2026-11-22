"""The chat layer over the career agent, with scripted fake models and a fake agent."""

import json

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from app.graph.builder import AgentRunResult
from app.graph.chatbot import (
    CareerChatbot,
    ChatTurn,
    evidence_json,
    history_messages,
    ungrounded_job_ids,
)
from app.tools.search_tool import SearchToolJob, SearchToolResult
from app.tools.sql_tool import SQLToolResult


def _job(job_id: int, title: str, company: str) -> SearchToolJob:
    return SearchToolJob(
        job_id=job_id,
        score=0.9,
        title=title,
        company=company,
        location="Hà Nội",
        employment_type="Full-time",
        experience_years_min=0,
        required_skills=["Python", "PyTorch"],
        preferred_skills=[],
        evidence="long description " * 50,
    )


def search_run(question: str) -> AgentRunResult:
    return AgentRunResult(
        question=question,
        route="search",
        answer="Search result: template",
        search_result=SearchToolResult(
            query=question, jobs=[_job(12, "AI Engineer", "FPT"), _job(30, "ML Engineer", "VNG")]
        ),
        latency_ms=5,
    )


def sql_run(question: str) -> AgentRunResult:
    return AgentRunResult(
        question=question,
        route="sql",
        answer="SQL result: template",
        sql_result=SQLToolResult(
            sql="SELECT COUNT(*) FROM jobs LIMIT 100", columns=["count"], rows=[[37]], row_count=1
        ),
        latency_ms=5,
    )


class FakeAgent:
    def __init__(self, run=search_run) -> None:
        self.questions: list[str] = []
        self.run = run

    def invoke(self, question: str) -> AgentRunResult:
        self.questions.append(question)
        return self.run(question)


class ScriptedModel:
    """Returns the scripted replies in order and records what it was sent."""

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.inputs: list[list] = []

    def runnable(self):
        def run(messages):
            self.inputs.append(list(messages))
            reply = self.replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

        return RunnableLambda(run)


def tool_call(question: str, call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "query_job_data", "args": {"question": question}, "id": call_id}],
    )


def chatbot(model: ScriptedModel, agent=None, answer_model: ScriptedModel | None = None, **kw):
    answer_model = answer_model or ScriptedModel()
    return CareerChatbot(agent or FakeAgent(), model.runnable(), answer_model.runnable(), **kw)


def test_greeting_is_answered_without_the_agent() -> None:
    agent = FakeAgent()
    model = ScriptedModel(AIMessage("Chào bạn! Mình giúp tìm job và thống kê thị trường."))

    result = chatbot(model, agent).invoke("chào bạn")

    assert result.answer.startswith("Chào bạn")
    assert agent.questions == []
    assert result.tool_runs == []
    assert result.llm_calls == 1
    assert isinstance(model.inputs[0][0], SystemMessage)


def test_data_question_calls_the_agent_and_answers_from_its_result() -> None:
    agent = FakeAgent()
    model = ScriptedModel(
        tool_call("Tìm job AI cho fresher ở Hà Nội"),
        AIMessage("Hợp nhất là job 12 (AI Engineer) ở FPT."),
    )

    result = chatbot(model, agent).invoke("có job AI nào cho fresher ở HN không?")

    assert agent.questions == ["Tìm job AI cho fresher ở Hà Nội"]
    assert result.answer == "Hợp nhất là job 12 (AI Engineer) ở FPT."
    assert [run.route for run in result.tool_runs] == ["search"]
    assert result.warnings == []
    assert result.llm_calls == 2
    tool_message = model.inputs[1][-1]
    assert isinstance(tool_message, ToolMessage)
    evidence = json.loads(tool_message.content)
    assert [job["job_id"] for job in evidence["jobs"]] == [12, 30]
    assert "evidence" not in evidence["jobs"][0]  # long documents stay out of the prompt


def test_job_ids_missing_from_the_results_are_flagged() -> None:
    model = ScriptedModel(tool_call("job AI"), AIMessage("Xem job 12 và job 99."))

    result = chatbot(model).invoke("job AI?")

    assert result.warnings == ["Câu trả lời nhắc job không có trong dữ liệu tra được: 99"]


def test_tool_calls_are_capped_then_the_answer_model_replies() -> None:
    agent = FakeAgent()
    model = ScriptedModel(tool_call("q1", "a"), tool_call("q2", "b"))
    answer_model = ScriptedModel(AIMessage("Tóm tắt từ hai lần tra."))

    result = chatbot(model, agent, answer_model, max_tool_calls=2).invoke("so sánh")

    assert agent.questions == ["q1", "q2"]
    assert result.answer == "Tóm tắt từ hai lần tra."
    assert len(answer_model.inputs) == 1


def test_agent_failure_is_reported_to_the_model() -> None:
    class BrokenAgent:
        def invoke(self, question):
            raise TimeoutError("db")

    model = ScriptedModel(tool_call("đếm job"), AIMessage("Hiện chưa tra được dữ liệu."))

    result = chatbot(model, BrokenAgent()).invoke("có bao nhiêu job?")

    assert "TimeoutError" in model.inputs[1][-1].content
    assert result.answer == "Hiện chưa tra được dữ liệu."
    assert result.tool_runs == []


def test_model_failure_after_the_tool_returns_the_agent_answer() -> None:
    model = ScriptedModel(tool_call("đếm job"), RuntimeError("chat model down"))

    result = chatbot(model, FakeAgent(sql_run)).invoke("có bao nhiêu job?")

    assert result.answer == "SQL result: template"
    assert result.warnings == ["Mô hình chat lỗi; hiển thị kết quả tra cứu gốc."]


def test_model_failure_before_any_tool_raises() -> None:
    with pytest.raises(RuntimeError):
        chatbot(ScriptedModel(RuntimeError("down"))).invoke("chào")


def test_history_keeps_the_newest_turns_within_the_token_budget() -> None:
    history = [ChatTurn(role="user", content="a" * 3000) for _ in range(5)]
    history.append(ChatTurn(role="assistant", content="b" * 8000))

    messages, dropped = history_messages(history, budget_tokens=4000)

    # 8000 chars ~ 2671 tokens, 3000 chars ~ 1004 tokens: the reply and one question fit.
    assert [type(m).__name__ for m in messages] == ["HumanMessage", "AIMessage"]
    assert dropped == 4


def test_context_size_comes_from_the_provider_usage() -> None:
    first = tool_call("đếm job")
    first.usage_metadata = {"input_tokens": 1200, "output_tokens": 30, "total_tokens": 1230}
    final = AIMessage("Có 37 job.")
    final.usage_metadata = {"input_tokens": 1500, "output_tokens": 20, "total_tokens": 1520}

    result = chatbot(ScriptedModel(first, final), FakeAgent(sql_run)).invoke("bao nhiêu job?")

    assert result.context_tokens == 1500  # the last call's prompt is the context size
    assert result.tokens_used == 2750
    assert result.context_limit == 100_000
    assert result.history_dropped == 0


def test_follow_up_sees_history_but_must_call_the_tool_again() -> None:
    agent = FakeAgent(sql_run)
    model = ScriptedModel(tool_call("Có bao nhiêu job AI ở Hồ Chí Minh?"), AIMessage("Có 37 job."))
    history = [
        ChatTurn(role="user", content="có bao nhiêu job AI ở Hà Nội?"),
        ChatTurn(role="assistant", content="Có 20 job."),
    ]

    result = chatbot(model, agent).invoke("còn ở HCM thì sao?", history)

    assert agent.questions == ["Có bao nhiêu job AI ở Hồ Chí Minh?"]
    sent = model.inputs[0]
    assert [type(m).__name__ for m in sent] == [
        "SystemMessage",
        "HumanMessage",
        "AIMessage",
        "HumanMessage",
    ]
    assert result.answer == "Có 37 job."


def test_numbers_from_sql_rows_count_as_known_ids() -> None:
    assert ungrounded_job_ids("job 37", [sql_run("x")]) == []
    assert ungrounded_job_ids("không có ID nào", []) == []


def test_evidence_keeps_sql_rows() -> None:
    evidence = json.loads(evidence_json(sql_run("đếm")))

    assert evidence == {
        "route": "sql",
        "sql": {"columns": ["count"], "rows": [[37]], "row_count": 1},
    }
