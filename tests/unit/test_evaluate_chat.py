"""Scoring of the chat eval, with a scripted model (no API calls)."""

import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.graph.chatbot import CareerChatbot
from evaluation.evaluate_chat import CASES_FILE, plain, rewrite_problems, score_chat


def tool_reply(question: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "query_job_data", "args": {"question": question}, "id": "c1"}],
    )


def scripted(*replies) -> CareerChatbot:
    queue = list(replies)
    model = RunnableLambda(lambda messages: queue.pop(0))
    return CareerChatbot(None, model, model)


FOLLOW_UP = {
    "id": "follow",
    "history": [
        {"role": "user", "content": "có bao nhiêu job AI ở Hà Nội?"},
        {"role": "assistant", "content": "Có 20 job."},
    ],
    "message": "còn ở HCM thì sao?",
    "expect_tool": True,
    "must_contain": [["ai"], ["ho chi minh", "hcm"]],
    "must_not_contain": ["ha noi"],
}
GREETING = {"id": "hi", "message": "chào bạn", "expect_tool": False}


def test_plain_drops_vietnamese_accents() -> None:
    assert plain("Hồ Chí Minh, Đà Nẵng") == "ho chi minh, da nang"


def test_rewrite_problems_lists_missing_and_banned_words() -> None:
    assert rewrite_problems("Có bao nhiêu job AI ở Hồ Chí Minh?", FOLLOW_UP) == []
    assert rewrite_problems("Còn ở HCM thì sao?", FOLLOW_UP) == ["missing ai"]
    assert rewrite_problems("Job AI ở HCM và Hà Nội", FOLLOW_UP) == ["contains ha noi"]


def test_score_chat_counts_tool_decisions_and_rewrites() -> None:
    model = scripted(
        AIMessage("Chào bạn!"),
        tool_reply("Có bao nhiêu job AI ở Hồ Chí Minh?"),
        tool_reply("job ở HCM"),  # lost "AI" from the earlier turn
        tool_reply("chào"),  # a greeting should not call the tool
    )

    result = score_chat(model, [GREETING, FOLLOW_UP, FOLLOW_UP, GREETING])

    assert result["tool_decision_accuracy"] == 0.75
    assert result["rewrite_accuracy"] == 0.5
    assert result["all_correct"] == 0.5
    assert result["cases"][2]["problems"] == ["missing ai"]


def test_model_errors_are_recorded() -> None:
    def fail(messages):
        raise TimeoutError

    result = score_chat(CareerChatbot(None, RunnableLambda(fail), None), [GREETING])

    assert result["cases"][0]["error"] == "TimeoutError"
    assert result["tool_decision_accuracy"] == 0


def test_case_file_is_well_formed() -> None:
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert case["expect_tool"] == bool(case.get("must_contain"))


def test_retry_after_made_up_numbers_is_scored_separately() -> None:
    chatbot = scripted(AIMessage("Có 2 job Golang."), tool_reply("job AI ở Hồ Chí Minh"))

    result = score_chat(chatbot, [FOLLOW_UP])

    row = result["cases"][0]
    assert row["retried"] and row["tool_correct"]
    assert row["first_reply_tool_correct"] is False
    assert row["first_reply"] == "Có 2 job Golang."
    assert result["retries"] == 1
