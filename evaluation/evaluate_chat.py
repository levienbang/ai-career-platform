"""The chat layer's two own decisions, on chat_cases.json, with no LLM judge.

1. Tool or not: greetings, thanks, capability and off-topic messages must not call
   query_job_data; data questions must.
2. Standalone rewrite: the tool question must carry what the follow-up refers to
   ("còn ở HCM thì sao?" -> AI + Hồ Chí Minh), checked by keyword groups.

Only the chatbot's first model step runs (CHAT_PROVIDER, Qwen by default), including
its retry when a reply states numbers before any lookup: one or two chat calls per case
and no agent (DeepSeek) calls. ``first_reply_*`` scores the reply before that retry.
Keywords are matched without accents.
"""

import json
import unicodedata
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.config import Settings
from app.graph.chatbot import SYSTEM_PROMPT, CareerChatbot, ChatTurn, history_messages
from evaluation.metrics import percentile

CASES_FILE = Path(__file__).with_name("chat_cases.json")
RESULTS_FILE = Path(__file__).with_name("chat_results.json")


def plain(text: str) -> str:
    """Lowercase without Vietnamese accents, so "Hồ Chí Minh" matches "ho chi minh"."""
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    return "".join(char for char in text if unicodedata.category(char) != "Mn")


def rewrite_problems(question: str, case: dict) -> list[str]:
    """Keyword groups the rewritten question misses, and banned words it contains."""
    text = plain(question)
    problems = [
        "missing " + "|".join(group)
        for group in case.get("must_contain", [])
        if not any(plain(word) in text for word in group)
    ]
    problems += [
        f"contains {word}" for word in case.get("must_not_contain", []) if plain(word) in text
    ]
    return problems


def _tool_questions(reply: BaseMessage | None) -> list[str]:
    calls = getattr(reply, "tool_calls", None) or []
    return [str(call.get("args", {}).get("question", "")) for call in calls]


def score_case(chatbot: CareerChatbot, case: dict) -> dict[str, object]:
    history = [ChatTurn.model_validate(turn) for turn in case.get("history", [])]
    earlier, _ = history_messages(history)
    messages = [SystemMessage(SYSTEM_PROMPT), *earlier, HumanMessage(case["message"])]
    started = perf_counter()
    reply: BaseMessage | None = None
    received: list[BaseMessage] = []
    try:
        reply, received = chatbot.model_turn(messages)
        error = None
    except Exception as exc:  # report, do not stop the whole evaluation
        error = type(exc).__name__
    latency_ms = round((perf_counter() - started) * 1000)
    questions = _tool_questions(reply)
    tool_correct = error is None and bool(questions) == case["expect_tool"]
    first_called = bool(_tool_questions(received[0])) if received else False
    problems = [problem for question in questions for problem in rewrite_problems(question, case)]
    if case["expect_tool"] and questions and not problems:
        rewrite_correct: bool | None = True
    elif case["expect_tool"] and questions:
        rewrite_correct = False
    else:
        rewrite_correct = None  # no tool expected, or none called (already a tool miss)
    return {
        "id": case["id"],
        "message": case["message"],
        "expect_tool": case["expect_tool"],
        "called_tool": bool(questions),
        "tool_questions": questions,
        "reply": None if questions or reply is None else reply.text[:300],
        "first_reply": None if first_called or not received else received[0].text[:300],
        "retried": len(received) > 1,
        "first_reply_tool_correct": error is None and first_called == case["expect_tool"],
        "tool_correct": tool_correct,
        "rewrite_correct": rewrite_correct,
        "problems": problems,
        "latency_ms": latency_ms,
        "error": error,
    }


def score_chat(chatbot: CareerChatbot, cases: list[dict]) -> dict[str, object]:
    rows = [score_case(chatbot, case) for case in cases]
    rewrites = [row["rewrite_correct"] for row in rows if row["rewrite_correct"] is not None]
    latencies = [row["latency_ms"] for row in rows]
    return {
        "case_count": len(rows),
        "first_reply_tool_accuracy": mean(row["first_reply_tool_correct"] for row in rows),
        "tool_decision_accuracy": mean(row["tool_correct"] for row in rows),
        "retries": sum(row["retried"] for row in rows),
        "rewrite_accuracy": mean(rewrites) if rewrites else None,
        "rewrite_count": len(rewrites),
        "all_correct": mean(
            row["tool_correct"] and row["rewrite_correct"] is not False for row in rows
        ),
        "median_latency_ms": median(latencies),
        "p95_latency_ms": percentile(latencies, 0.95),
        "cases": rows,
    }


def evaluate() -> dict[str, object]:
    from app.graph.chatbot import build_chatbot

    settings = Settings()
    chatbot = build_chatbot(agent=None, settings=settings)  # the agent is never called
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    metrics = {"chat_provider": settings.chat_provider, **score_chat(chatbot, cases)}
    if settings.chat_provider == "bedrock":
        metrics["chat_model"] = settings.bedrock_model
    RESULTS_FILE.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, ensure_ascii=False))
