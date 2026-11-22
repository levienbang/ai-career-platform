"""Turn the chat session's messages into the plain-text history sent to /agent/chat.

Only text goes back to the API: earlier tool results are summarised, never resent as
data, so a follow-up question makes the chat model look the data up again.
"""

# The API trims history to its token budget (CHAT_CONTEXT_TOKENS); these caps only
# keep the request within the API's limits.
HISTORY_TURNS = 400
MAX_TURN_CHARS = 8000


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def summarize(message: dict) -> str | None:
    """One short line per message; None for messages that add no context."""
    if message.get("error"):
        return None
    kind = message.get("kind")
    data = message.get("data") or {}
    if kind == "text":
        return message.get("content", "").replace("📎", "CV:").strip() or None
    if kind == "chat":
        return data.get("answer")
    if kind == "agent":
        jobs = [job["title"] for job in (data.get("search_result") or {}).get("jobs", [])]
        if jobs:
            return "tìm được job: " + ", ".join(jobs[:4])
        sql = data.get("sql_result")
        if sql and sql.get("rows"):
            return f"kết quả thống kê: {sql['rows'][:3]}"
        return data.get("answer")
    if kind == "cv_match":
        skills = [item["skill"] for item in data.get("recognized_skills", [])][:8]
        jobs = [job["title"] for job in data.get("jobs", [])][:4]
        return f"CV có skill {', '.join(skills)}; job hợp: {', '.join(jobs)}"
    if kind == "cv_gap":
        missing = [item["skill"] for item in data.get("missing_required_skills", [])][:6]
        return "CV còn thiếu: " + ", ".join(missing)
    return None


def chat_history(messages: list[dict]) -> list[dict[str, str]]:
    """The last turns as [{"role": "user"|"assistant", "content": text}]."""
    history = []
    for message in messages[-HISTORY_TURNS:]:
        summary = summarize(message)
        if summary:
            history.append({"role": message["role"], "content": _short(summary, MAX_TURN_CHARS)})
    return history
