"""Turn the chat session's messages into the plain-text history sent to /agent/chat.

Only text goes back to the API: earlier tool results are summarised, never resent as
data, so a follow-up question makes the chat model look the data up again. A chat
reply that used the tool also sends the questions it looked up ("lookups"); the API
replays them as tool calls whose results were removed, so the model sees that earlier
answers came from lookups rather than from memory.
"""

MAX_LOOKUPS = 3

# The API trims history to its token budget (CHAT_CONTEXT_TOKENS); these caps only
# keep the request within the API's limits.
HISTORY_TURNS = 400
MAX_TURN_CHARS = 8000


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def lookups(message: dict) -> list[str]:
    if message.get("kind") != "chat" or message.get("error"):
        return []
    runs = (message.get("data") or {}).get("tool_runs") or []
    return [_short(run["question"], 500) for run in runs if run.get("question")][:MAX_LOOKUPS]


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


def chat_history(messages: list[dict]) -> list[dict]:
    """The last turns as [{"role": "user"|"assistant", "content": text, "lookups"?: [...]}]."""
    history = []
    for message in messages[-HISTORY_TURNS:]:
        summary = summarize(message)
        if not summary:
            continue
        turn: dict = {"role": message["role"], "content": _short(summary, MAX_TURN_CHARS)}
        if questions := lookups(message):
            turn["lookups"] = questions
        history.append(turn)
    return history
