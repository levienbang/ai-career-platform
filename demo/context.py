"""Fold recent chat turns into the single question string sent to /agent/query."""

MAX_QUERY_CHARS = 500  # NaturalLanguageQuery.question limit in the API
HISTORY_TURNS = 5


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


def build_query(question: str, messages: list[dict]) -> str:
    """Prefix up to five recent turns, trimmed so the result fits the API limit.

    The current question is always kept whole; older context is dropped first.
    """
    question = _short(question, MAX_QUERY_CHARS)
    lines = []
    for message in messages[-HISTORY_TURNS:]:
        summary = summarize(message)
        if summary:
            who = "Tôi" if message["role"] == "user" else "Trợ lý"
            lines.append(f"- {who}: {_short(summary, 120)}")
    suffix = f"Câu hỏi hiện tại: {question}"
    if not lines or len(suffix) >= MAX_QUERY_CHARS:
        return question
    header = "Ngữ cảnh trước đó:\n"
    while lines and len(header) + len("\n".join(lines)) + 1 + len(suffix) > MAX_QUERY_CHARS:
        lines.pop(0)  # drop the oldest turn first
    if not lines:
        return question
    return header + "\n".join(lines) + "\n" + suffix
