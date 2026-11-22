"""Turn API JSON into Markdown for chat messages. Pure functions, easy to test."""

from typing import Any

RERANKER_LABELS = {
    "llm": "DeepSeek",
    "cross_encoder": "bge (local)",
    "none": "không rerank (thứ tự hybrid)",
}


def _cell(value: Any) -> str:
    text = "—" if value is None or value == "" else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    if not rows:
        return "_Không có dòng nào._"
    columns = columns or list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
        *("| " + " | ".join(_cell(row.get(column)) for column in columns) + " |" for row in rows),
    ]
    return "\n".join(lines)


def reranker_note(search: dict[str, Any]) -> str:
    used = search.get("reranker")
    if not used:
        return ""
    label = RERANKER_LABELS.get(used, used)
    if search.get("reranker_fallback"):
        return f"⚠️ Reranker chính lỗi, đã dùng dự phòng: **{label}**"
    return f"Reranker: {label}"


def jobs_table(jobs: list[dict[str, Any]]) -> str:
    return table(
        [
            {
                "#": rank,
                "id": job["job_id"],
                "score": round(job["score"], 3),
                "title": job["title"],
                "company": job.get("company"),
                "location": job.get("location"),
                "required": ", ".join((job.get("required_skills") or [])[:6]),
            }
            for rank, job in enumerate(jobs, 1)
        ]
    )


def result_tables(run: dict[str, Any]) -> list[str]:
    """Tables straight from one agent run's tool results (SQL rows, jobs)."""
    parts: list[str] = []
    sql = run.get("sql_result")
    search = run.get("search_result")
    if sql and sql.get("rows"):
        rows = [dict(zip(sql["columns"], row, strict=False)) for row in sql["rows"]]
        parts.append(f"**Kết quả thống kê** ({sql['row_count']} dòng)\n\n" + table(rows))
    if search and search.get("jobs"):
        parts.append(f"**Job phù hợp**\n\n{jobs_table(search['jobs'])}")
        note = reranker_note(search)
        if note:
            parts.append(f"_{note}_")
    return parts


def _thousands(value: int) -> str:
    if value < 1000:
        return str(value)
    return f"{value / 1000:.1f}".removesuffix(".0") + "K"


def context_note(data: dict[str, Any]) -> str:
    """'Ngữ cảnh: 20.1K / 100K token (20%)', from the provider's own token count."""
    used, limit = data.get("context_tokens") or 0, data.get("context_limit") or 0
    if not used or not limit:
        return ""
    note = f"Ngữ cảnh: {_thousands(used)} / {_thousands(limit)} token ({used / limit:.0%})"
    if spent := data.get("tokens_used"):
        note += f" · lượt này tốn {_thousands(spent)} token"
    if dropped := data.get("history_dropped"):
        note += f" · đã bỏ {dropped} tin cũ cho vừa ngữ cảnh"
    return note


def chat_answer(data: dict[str, Any]) -> str:
    """The chat model's reply, its warnings, then the data it was based on."""
    parts = [data.get("answer") or "Không có câu trả lời."]
    parts += [f"⚠️ {warning}" for warning in data.get("warnings") or []]
    runs = data.get("tool_runs") or []
    for run in runs:
        for error in run.get("errors") or []:
            parts.append(f"⚠️ {error}")
        tables = result_tables(run)
        if tables and len(runs) > 1:
            parts.append(f"---\n_Tra cứu: {run['question']}_")
        parts += tables
    if note := context_note(data):
        parts.append(f"_📊 {note}_")
    return "\n\n".join(parts)


def cv_match(data: dict[str, Any], file_name: str) -> str:
    skills = [item["skill"] for item in data.get("recognized_skills", [])]
    parts = [
        f"**CV:** `{file_name}` · nhận diện **{len(skills)}** skill",
        " ".join(f"`{skill}`" for skill in skills) or "—",
    ]
    if data.get("unrecognized_skills"):
        parts.append(
            "_Skill chưa có job nào yêu cầu:_ " + ", ".join(data["unrecognized_skills"][:15])
        )
    jobs = data.get("jobs", [])
    if not jobs:
        parts.append("Không tìm thấy job phù hợp.")
        return "\n\n".join(parts)
    parts.append(
        table(
            [
                {
                    "#": rank,
                    "final": round(job["final_score"], 3),
                    "semantic": round(job["semantic_score"], 3),
                    "skill": round(job["fit_score"], 2),
                    "title": job["title"],
                    "location": job.get("location"),
                    "khớp": ", ".join(job["matched_skills"][:6]),
                    "thiếu (bắt buộc)": ", ".join(job["missing_required_skills"][:6]),
                }
                for rank, job in enumerate(jobs, 1)
            ]
        )
    )
    parts.append("_final = 0.6 × semantic + 0.4 × skill (độ phủ skill bắt buộc/ưu tiên)._")
    return "\n\n".join(parts)


def skill_gap(data: dict[str, Any], file_name: str) -> str:
    missing = data.get("missing_required_skills", [])
    header = f"**Skill gap** của `{file_name}` so với {len(data.get('jobs', []))} job"
    if not missing:
        return header + "\n\nKhông thiếu skill bắt buộc nào trong các job đã chọn."
    return (
        header
        + "\n\nSkill **bắt buộc** còn thiếu (tỉ lệ job yêu cầu):\n\n"
        + table([{"skill": s["skill"], "thiếu ở": f"{s['frequency']:.0%}"} for s in missing])
    )


def admin_stats(data: dict[str, Any]) -> str:
    decisions = data.get("skill_decisions", {})
    fallbacks = data.get("reranker_fallbacks") or {}
    llm_fallbacks = data.get("llm_fallbacks") or {}
    indexed = data.get("indexed_jobs")
    rows = [
        {"mục": "Job", "giá trị": data["jobs"]},
        {"mục": "Job đã index (Qdrant)", "giá trị": "Qdrant lỗi" if indexed is None else indexed},
        {"mục": "Công ty", "giá trị": data["companies"]},
        {"mục": "Skill", "giá trị": data["skills"]},
        {"mục": "Skill chờ duyệt (pending)", "giá trị": decisions.get("pending", 0)},
        {"mục": "Alias đã gộp", "giá trị": decisions.get("alias", 0)},
        {"mục": "Lượt gọi trả phí hôm nay", "giá trị": data["paid_calls_today"]},
        {
            "mục": "Reranker fallback (từ lúc API khởi động)",
            "giá trị": ", ".join(f"{RERANKER_LABELS.get(k, k)}: {v}" for k, v in fallbacks.items())
            or 0,
        },
        {
            "mục": "LLM dự phòng (DeepSeek lỗi, từ lúc API khởi động)",
            "giá trị": ", ".join(f"{k}: {v}" for k, v in llm_fallbacks.items()) or 0,
        },
    ]
    note = ""
    if indexed is not None and indexed != data["jobs"]:
        note = "\n\n⚠️ Số job đã index khác số job trong DB: bấm **Index** để đồng bộ."
    return "**Thống kê hệ thống**\n\n" + table(rows) + note
