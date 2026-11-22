"""The Chainlit chat's Markdown rendering; chat/ is a separate app, loaded by path."""

import importlib.util
from pathlib import Path

CHAT_DIR = Path(__file__).parents[2] / "chat"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"chat_{name}", CHAT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


render = _load("render")


def _job(job_id: int, title: str) -> dict:
    return {
        "job_id": job_id,
        "score": 0.91234,
        "title": title,
        "company": None,
        "location": "Hà Nội",
        "required_skills": ["Python", "SQL"],
    }


def test_table_escapes_pipes_and_blanks() -> None:
    markdown = render.table([{"a": "x|y", "b": None}])

    assert markdown.splitlines() == ["| a | b |", "|---|---|", "| x\\|y | — |"]


def test_chat_answer_shows_the_reply_then_the_data() -> None:
    data = {
        "answer": "Hợp nhất là job 7 (Android Developer).",
        "warnings": ["Câu trả lời nhắc job không có trong dữ liệu tra được: 99"],
        "tool_runs": [
            {
                "question": "job Android",
                "search_result": {
                    "jobs": [_job(7, "Android Developer")],
                    "reranker": "cross_encoder",
                    "reranker_fallback": True,
                },
            }
        ],
    }

    markdown = render.chat_answer(data)

    assert markdown.startswith("Hợp nhất là job 7")
    assert "⚠️ Câu trả lời nhắc job" in markdown
    assert "| 1 | 7 |" in markdown
    assert "dự phòng: **bge (local)**" in markdown


def test_chat_answer_labels_each_lookup_when_there_are_several() -> None:
    sql = {"columns": ["city", "n"], "rows": [["HCM", 12]], "row_count": 1}
    data = {
        "answer": "So sánh xong.",
        "tool_runs": [
            {"question": "đếm job HCM", "sql_result": sql, "errors": ["sql: retried"]},
            {"question": "đếm job HN", "sql_result": sql},
        ],
    }

    markdown = render.chat_answer(data)

    assert "| HCM | 12 |" in markdown
    assert "_Tra cứu: đếm job HN_" in markdown
    assert "⚠️ sql: retried" in markdown


def test_context_meter_shows_usage_against_the_budget() -> None:
    data = {
        "answer": "ok",
        "context_tokens": 20_100,
        "context_limit": 100_000,
        "tokens_used": 25_400,
        "history_dropped": 3,
    }

    assert render.chat_answer(data).endswith(
        "_📊 Ngữ cảnh: 20.1K / 100K token (20%) · lượt này tốn 25.4K token"
        " · đã bỏ 3 tin cũ cho vừa ngữ cảnh_"
    )


def test_greeting_has_no_tables() -> None:
    assert render.chat_answer({"answer": "Chào bạn!", "tool_runs": []}) == "Chào bạn!"


def test_admin_stats_warn_when_index_is_out_of_sync() -> None:
    stats = {
        "jobs": 10,
        "companies": 2,
        "skills": 5,
        "skill_decisions": {"pending": 3},
        "indexed_jobs": 8,
        "paid_calls_today": 4,
        "reranker_fallbacks": {"none": 1},
    }

    markdown = render.admin_stats(stats)

    assert "| Skill chờ duyệt (pending) | 3 |" in markdown
    assert "không rerank (thứ tự hybrid): 1" in markdown
    assert "Index" in markdown.split("⚠️")[1]


def test_chat_history_sends_text_only() -> None:
    context = _load("context")
    messages = [
        {"role": "user", "kind": "text", "content": "Tìm job Python"},
        {"role": "assistant", "kind": "chat", "data": {"answer": "Có job 12 ở FPT."}},
        {"role": "assistant", "kind": "chat", "error": True, "data": {}},
        {"role": "user", "kind": "text", "content": "x" * 9000},
    ]

    history = context.chat_history(messages)

    assert history[:2] == [
        {"role": "user", "content": "Tìm job Python"},
        {"role": "assistant", "content": "Có job 12 ở FPT."},
    ]
    assert len(history) == 3
    assert len(history[-1]["content"]) == context.MAX_TURN_CHARS
