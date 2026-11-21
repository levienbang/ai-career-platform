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


def test_agent_answer_shows_jobs_and_flags_a_reranker_fallback() -> None:
    data = {
        "answer": "template answer",
        "search_result": {
            "jobs": [_job(7, "Android Developer")],
            "reranker": "cross_encoder",
            "reranker_fallback": True,
        },
    }

    markdown = render.agent_answer(data)

    assert "Android Developer" in markdown
    assert "dự phòng: **bge (local)**" in markdown
    assert "template answer" not in markdown


def test_agent_answer_falls_back_to_text_and_shows_errors() -> None:
    markdown = render.agent_answer({"answer": "No data.", "errors": ["sql: timeout"]})

    assert markdown.startswith("No data.")
    assert "⚠️ sql: timeout" in markdown


def test_sql_rows_become_a_table() -> None:
    data = {
        "answer": "",
        "sql_result": {"columns": ["city", "n"], "rows": [["HCM", 12]], "row_count": 1},
    }

    assert "| HCM | 12 |" in render.agent_answer(data)


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


def test_chat_context_folds_recent_turns_into_the_question() -> None:
    context = _load("context")
    history = [
        {"role": "user", "kind": "text", "content": "Tìm job Python"},
        {
            "role": "assistant",
            "kind": "agent",
            "data": {"search_result": {"jobs": [{"title": "Python Dev"}]}},
        },
    ]

    query = context.build_query("còn ở Hà Nội thì sao?", history)

    assert "tìm được job: Python Dev" in query
    assert query.endswith("Câu hỏi hiện tại: còn ở Hà Nội thì sao?")
    assert len(context.build_query("x" * 600, history)) <= context.MAX_QUERY_CHARS
