"""Chat demo for the AI Career Platform: ask questions or drop a CV (PDF) into the chat.

Text messages go to POST /agent/query, prefixed with the last five turns
(see context.py) so follow-up questions keep their context. A PDF attached to a message goes to
POST /cv/match; the result offers a skill-gap analysis via POST /cv/upload.
The CV is only sent to the configured local API and kept in this browser session.
Conversations (messages and API results, never the PDF bytes) are saved as JSON
files in demo/.chats so they survive a page reload.
"""

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import streamlit as st
from context import build_query

DEFAULT_API = os.getenv("CAREER_API_URL", "http://localhost:8000")
TIMEOUT = httpx.Timeout(180.0, connect=5.0)
CHAT_DIR = Path(os.getenv("CAREER_CHAT_DIR", Path(__file__).parent / ".chats"))

st.set_page_config(page_title="Career Chat", page_icon="💼", layout="wide")


# --------------------------------------------------------------------------- API


def _error_text(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = response.text[:300]
    return f"HTTP {response.status_code}: {detail}"


def call_api(method: str, path: str, **kwargs: Any) -> tuple[dict | None, str | None]:
    """Return (json, None) on success or (None, readable error)."""
    url = st.session_state.api_url.rstrip("/") + path
    try:
        response = httpx.request(method, url, timeout=TIMEOUT, **kwargs)
    except httpx.TimeoutException:
        return None, f"Hết thời gian chờ API ({path})."
    except httpx.HTTPError as error:
        return None, f"Không kết nối được API tại {url}: {error}"
    if response.is_error:
        return None, _error_text(response)
    return response.json(), None


def ask_agent(question: str, history: list[dict]) -> dict:
    query = build_query(question, history)
    data, error = call_api("POST", "/agent/query", json={"question": query})
    return {"kind": "agent", "data": data, "error": error, "query": query}


def match_cv(name: str, content: bytes, limit: int) -> dict:
    data, error = call_api(
        "POST",
        "/cv/match",
        files={"file": (name, content, "application/pdf")},
        data={"limit": str(limit)},
    )
    return {"kind": "cv_match", "data": data, "error": error, "file_name": name}


def skill_gap(name: str, content: bytes, job_ids: list[int]) -> dict:
    data, error = call_api(
        "POST",
        "/cv/upload",
        files={"file": (name, content, "application/pdf")},
        data={"job_ids": [str(job_id) for job_id in job_ids]},
    )
    return {"kind": "cv_gap", "data": data, "error": error, "file_name": name}


# ----------------------------------------------------------------------- render


def render_agent(data: dict) -> None:
    route = data.get("route", "?")
    st.caption(f"Route: **{route}** · {data.get('latency_ms', 0) / 1000:.1f}s")
    for error in data.get("errors") or []:
        st.warning(error)

    sql = data.get("sql_result")
    search = data.get("search_result")
    if sql:
        rows = [dict(zip(sql["columns"], row, strict=False)) for row in sql["rows"]]
        st.markdown(f"**Kết quả SQL** ({sql['row_count']} dòng)")
        st.dataframe(rows, use_container_width=True, hide_index=True)
        with st.expander("Câu SQL đã chạy"):
            st.code(sql["sql"], language="sql")
    if search and search.get("jobs"):
        st.markdown(f"**Job tìm được** cho “{search['query']}”")
        st.dataframe(
            [
                {
                    "id": job["job_id"],
                    "score": round(job["score"], 3),
                    "title": job["title"],
                    "company": job.get("company") or "—",
                    "location": job.get("location") or "—",
                    "required": ", ".join(job.get("required_skills") or []),
                }
                for job in search["jobs"]
            ],
            use_container_width=True,
            hide_index=True,
        )
    if not sql and not (search and search.get("jobs")):
        st.info(data.get("answer") or "Không có kết quả.")


def render_cv_match(message: dict) -> None:
    data = message["data"]
    recognized = [item["skill"] for item in data.get("recognized_skills", [])]
    st.markdown(f"**CV:** `{message['file_name']}` · nhận diện **{len(recognized)}** skill")
    st.write(" ".join(f"`{skill}`" for skill in recognized) or "—")
    if data.get("unrecognized_skills"):
        with st.expander(f"{len(data['unrecognized_skills'])} skill chưa có job nào yêu cầu"):
            st.write(", ".join(data["unrecognized_skills"]))

    jobs = data.get("jobs", [])
    if not jobs:
        st.info("Không tìm thấy job phù hợp.")
        return
    st.dataframe(
        [
            {
                "#": rank,
                "final": round(job["final_score"], 3),
                "semantic": round(job["semantic_score"], 3),
                "skill": round(job["skill_score"], 2),
                "title": job["title"],
                "location": job.get("location") or "—",
                "khớp": ", ".join(job["matched_skills"]),
                "thiếu (bắt buộc)": ", ".join(job["missing_required_skills"]),
            }
            for rank, job in enumerate(jobs, 1)
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "final": st.column_config.ProgressColumn("final", min_value=0, max_value=1),
        },
    )
    with st.expander("Câu truy vấn ngữ nghĩa dựng từ CV"):
        st.write(data.get("query", ""))


def render_cv_gap(message: dict) -> None:
    data = message["data"]
    st.markdown(f"**Skill gap** của `{message['file_name']}` so với {len(data['jobs'])} job")
    missing = sorted(data.get("missing_required_skills", []), key=lambda s: -s["frequency"])
    if missing:
        st.markdown("Skill **bắt buộc** còn thiếu (tỉ lệ job yêu cầu):")
        st.dataframe(
            [{"skill": s["skill"], "thiếu ở": f"{s['frequency']:.0%}"} for s in missing],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("Không thiếu skill bắt buộc nào trong các job đã chọn.")


def render(message: dict) -> None:
    if message.get("error"):
        st.error(message["error"])
    if message.get("query") and "\n" in message["query"]:
        with st.expander("Query đã gửi (kèm ngữ cảnh)"):
            st.text(message["query"])
    if message.get("error"):
        return
    if message["kind"] == "text":
        st.markdown(message["content"])
    elif message["kind"] == "agent":
        render_agent(message["data"])
    elif message["kind"] == "cv_match":
        render_cv_match(message)
    elif message["kind"] == "cv_gap":
        render_cv_gap(message)


# --------------------------------------------------------------------- storage


def _chat_path(chat_id: str) -> Path:
    return CHAT_DIR / f"{chat_id}.json"


def list_chats() -> list[dict]:
    """Saved conversations, most recently updated first."""
    chats = []
    for path in CHAT_DIR.glob("*.json"):
        try:
            chats.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return sorted(chats, key=lambda chat: chat["updated_at"], reverse=True)


def save_current_chat() -> None:
    messages = st.session_state.messages
    if not messages:
        return
    first = next((m["content"] for m in messages if m["role"] == "user"), "Cuộc trò chuyện")
    title = " ".join(first.replace("📎", "").split())[:40] or "Cuộc trò chuyện"
    now = datetime.now(UTC).isoformat()
    path = _chat_path(st.session_state.chat_id)
    created = now
    if path.exists():
        created = json.loads(path.read_text(encoding="utf-8")).get("created_at", now)
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "id": st.session_state.chat_id,
                "title": title,
                "created_at": created,
                "updated_at": now,
                "messages": messages,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def open_chat(chat_id: str | None) -> None:
    """Switch to a saved chat, or start a new empty one when chat_id is None."""
    st.session_state.chat_id = chat_id or uuid.uuid4().hex
    st.session_state.messages = []
    st.session_state.pop("cv", None)  # PDF bytes never outlive the chat they came from
    if chat_id and _chat_path(chat_id).exists():
        saved = json.loads(_chat_path(chat_id).read_text(encoding="utf-8"))
        st.session_state.messages = saved.get("messages", [])


# ------------------------------------------------------------------------- page

if "chat_id" not in st.session_state:
    open_chat(None)
if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API

with st.sidebar:
    if st.button("＋ Cuộc trò chuyện mới", use_container_width=True, type="primary"):
        open_chat(None)
        st.rerun()

    st.subheader("Lịch sử")
    chats = list_chats()
    if not chats:
        st.caption("Chưa có cuộc trò chuyện nào.")
    for chat in chats:
        current = chat["id"] == st.session_state.chat_id
        if st.button(
            ("▶ " if current else "") + chat["title"],
            key=f"chat-{chat['id']}",
            use_container_width=True,
            type="secondary",
            disabled=current,
        ):
            open_chat(chat["id"])
            st.rerun()
    if st.session_state.messages and st.button(
        "🗑 Xoá cuộc trò chuyện này", use_container_width=True
    ):
        _chat_path(st.session_state.chat_id).unlink(missing_ok=True)
        open_chat(None)
        st.rerun()

    st.divider()
    with st.expander("Cài đặt"):
        st.session_state.api_url = st.text_input("API URL", st.session_state.api_url)
        limit = st.slider("Số job gợi ý từ CV", 1, 20, 10)
        gap_top = st.slider("Phân tích skill gap trên top N job", 1, 20, 5)
    _, ready_error = call_api("GET", "/ready")
    if ready_error:
        st.error(f"API chưa sẵn sàng: {ready_error}")
    else:
        st.caption("🟢 API sẵn sàng")
    st.caption(
        "Gõ câu hỏi (vd: *Có bao nhiêu job yêu cầu Python?*, *Tìm job AI ở Hà Nội*) "
        "hoặc kéo CV PDF vào khung chat."
    )

st.title("💼 Career Chat")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render(message)

cv = st.session_state.get("cv")
if cv and cv.get("matched_ids"):
    label = f"Phân tích skill gap của CV với top {min(gap_top, len(cv['matched_ids']))} job"
    if st.button(label):
        with st.chat_message("assistant"), st.spinner("Đang phân tích skill gap..."):
            reply = skill_gap(cv["name"], cv["content"], cv["matched_ids"][:gap_top])
            reply["role"] = "assistant"
            render(reply)
        st.session_state.messages.append(reply)
        save_current_chat()

prompt = st.chat_input(
    "Hỏi về job, kỹ năng… hoặc kéo CV (PDF) vào đây",
    accept_file=True,
    file_type=["pdf"],
)
if prompt:
    text = (prompt.text or "").strip()
    files = list(prompt.files or [])
    user_note = text or ""
    if files:
        user_note = (user_note + "\n\n" if user_note else "") + f"📎 {files[0].name}"
    user_message = {"role": "user", "kind": "text", "content": user_note}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        render(user_message)

    with st.chat_message("assistant"):
        if files:
            pdf = files[0]
            content = pdf.getvalue()
            with st.spinner("Đang đọc CV và tìm job phù hợp..."):
                reply = match_cv(pdf.name, content, limit)
            if reply["data"]:
                st.session_state.cv = {
                    "name": pdf.name,
                    "content": content,
                    "matched_ids": [job["job_id"] for job in reply["data"].get("jobs", [])],
                }
        else:
            with st.spinner("Agent đang xử lý..."):
                reply = ask_agent(text, st.session_state.messages[:-1])
        reply["role"] = "assistant"
        render(reply)
    st.session_state.messages.append(reply)
    save_current_chat()
    st.rerun()
