"""Chat demo for the AI Career Platform: ask questions or drop a CV (PDF) into the chat.

Text messages go to POST /agent/query. A PDF attached to a message goes to
POST /cv/match; the result offers a skill-gap analysis via POST /cv/upload.
The CV is only sent to the configured local API and kept in this browser session.
"""

import os
from typing import Any

import httpx
import streamlit as st

DEFAULT_API = os.getenv("CAREER_API_URL", "http://localhost:8000")
TIMEOUT = httpx.Timeout(180.0, connect=5.0)

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


def ask_agent(question: str) -> dict:
    data, error = call_api("POST", "/agent/query", json={"question": question})
    return {"kind": "agent", "data": data, "error": error}


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
        return
    if message["kind"] == "text":
        st.markdown(message["content"])
    elif message["kind"] == "agent":
        render_agent(message["data"])
    elif message["kind"] == "cv_match":
        render_cv_match(message)
    elif message["kind"] == "cv_gap":
        render_cv_gap(message)


# ------------------------------------------------------------------------- page

if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API

with st.sidebar:
    st.header("Cài đặt")
    st.session_state.api_url = st.text_input("API URL", st.session_state.api_url)
    limit = st.slider("Số job gợi ý từ CV", 1, 20, 10)
    gap_top = st.slider("Phân tích skill gap trên top N job", 1, 20, 5)
    _, ready_error = call_api("GET", "/ready")
    if ready_error:
        st.error(f"API chưa sẵn sàng: {ready_error}")
    else:
        st.success("API sẵn sàng")
    if st.button("Xoá hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("cv", None)
        st.rerun()
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
                reply = ask_agent(text)
        reply["role"] = "assistant"
        render(reply)
    st.session_state.messages.append(reply)
    st.rerun()
