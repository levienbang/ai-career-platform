"""ChatGPT-style chat for the AI Career Platform (Chainlit), with admin buttons.

Local use: one login (admin / admin by default) that is also the admin. Set
CHAT_PASSWORD before exposing it; APP_ENV=production refuses the default.

- Text goes to POST /agent/query with the last turns folded in (context.py).
- Attach a PDF to match a CV (/cv/match, then a skill-gap button for /cv/upload).
- Attach a .json/.csv job file to import it (/jobs/import).
- Admin buttons run at once (stats, skill review, index, LLM check); type "admin" to
  show them again. Chats are saved in a local SQLite file (history.py).
"""

import hmac
import os
from pathlib import Path

import api
import chainlit as cl
import history
import render
from context import build_query

USERNAME = os.getenv("CHAT_USERNAME", "admin")
PASSWORD = os.getenv("CHAT_PASSWORD", "admin")
if os.getenv("APP_ENV") == "production" and PASSWORD == "admin":
    raise RuntimeError("Set CHAT_PASSWORD: the default admin password is for local use only")
REVIEW_PAGE = 10
GAP_TOP = 5
HISTORY_TURNS = 10
MENU_WORDS = {"admin", "menu", "quản trị", "quan tri"}


@cl.data_layer
def data_layer():
    return history.build_data_layer()


@cl.password_auth_callback
async def login(username: str, password: str) -> cl.User | None:
    if hmac.compare_digest(username.encode(), USERNAME.encode()) and hmac.compare_digest(
        password.encode(), PASSWORD.encode()
    ):
        return cl.User(identifier=USERNAME)
    return None


@cl.set_starters
async def starters(user=None, language=None) -> list[cl.Starter]:
    return [
        cl.Starter(label="Đếm job", message="Có bao nhiêu job software ở Hồ Chí Minh?"),
        cl.Starter(label="Tìm job", message="Tìm job AI engineer cho người biết PyTorch"),
        cl.Starter(
            label="Câu vòng vo",
            message="lập trình ứng dụng cho điện thoại chạy hệ điều hành của Google",
        ),
        cl.Starter(label="Quản trị", message="admin"),
    ]


def _toolbar() -> list[cl.Action]:
    return [
        cl.Action(name="stats", payload={}, label="Thống kê", icon="chart-column"),
        cl.Action(name="review", payload={}, label="Duyệt skill", icon="list-checks"),
        cl.Action(name="aliases", payload={}, label="Alias LLM", icon="undo-2"),
        cl.Action(name="index", payload={}, label="Index", icon="database"),
        cl.Action(name="import_help", payload={}, label="Nạp job", icon="upload"),
        cl.Action(name="check_llm", payload={}, label="Kiểm tra DeepSeek", icon="plug"),
    ]


async def _say(content: str, *, toolbar: bool = False, actions=None) -> None:
    all_actions = [*(actions or []), *(_toolbar() if toolbar else [])]
    await cl.Message(content=content, actions=all_actions).send()


@cl.on_chat_start
async def start() -> None:
    cl.user_session.set("history", [])


@cl.on_chat_resume
async def resume(thread: dict) -> None:
    # Rebuild the context for follow-up questions from the saved messages.
    turns = []
    for step in thread.get("steps", []):
        if step.get("type") == "user_message":
            turns.append({"role": "user", "kind": "text", "content": step.get("output", "")})
        elif step.get("type") == "assistant_message" and step.get("output"):
            turns.append({"role": "assistant", "kind": "text", "content": step["output"][:300]})
    cl.user_session.set("history", turns[-HISTORY_TURNS:])


def _remember(user_text: str, reply: dict) -> None:
    turns = cl.user_session.get("history") or []
    turns += [{"role": "user", "kind": "text", "content": user_text}, reply]
    cl.user_session.set("history", turns[-HISTORY_TURNS:])


def _kind(element) -> str | None:
    name = (element.name or "").lower()
    if (element.mime or "").endswith("pdf") or name.endswith(".pdf"):
        return "pdf"
    if name.endswith((".json", ".csv")):
        return "jobs"
    return None


@cl.on_message
async def on_message(message: cl.Message) -> None:
    text = message.content.strip()
    files = [(el, _kind(el)) for el in message.elements if el.path and _kind(el)]
    if files:
        element, kind = files[0]
        await (_match_cv(element, text) if kind == "pdf" else _import(element))
    elif text.lower() in MENU_WORDS:
        await _say("**Công cụ quản trị**: bấm để chạy.", toolbar=True)
    elif text:
        await _ask_agent(text)


# ------------------------------------------------------------------ chat


async def _ask_agent(text: str) -> None:
    query = build_query(text, cl.user_session.get("history") or [])
    async with cl.Step(name="Agent", type="tool") as step:
        step.input = query
        data, error = await api.ask_agent(query)
        if data:
            step.output = f"route: {data.get('route')} · {data.get('latency_ms', 0) / 1000:.1f}s"
    if error:
        await _say(f"❌ {error}")
        return
    if data.get("sql_result"):
        async with cl.Step(name="SQL do LLM viết", type="tool", language="sql") as step:
            step.output = data["sql_result"]["sql"]
    await _say(render.agent_answer(data))
    _remember(text, {"role": "assistant", "kind": "agent", "data": data})


async def _match_cv(pdf, text: str) -> None:
    content = Path(pdf.path).read_bytes()
    if text:
        await _say("_Lưu ý: phần chữ gửi kèm CV chưa được dùng; hệ thống chỉ chạy CV match._")
    async with cl.Step(name="CV match", type="tool") as step:
        step.input = pdf.name
        data, error = await api.match_cv(pdf.name, content)
    if error:
        await _say(f"❌ {error}")
        return
    # The PDF stays in this session's memory only, for the skill-gap button.
    job_ids = [job["job_id"] for job in data.get("jobs", [])]
    cl.user_session.set("cv", {"name": pdf.name, "content": content, "job_ids": job_ids})
    actions = []
    if job_ids:
        actions.append(
            cl.Action(
                name="skill_gap",
                payload={},
                label=f"Phân tích skill gap với top {min(GAP_TOP, len(job_ids))} job",
                icon="search-check",
            )
        )
    await _say(render.cv_match(data, pdf.name), actions=actions)
    _remember(f"CV: {pdf.name}", {"role": "assistant", "kind": "cv_match", "data": data})


@cl.action_callback("skill_gap")
async def on_skill_gap(action: cl.Action) -> None:
    cv = cl.user_session.get("cv")
    if not cv:
        await _say("Hãy gửi lại CV (khi mở lại cuộc trò chuyện cũ, CV không còn trong bộ nhớ).")
        return
    data, error = await api.skill_gap(cv["name"], cv["content"], cv["job_ids"][:GAP_TOP])
    await _say(f"❌ {error}" if error else render.skill_gap(data, cv["name"]))
    if data:
        _remember("skill gap", {"role": "assistant", "kind": "cv_gap", "data": data})


# ----------------------------------------------------------------- admin


@cl.action_callback("stats")
async def on_stats(action: cl.Action) -> None:
    data, error = await api.call("GET", "/admin/stats", admin=True)
    await _say(f"❌ {error}" if error else render.admin_stats(data), toolbar=True)


async def _review(offset: int = 0, decision: str = "pending") -> None:
    data, error = await api.call(
        "GET",
        "/admin/skill-decisions",
        admin=True,
        params={"decision": decision, "limit": REVIEW_PAGE, "offset": offset},
    )
    if error:
        await _say(f"❌ {error}", toolbar=True)
        return
    title = "Skill chờ duyệt" if decision == "pending" else "Alias LLM đã gộp"
    shown = f"{offset + 1}–{offset + len(data['items'])}" if data["items"] else "0"
    hint = (
        "Gõ tên skill chuẩn rồi bấm **Gộp**, hoặc **Giữ riêng**."
        if decision == "pending"
        else "**Hoàn tác** tách alias ra lại thành skill riêng."
    )
    # One table element; its buttons call keep_new / merge / revert below in place.
    table = cl.CustomElement(
        name="SkillReview",
        props={"decision": decision, "items": data["items"]},
        display="inline",
    )
    more = []
    if offset + REVIEW_PAGE < data["total"]:
        more.append(
            cl.Action(
                name="more",
                payload={"offset": offset + REVIEW_PAGE, "decision": decision},
                label=f"Xem {REVIEW_PAGE} mục tiếp",
            )
        )
    await cl.Message(
        content=f"**{title}**: {data['total']} mục · đang xem {shown}. {hint}",
        elements=[table],
        actions=[*more, *_toolbar()],
    ).send()


@cl.action_callback("review")
async def on_review(action: cl.Action) -> None:
    await _review()


@cl.action_callback("aliases")
async def on_aliases(action: cl.Action) -> None:
    await _review(decision="alias")


@cl.action_callback("more")
async def on_more(action: cl.Action) -> None:
    await action.remove()
    await _review(action.payload["offset"], action.payload["decision"])


async def _decide(path: str, body: dict, done: str) -> dict:
    """Run one review decision; the table row shows the returned message."""
    data, error = await api.call("POST", f"/admin/skill-decisions/{path}", admin=True, json=body)
    if error:
        return {"ok": False, "message": error}
    return {"ok": True, "message": done + (" · cần Index" if data.get("reindex_needed") else "")}


@cl.action_callback("keep_new")
async def on_keep_new(action: cl.Action) -> dict:
    return await _decide("keep-new", {"name": action.payload["name"]}, "đã giữ riêng")


@cl.action_callback("merge")
async def on_merge(action: cl.Action) -> dict:
    name, canonical = action.payload["name"], action.payload["canonical"].strip()
    return await _decide("approve", {"name": name, "canonical": canonical}, f"→ {canonical}")


@cl.action_callback("revert")
async def on_revert(action: cl.Action) -> dict:
    return await _decide("revert", {"name": action.payload["name"]}, "đã tách ra")


@cl.action_callback("index")
async def on_index(action: cl.Action) -> None:
    async with cl.Step(name="Index Qdrant", type="tool") as step:
        data, error = await api.call("POST", "/search/index", admin=True, json={})
        step.output = error or data
    if error:
        await _say(f"❌ {error}", toolbar=True)
        return
    await _say(
        f"✅ Index xong: **{data['indexed']}** job được embed lại, **{data['deleted']}** "
        "điểm cũ bị xoá (job không đổi thì bỏ qua, không tốn embedding).",
        toolbar=True,
    )


@cl.action_callback("import_help")
async def on_import_help(action: cl.Action) -> None:
    await _say(
        "Bấm 📎 trong khung chat, chọn file job **.json** hoặc **.csv** (≤ 1 MB) rồi gửi. "
        "Mỗi job mới tốn một lượt DeepSeek. Mẫu: `data/examples/sample_jobs.json`."
    )


async def _import(upload) -> None:
    async with cl.Step(name="Nạp job", type="tool") as step:
        step.input = upload.name
        data, error = await api.call(
            "POST",
            "/jobs/import",
            admin=True,
            files={"file": (upload.name, Path(upload.path).read_bytes(), upload.mime or "")},
        )
    if error:
        await _say(f"❌ {error}", toolbar=True)
        return
    errors = "".join(
        f"\n- bản ghi {item['record']}: {item['error']}" for item in data.get("errors", [])[:10]
    )
    await _say(
        f"✅ Đã xử lý **{data['processed']}**: thêm **{data['inserted']}**, "
        f"trùng **{data['skipped_duplicates']}**, lỗi **{data['failed']}**.{errors}\n\n"
        "Bấm **Index** để đưa job mới vào tìm kiếm.",
        toolbar=True,
    )


@cl.action_callback("check_llm")
async def on_check_llm(action: cl.Action) -> None:
    data, error = await api.call("POST", "/admin/check-llm", admin=True)
    await _say(
        f"❌ {error}"
        if error
        else f"✅ DeepSeek trả lời đúng schema · model `{data['model']}` · "
        f"{data['latency_ms'] / 1000:.1f}s",
        toolbar=True,
    )
