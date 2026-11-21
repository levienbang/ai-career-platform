# Career Chat (Chainlit)

ChatGPT-style UI for the API in this repo, with admin buttons in place of the
maintenance scripts. It replaced the earlier Streamlit demo.

```bash
cd chat
cp .env.example .env          # set CAREER_ADMIN_API_KEY and CHAINLIT_AUTH_SECRET
uv run chainlit create-secret # paste the value into .env
uv run chainlit run app.py --port 8502
```

Open <http://localhost:8502> and log in with `admin` / `admin` (change
`CHAT_USERNAME` / `CHAT_PASSWORD` before any public use). There is one account and it
is the admin: this UI is for local use.

Click **⚙️ Quản trị** next to the send button (`public/admin-button.js`), or type `admin`,
to show the admin buttons; every admin result shows them again. Buttons run at once.

| You do | The chat calls | Replaces |
|---|---|---|
| Type a question | `POST /agent/query` (last turns folded in, `context.py`) | – |
| Attach a CV (PDF) | `POST /cv/match`, then a button for `POST /cv/upload` | – |
| Attach a job file (.json / .csv) | `POST /jobs/import` | uploading with curl |
| **Thống kê** | `GET /admin/stats` | counting rows by hand |
| **Duyệt skill** | `GET /admin/skill-decisions` as one table (10 per page); each row has a target box, **Gộp** and **Giữ riêng** (`public/elements/SkillReview.jsx`) | `review_skill_decisions.py list/approve/keep-new` |
| **Alias LLM** | the same table for aliases, with **Hoàn tác** per row | `review_skill_decisions.py revert` |
| **Index** | `POST /search/index` | `scripts/index_jobs.py` |
| **Kiểm tra DeepSeek** | `POST /admin/check-llm` (one DeepSeek call) | `scripts/check_deepseek.py` |

The scripts still exist for deployment and for use without the UI.

Chats are saved in `.chats.db` (SQLite, gitignored; `CHAT_HISTORY_DB` moves it) and
listed in the sidebar. Only messages are saved: the CV itself stays in the session's
memory for the skill-gap button, so after reopening an old chat, attach it again.
Production (`docker-compose.prod.yml`) builds `chat/Dockerfile`, keeps history in a
volume and refuses to start with the default password.
