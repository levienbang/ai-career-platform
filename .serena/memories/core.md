# Project Core
- AI Career Intelligence Platform backend; current implemented scope is milestones 1–4: PostgreSQL schema, structured ingestion, BM25/dense retrieval, reciprocal-rank fusion, and bounded reranking.
- Source map: `app/api` FastAPI routes; `app/services` business services; `app/db` SQLAlchemy/session/repositories; `app/ingestion` cleaning/extraction/validation pipeline; `app/retrieval` keyword/dense/hybrid/reranking/Qdrant; `app/schemas` explicit Pydantic API models; `scripts` seed/index utilities; `migrations` Alembic revisions; `evaluation` retrieval benchmark; `tests` test suite.
- Roadmap and scope authority: `docs/PROJECT_SPEC.md`; implement only the user-requested milestone and do not edit the spec unless explicitly requested.
- Required implementation order: PostgreSQL/schema -> ingestion/extraction -> dense retrieval -> hybrid/reranking -> independent SQL/search tools -> LangGraph routing -> minimal CV skill gap -> evaluation/deployment.
- Keep business logic outside API routes. SQL and retrieval tools must work independently before LangGraph integration.
- Never commit secrets, API keys, CV contents, or private data. Update `.env.example` when adding environment variables.
- Read stack details in `mem:tech_stack`, codebase conventions in `mem:conventions`, runnable workflows in `mem:suggested_commands`, and completion gates in `mem:task_completion`.