# Technology Stack
- Python >=3.12; package metadata/build via `pyproject.toml` and setuptools.
- FastAPI + Pydantic Settings; Uvicorn ASGI server.
- PostgreSQL 17 (Compose), SQLAlchemy 2, psycopg 3, Alembic.
- Qdrant 1.19.1 (Compose) for dense vectors; in-process BM25 baseline; reciprocal-rank fusion; provider-backed reranking.
- LangChain Core and Google GenAI integration are dependencies; LangGraph is roadmap-only and must not be added before independent tools are complete.
- Pytest, pytest-cov, HTTPX, Ruff are development dependencies.
- Docker Compose runs PostgreSQL, Qdrant, and API; dev override bind-mounts source and enables Uvicorn reload.
- Local `.venv` exists but may be empty; install with `python -m pip install -e '.[dev]'` before expecting LSP import resolution or local tests.