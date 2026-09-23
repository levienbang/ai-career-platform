# Suggested Commands
- Local setup: `python3 -m venv .venv`; `source .venv/bin/activate`; `python -m pip install -e '.[dev]'`.
- Local database/app: `alembic upgrade head`; `python -m scripts.seed`; `uvicorn app.main:app --reload`.
- Production-style Compose: `docker compose up --build -d`; seed with `docker compose exec api python -m scripts.seed`; index with `docker compose exec api python -m scripts.index_jobs`.
- Dev Compose: `docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build -d`; subsequent start can omit `--build`; logs via the same files plus `logs -f api`.
- Tests/checks: `pytest`; `ruff check .`; `ruff format --check .`. In dev Compose: `docker compose -f docker-compose.yml -f docker-compose.dev.yml exec api pytest -q`.
- Retrieval evaluation after seeding/indexing: `docker compose exec api python -m evaluation.evaluate_retrieval`.
- Serena memory reference check: `serena memories check` from repository root.