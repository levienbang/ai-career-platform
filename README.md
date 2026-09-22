# AI Career Intelligence Platform

Milestones 1–2 provide the FastAPI/PostgreSQL foundation and a small job-ingestion
pipeline: CSV/JSON loading, cleaning, LangChain structured extraction, Pydantic
validation, taxonomy-backed skill normalization, deduplication, and persistence.
Retrieval, Qdrant, embeddings, SQL tools, and LangGraph remain out of scope.

## Run with Docker Compose

```bash
cp .env.example .env
docker compose up --build -d
docker compose exec api python -m scripts.seed
curl http://localhost:8000/health
curl http://localhost:8000/jobs
```

Swagger UI is available at <http://localhost:8000/docs>.

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
alembic upgrade head
python -m scripts.seed
uvicorn app.main:app --reload
```

Run checks:

```bash
ruff check .
pytest
```

The database URL defaults to local PostgreSQL and can be overridden with
`DATABASE_URL`.

## Job ingestion

Seed the controlled skill taxonomy before importing jobs:

```bash
docker compose exec api python -m scripts.seed
```

Configure the extraction provider in `.env`:

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=your-model-name
LLM_API_KEY=your-api-key
LLM_MAX_RETRIES=2
```

`LLM_API_KEY` must remain local and must never be committed. Import a small CSV or
JSON file with:

```bash
curl -F "file=@data/sample_jobs.json" http://localhost:8000/jobs/import
```

The response reports `processed`, `inserted`, `skipped_duplicates`, `failed`, and
record-level validation errors. Unknown skill names fail validation until an alias
is deliberately added to the `skills`/`skill_aliases` taxonomy. Re-importing the
same source URL or normalized content does not create another job.

The included fixture intentionally contains aliases, a duplicate source URL, and
one invalid record. Automated tests replace the LLM extractor with a deterministic
fake and never make a provider request.
