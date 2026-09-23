# AI Career Intelligence Platform

Milestones 1–3 provide the FastAPI/PostgreSQL foundation, structured job ingestion,
and two independent retrieval baselines: BM25 keyword search over PostgreSQL jobs and
dense semantic search over a Qdrant index. Hybrid fusion, reranking, SQL tools,
LangGraph, CV parsing, and a generated-answer API remain out of scope.

## Run with Docker Compose

```bash
cp .env.example .env
docker compose up --build -d
docker compose exec api python -m scripts.seed
docker compose exec api python -m scripts.index_jobs
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
ruff format --check .
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
LLM_PROVIDER=google
LLM_MODEL=gemini-3.5-flash
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

## Retrieval baselines

Configure embeddings and Qdrant in `.env`:

```dotenv
EMBEDDING_PROVIDER=google
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_API_KEY=
EMBEDDING_DIMENSIONS=768
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=jobs
```

When `EMBEDDING_API_KEY` is empty, the application uses the local `LLM_API_KEY`.
Neither value belongs in source control. Google document and query embeddings use
the retrieval-specific task types and vectors are normalized before Qdrant cosine
search.

The search document contains title, location, employment type, minimum experience,
required/preferred canonical skills, and description. These fields express role,
constraints, seniority, and duties that a candidate normally searches for. Company
is retained as Qdrant metadata but excluded from the embedded text because employer
names can dominate similarity without describing role fit. This can underperform for
queries that explicitly search by company; metadata filtering can address that in a
later milestone.

Index all jobs and remove Qdrant points whose PostgreSQL jobs no longer exist:

```bash
docker compose exec api python -m scripts.index_jobs
```

The point ID is the PostgreSQL `job_id`, so indexing a new or updated job performs an
upsert instead of creating a duplicate. `POST /search/index` runs the same full sync;
passing `{"job_id": 123}` indexes one job. After ingestion, run one of these index
operations. There is no background queue in Milestone 3, so PostgreSQL and Qdrant are
briefly out of sync until this explicit step runs.

Search examples:

```bash
curl -X POST http://localhost:8000/search/keyword \
  -H 'Content-Type: application/json' \
  -d '{"query":"computer vision internship", "limit":5}'

curl -X POST http://localhost:8000/search/semantic \
  -H 'Content-Type: application/json' \
  -d '{"query":"work with images as an intern", "limit":5}'
```

Keyword search is an in-process BM25 baseline rebuilt from the current PostgreSQL
rows for each request. It is deterministic and needs no external model, but exact
token mismatch and Vietnamese word segmentation can reduce recall. Dense retrieval
handles paraphrases better, but quality depends on the embedding model and it cannot
see a PostgreSQL update until the Qdrant point is reindexed.

Run the labelled 12-query evaluation after seeding and indexing:

```bash
docker compose exec api python -m evaluation.evaluate_retrieval
```

The command reports Recall@5 plus mean and median request latency for both methods
and writes `evaluation/retrieval_results.json`. Tests use a deterministic fake
embedding provider and never call a paid embedding API.
