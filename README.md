# AI Career Intelligence Platform

Milestones 1–5 provide the FastAPI/PostgreSQL foundation, structured job ingestion,
BM25 and dense retrieval, Reciprocal Rank Fusion (RRF), bounded reranking, independent
SQL/search tools, and a LangGraph router. CV parsing, frontend, and Milestone 6 work
remain out of scope.

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

## Fast development with Docker

Build the development image once, then run Compose with source mounts and Uvicorn
reload:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build -d
```

After the first build, ordinary Python changes under `app/`, `scripts/`,
`evaluation/`, or `tests/` are visible inside the container immediately. Changes in
`app/` automatically restart Uvicorn; no image rebuild is needed:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f api
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec api pytest -q
```

Rebuild the dev image only after changing `pyproject.toml`, the Dockerfile, Python
version, or another system dependency. Use plain `docker compose up --build -d` when
you intentionally want to verify the production-style image without source mounts.

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

The seed command loads the taxonomy and 12 evaluation jobs from `data/seed_data.json`.
`data/sample_jobs.json` and `data/sample_jobs.csv` are separate import examples and
test fixtures; they are not loaded by the seed command.

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
HYBRID_CANDIDATE_LIMIT=20
RRF_K=60
RERANKER_PROVIDER=google
RERANKER_MODEL=gemini-3.5-flash
RERANKER_API_KEY=
RERANKER_TIMEOUT_SECONDS=30
RERANKER_MAX_CANDIDATES=20
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=jobs
```

When `EMBEDDING_API_KEY` is empty, the application uses the local `LLM_API_KEY`.
Neither value belongs in source control. The reranker also falls back to
`LLM_API_KEY` when `RERANKER_API_KEY` is empty. Google document and query embeddings use
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

Reindex once after upgrading from Milestone 3 because the Qdrant payload now includes
the complete search document needed by the reranker.

Search examples:

```bash
curl -X POST http://localhost:8000/search/keyword \
  -H 'Content-Type: application/json' \
  -d '{"query":"computer vision internship", "limit":5}'

curl -X POST http://localhost:8000/search/semantic \
  -H 'Content-Type: application/json' \
  -d '{"query":"work with images as an intern", "limit":5}'

curl -X POST http://localhost:8000/search/hybrid \
  -H 'Content-Type: application/json' \
  -d '{"query":"computer vision internship for fresher", "limit":5}'

curl -X POST http://localhost:8000/search/reranked \
  -H 'Content-Type: application/json' \
  -d '{"query":"computer vision internship for fresher", "limit":5}'
```

Keyword search is an in-process BM25 baseline rebuilt from the current PostgreSQL
rows for each request. It is deterministic and needs no external model, but exact
token mismatch and Vietnamese word segmentation can reduce recall. Dense retrieval
handles paraphrases better, but quality depends on the embedding model and it cannot
see a PostgreSQL update until the Qdrant point is reindexed.

Hybrid search retrieves independently from BM25 and dense search, then applies RRF
with a stable rank-based tie-break. Raw BM25 and cosine scores are deliberately not
added because they have unrelated scales. A job returned by both sources receives
two reciprocal-rank contributions and is emitted only once by `job_id`.

Reranked search sends only the top hybrid candidate pool (20 by default) to the
configured reranker. The response must contain every candidate ID exactly once;
unknown, missing, or duplicate IDs are rejected. Provider configuration and timeout
errors return HTTP 503. This strict behavior avoids silently presenting hybrid output
as if it had been reranked. Reranking can improve top-result ordering, but adds an LLM
request and therefore substantially increases latency and external API dependency.

Run the labelled 12-query evaluation after seeding and indexing:

```bash
docker compose exec api python -m evaluation.evaluate_retrieval
```

The command compares keyword, dense, hybrid, and hybrid + reranking using Recall@5,
MRR@10, mean latency, and median latency, then writes
`evaluation/retrieval_results.json`. The same fixed labels from Milestone 3 are used.
Tests use deterministic fake embedding and reranking providers and never call a paid
API.

## SQL/Search tools and LangGraph agent

Configure the agent model and a dedicated PostgreSQL read-only account:

```dotenv
AGENT_PROVIDER=google
AGENT_MODEL=gemini-3.5-flash
AGENT_API_KEY=
AGENT_TIMEOUT_SECONDS=30
AGENT_MAX_RETRIES=1
AGENT_SEARCH_LIMIT=5
AGENT_TOOL_MAX_RETRIES=1
SQL_MAX_ROWS=100
SQL_STATEMENT_TIMEOUT_MS=3000
POSTGRES_READONLY_USER=career_readonly
POSTGRES_READONLY_PASSWORD=choose-a-local-password
READONLY_DATABASE_URL=postgresql+psycopg://career_readonly:choose-a-local-password@postgres:5432/career
```

`AGENT_API_KEY` falls back to the local `LLM_API_KEY`. Never commit either key or a
real database password. On a fresh PostgreSQL volume, Compose creates the read-only
role automatically. With an existing volume, run the idempotent setup script once
after adding the two `POSTGRES_READONLY_*` values to `.env`:

```bash
docker compose up -d postgres
docker compose exec postgres /docker-entrypoint-initdb.d/10-readonly-role.sh
```

The SQL Tool treats generated SQL as untrusted. SQLGlot parses the PostgreSQL AST;
the validator accepts exactly one `SELECT` or `WITH ... SELECT`, rejects DML, DDL,
administrative nodes, unsafe functions, `SELECT *`, system tables, unknown columns,
locking queries, and multiple statements. It adds/caps `LIMIT`, while the executor
also uses a read-only transaction and a local statement timeout. These application
checks complement—not replace—the database role's SELECT-only grants.

The Search Tool is a thin wrapper over the existing hybrid plus reranking pipeline.
The graph routes each question to SQL, search, or both. Its final answer is rendered
only from structured tool output; a failed or empty tool produces an explicit
fallback instead of an inferred answer.

Try the independent SQL Tool and the graph:

```bash
curl -X POST http://localhost:8000/analytics/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Có bao nhiêu job yêu cầu Docker?"}'

curl -X POST http://localhost:8000/agent/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Tìm internship AI dùng Docker"}'
```

Run the fixed 12-question, pre-labelled agent evaluation with live PostgreSQL,
Qdrant, embedding, reranking, and agent provider configuration:

```bash
docker compose exec api python -m evaluation.evaluate_agent
```

It writes routing accuracy, required-tool success rate, mean/median/P95 latency, and
per-case failures to `evaluation/agent_results.json`. For `both` questions, search
runs first. The router marks whether SQL should cover the whole dataset or only the
returned job IDs. A statistic about "the jobs just found" therefore covers at most
the configured top search results, not the entire matching corpus. If a required
tool fails, the graph gives an explicit fallback.

To evaluate only routing without sending job candidate content to the reranker, run:

```bash
docker compose exec api python -m evaluation.evaluate_routing
```

This sends only the 12 labelled question strings to the configured agent model and
writes `evaluation/routing_results.json`.
