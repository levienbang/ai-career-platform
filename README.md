# AI Career Intelligence Platform

Milestones 1–5 provide the FastAPI/PostgreSQL foundation, structured job ingestion,
BM25 and dense retrieval, Reciprocal Rank Fusion (RRF), bounded reranking, independent
SQL/search tools, and a LangGraph router. Milestone 6 adds a one-request CV PDF
skill-gap report. Frontend and application tracking remain out of scope.

## Run with Docker Compose

```bash
cp .env.example .env
docker compose up --build -d
docker compose exec api python -m scripts.seed
docker compose exec api python -m scripts.index_jobs
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/jobs
```

Swagger UI is available at <http://localhost:8000/docs>.

`/health` checks the API process. `/ready` checks PostgreSQL and Qdrant and returns
503 if either dependency is unavailable; Compose uses `/ready` for the API
healthcheck. Existing volumes are retained by normal `up --build` commands.

## Architecture

```mermaid
flowchart LR
  U[Swagger / FastAPI] --> I[Job ingestion]
  I --> P[(PostgreSQL: jobs and skill taxonomy)]
  P --> X[Explicit index operation]
  X --> Q[(Qdrant: job vectors)]
  U --> A[LangGraph router]
  A --> S[Validated SQL tool]
  S --> P
  A --> R[Keyword + dense + RRF + reranker]
  R --> P
  R --> Q
  U --> C[CV PDF loader → structured extractor → evidence and alias check → skill gap]
  C --> P
```

Routes and tools are kept separate so SQL safety, retrieval, and CV scoring can be
tested without a model call. The CV endpoint reads a PDF in memory and does not
persist its text or extracted profile. Indexing is explicit after import, so a
new PostgreSQL job is not searchable in Qdrant until indexed.

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

## Real jobs from Crawl and CV matching

Apply the latest migration before importing. In the separate Crawl repository,
export and deliver VietJobs records with the `extended` payload profile and
JSON null for missing companies. On a fresh database, seed only the taxonomy so
the demo jobs do not affect real-data CV matches. Then index the imported jobs before
requesting CV matches:

```bash
# LLM repository, before delivery
uv run alembic upgrade head
uv run python -m scripts.seed --taxonomy-only

# Crawl repository
DELIVERY_PAYLOAD_PROFILE=extended DELIVERY_MISSING_COMPANY=send_null \
  python -m job_pipeline export --source vietjobs
python -m job_pipeline deliver --source vietjobs --endpoint http://localhost:8000/jobs/import

# LLM repository
uv run python scripts/index_jobs.py
curl -F 'file=@cv.pdf' -F 'limit=10' http://localhost:8000/cv/match
```

`company` can be null. Skills with `origin=extracted` came from job text and
have not been reviewed or merged into the curated taxonomy. Skills proposed by
the extractor without a matching mention in the source job are discarded.
CV matching combines dense semantic similarity with required and preferred
skill coverage; it does not compare years of experience or call the LLM
reranker. `CV_MATCH_CANDIDATES` controls the candidate pool (default 50), and
`CV_MATCH_SEMANTIC_WEIGHT` controls the semantic share of the final score
(default 0.6). Re-index all jobs after this milestone because the search
document no longer includes minimum experience.

## CV skill gap

Set `DEEPSEEK_API_KEY` in `.env` for CV structured extraction. Seed jobs
and taxonomy first. In Swagger UI, call `POST /cv/upload`
with one PDF and one or more `job_ids` form fields. For example:

```bash
curl -X POST http://localhost:8000/cv/upload \
  -F 'file=@/path/to/cv.pdf;type=application/pdf' \
  -F 'job_ids=1' -F 'job_ids=2'
```

The PDF is limited by `MAX_CV_BYTES` and `MAX_CV_PAGES`. A skill counts only when
the extractor supplies a quote found in the PDF text and that quote names the
canonical skill or an existing alias. Unknown skills are reported separately.
Each job's score is `0.8 × required coverage + 0.2 × preferred coverage`;
weights are renormalized if one group has no requirements. A job with no skill
requirements scores zero. Market frequency is the share of selected jobs missing
that skill. The response is immediate; CV text and profile data are not stored.

This approach can miss scanned PDFs and skills expressed without a configured
alias. Evidence checking blocks invented skills but cannot prove that a skill
was used proficiently.

## Job ingestion

Seed the controlled skill taxonomy before importing jobs:

```bash
docker compose exec api python -m scripts.seed
```

The seed command loads the taxonomy and 12 demo jobs from `data/seed_data.json`.
Use `python -m scripts.seed --taxonomy-only` (or add `--taxonomy-only` to the
Docker command above) on a fresh database for real-data tests. This seeds or
updates skills and aliases without creating companies or jobs; it leaves existing
jobs in place.
`data/sample_jobs.json` and `data/sample_jobs.csv` are separate import examples and
test fixtures; they are not loaded by the seed command.

Configure chat inference in `.env`:

```dotenv
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_STRUCTURED_OUTPUT_METHOD=function_calling
DEEPSEEK_THINKING=disabled
LLM_MAX_RETRIES=2
LLM_BATCH_SIZE=8
LLM_REQUESTS_PER_MINUTE=0
LLM_STRUCTURED_FAST_PATH=true
```

All chat and structured extraction calls (job extraction, CV extraction, agent
router, SQL generator, reranker) use the single `DEEPSEEK_MODEL`. Only embeddings
use Google. Timeouts and retries still apply per component.

`deepseek-v4-flash` runs in thinking mode unless told otherwise, and thinking
mode rejects the forced tool call that `function_calling` sends (HTTP 400
"Thinking mode does not support this tool_choice"). `DEEPSEEK_THINKING=disabled`
(the default) turns it off for every chat call, which also avoids paying for
reasoning tokens. Structured extraction does not need it.

The default `function_calling` method uses tool calling. Switch to `json_mode`
if needed; every system prompt then includes the JSON object schema. Pydantic
validates the final output in both modes. Missing `DEEPSEEK_API_KEY` causes a
configuration error. The key must remain in the ignored `.env`.

Check connectivity with exactly one structured-output request (no retries):

```bash
uv run python scripts/check_deepseek.py
# Optional alternative, also one request:
uv run python scripts/check_deepseek.py --method json_mode
```

Job content and parsed CV text are sent to the DeepSeek API. Google receives
embedding inputs via the separate `EMBEDDING_API_KEY`. This provider change does
not change embedding dimensions and does not require re-indexing.

Import a small CSV or JSON file with:

```bash
curl -F "file=@data/sample_jobs.json" http://localhost:8000/jobs/import
```

The response reports `processed`, `inserted`, `skipped_duplicates`, `failed`, and
record-level validation errors. A new skill with a matching mention in the job
is stored with `origin=extracted` for later review. Re-importing the
same source URL or normalized content does not create another job.

Import now runs in three stages within one request transaction:

1. Clean and validate each record, then deduplicate by source URL and raw hash
   against PostgreSQL and earlier records in the request. A valid lowercase
   SHA-256 `source_record_hash` is preserved; otherwise the cleaned, validated
   record is hashed as canonical JSON. Existing jobs retain null raw hashes.
2. Use a deterministic extraction when title, description and parseable source
   skills are present. Otherwise group records into sequential LLM batches.
3. Preserve content-hash business deduplication and skill evidence checks, then
   persist each record under a savepoint. Errors use the original 1-based record
   index and are sorted by that index.

VietJobs `technical_skills` do not distinguish required and preferred skills,
so they are stored as `required`. Explicit `required_skills` take priority;
`preferred_skills` remain preferred. Source skill strings support safe Python
list literals or comma, semicolon and newline delimiters. Skill evidence includes
the source title, description and original technical-skills field. Location,
employment type and experience are preserved or parsed from source fields.

`LLM_BATCH_SIZE` defaults to 8 (1–25). `LLM_REQUESTS_PER_MINUTE` defaults to 0
(unlimited); a positive value up to 1000 spaces request starts by `60 / RPM`
seconds, including retries and split batches. The limiter is shared within an
API process; separate workers or deployments need their own quota allocation.
`LLM_STRUCTURED_FAST_PATH=false` forces validated, unique records through the LLM
for debugging or comparison. Batches send only title, company, location,
employment type, description, source URL and technical skills. Source company
always wins; other supplied title/location/employment type/source URL values
override model output.

With no split fallback, budget up to
`ceil(llm_records / LLM_BATCH_SIZE) × (1 + LLM_MAX_RETRIES)` requests. A whole
batch failure retries, then recursively splits the batch in half, so repeated
failures can exceed this estimate. Missing or ambiguous response indices fail
only the corresponding records. Re-delivery of already persisted raw records
makes zero LLM calls; legacy jobs with null raw hashes still use source-URL and
content-hash deduplication. One INFO line per successful import request reports
`processed`, `skipped_before_llm`, `deterministic`, `llm_records`, `llm_calls`
and `failed` without logging job text.

CV search queries combine extracted summary, recognized skill names, projects,
experience and education in that order, capped at 2,000 characters. Raw CV text
and skill-evidence strings are excluded. CV-match scoring weights remain unchanged.

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
RERANKER_TIMEOUT_SECONDS=30
RERANKER_MAX_CANDIDATES=20
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=jobs
```

Dense and hybrid retrieval require `EMBEDDING_API_KEY`, independently of the chat
model. Missing this key is a configuration error; there is no keyword fallback.
Neither key belongs in source control. The reranker uses `DEEPSEEK_API_KEY` and
`DEEPSEEK_MODEL`. Google document and query
embeddings use the retrieval-specific task types and vectors are normalized before
Qdrant cosine search.

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

Configure the agent and a dedicated PostgreSQL read-only account:

```dotenv
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

The router and SQL generator use `DEEPSEEK_API_KEY` and `DEEPSEEK_MODEL`. Never commit API keys or a real database password. On a fresh PostgreSQL volume, Compose creates the read-only
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

## Milestone 7 evaluation

`evaluation/retrieval_cases.json` and `evaluation/agent_cases.json` retain their
original 12 labels each. `evaluation/milestone7_cases.json` adds 5 SQL, 5 CV, and
5 error/unanswerable cases, for 39 cases total. Labels were assigned from
`data/seed_data.json` before measuring results. SQL counts are scoped to its ten
`/eval/` jobs; CV fixtures contain synthetic skill text only.

Run the deterministic suite in a fresh, isolated database containing only the
12 jobs from `data/seed_data.json`. The evaluator rejects extra jobs because they
change retrieval rankings and make the fixed labels incomparable:

```bash
docker compose exec api python -m evaluation.evaluate_milestone7
```

It writes `evaluation/milestone7_results.json` with Recall@5, MRR@10, P50/P95
retrieval latency, routing intent/tool-selection accuracy and task success, SQL
executable/exact-result/unsafe-query rejection rates, CV gap correctness, and
overall P50/P95 latency and error rate. It uses an in-memory Qdrant collection,
synthetic embeddings, a lexical reranker, a heuristic router, a fake SQL agent
tool, and the real PostgreSQL seed data. SQL evaluation executes validated fixed
queries against PostgreSQL in read-only transactions. These results test the
pipeline and fixtures; they do **not** measure live LLM quality, token usage, or
provider cost. `evaluation.evaluate_retrieval`, `evaluation.evaluate_agent`, and
`evaluation.evaluate_routing` are separate live-provider evaluations and require
configured keys. Never run them for a no-API-key evaluation.

Set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and optionally
`LANGSMITH_PROJECT` to send agent graph traces. Tracing is disabled by default;
missing key or client setup failure leaves the graph usable. Tests verify a local
callback receives a complete run. A remote LangSmith trace requires a valid key
and network access.

Run the integration path with PostgreSQL and Qdrant running:

```bash
docker compose exec -e RUN_DOCKER_E2E=1 api pytest -q tests/test_e2e_pipeline.py
```

It imports a synthetic job, indexes it in an isolated Qdrant collection, routes a
question through a fake router and real search tool, checks the grounded answer,
then removes only its own job and collection.

## Engineering decisions and failure analysis

| Decision | Reason | Failure case and limit |
|---|---|---|
| Explicit PostgreSQL → Qdrant indexing | Keeps ingestion simple and repeatable | Retrieval is stale before indexing; the operator must run `scripts.index_jobs` or `/search/index`. |
| Existing skill taxonomy and quote evidence | Prevents the CV extractor from inventing recognized skills | An unlisted alias or scanned PDF can be missed; add reviewed aliases and provide text PDFs. |
| AST-validated SQL plus read-only transaction and role | Model output is untrusted | An unavailable read-only role blocks production SQL; configure and verify its grants on existing volumes. |
| RRF followed by bounded reranking | Combines dissimilar ranking scales | A weak reranker can lower Recall@5; compare all four methods before choosing. |
| Optional LangSmith callback | Tracing should not stop user queries | No remote trace is produced without a valid key; local callback tests only prove integration. |

The deterministic evaluation can appear strong because its fake router and
embeddings are designed around a small fixed corpus. Run the live evaluation
scripts separately before claiming model quality. The 2–3 minute Swagger demo
script is in `docs/DEMO.md`.

## Portfolio release audit (29 September 2026)

On an isolated Compose project with a fresh PostgreSQL volume, the 39-case
deterministic evaluation indexed 12 seed jobs. It measured the following; all
retrieval embeddings, reranking, and routing in this table use fakes:

| Method | Recall@5 | MRR@10 | P95 latency |
|---|---:|---:|---:|
| Keyword | 1.000 | 0.917 | 3.80 ms |
| Dense | 0.958 | 0.833 | 0.66 ms |
| Hybrid | 1.000 | 0.958 | 4.28 ms |
| Reranked | 1.000 | 1.000 | 3.86 ms |

The fake router selected the expected route in 12/12 cases. Five fixed SQL
queries returned exact results against PostgreSQL through a dedicated read-only
role, and the five error cases passed. These timings exclude provider/network
latency. An import-to-answer integration test passed against PostgreSQL and
Qdrant, with fake extraction, routing, embedding, and reranking.

The live model smoke test routed 0/3 synthetic questions: one provider HTTP 503,
one timeout, and one provider error. No current live retrieval or full agent
quality metric is available because `EMBEDDING_API_KEY` is not configured. The
saved `evaluation/retrieval_results.json` and `evaluation/routing_results.json`
are earlier runs and were not reproduced in this audit. Remote LangSmith tracing
and live CV extraction also remain unverified. A [sanitized local trace](evaluation/agent_trace_example.json)
records one complete fake-model search run on real PostgreSQL and Qdrant. See
[the audit and release notes](docs/PORTFOLIO_RELEASE.md)
for the acceptance checklist, failure analysis, demo steps, and accurate GitHub/CV
description.
