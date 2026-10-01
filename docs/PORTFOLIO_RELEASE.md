# Portfolio release audit — 29 September 2026

## Verdict

**Not ready to present as a fully live AI system.** The core services and a
synthetic end-to-end path work, but the current provider setup has no embedding
key and the three small live routing probes did not return a route. A portfolio
demo can show the working database, API, safety controls, and deterministic
pipeline if every fake is labelled. Do not claim live model quality or publish
the historical retrieval/routing metrics as current results.

This audit used a separate Compose project and new volumes. The existing `llm`
project and its volumes were not reset. The audit image was based on the existing
API image plus the chat provider dependency configured at audit time: full rebuilds from the
repository Dockerfile failed because package downloads were unavailable or
interrupted in this environment. Thus runtime startup passed, while clean image
build reproducibility remains open.

## Completion criteria from `PROJECT_SPEC.md` §19

| Criterion | Status and evidence |
|---|---|
| Compose runs API, PostgreSQL, Qdrant | **Partial:** all three were healthy on fresh audit volumes; a full image rebuild failed at dependency download. |
| Import and normalize jobs | **Partial:** import-to-answer test passed with a fake extractor; live structured extraction was not verified. |
| SQL analytics returns correct test results | **Partial:** 5/5 fixed queries ran correctly on PostgreSQL; live natural-language SQL generation was not verified. |
| Dense and hybrid search work | **Partial:** exercised with synthetic embeddings and real Qdrant; live embedding provider unavailable. |
| Reranker or reason for omission | **Met with limitation:** reranker exists and was evaluated with a lexical fake; provider reranking was not verified. |
| Compare four retrieval methods | **Met for deterministic fixtures:** see measured table below; these are not live quality scores. |
| Agent routes SQL and search | **Partial:** fake router 12/12; live smoke test 0/3 returned a route. |
| SQL tool rejects dangerous queries | **Met:** validator tests and 5/5 evaluation error cases passed; audit DB role had SELECT but no INSERT, UPDATE, or DELETE grants. |
| At least 30 evaluation cases | **Met:** 39 labelled cases: 12 retrieval, 12 routing, 5 SQL, 5 CV, 5 error. |
| Retrieval and routing metrics | **Met for deterministic fixtures only:** live metrics remain unverified. |
| Trace for one complete agent run | **Met locally:** `evaluation/agent_trace_example.json` records route → search → answer with 6 matched start/end events; remote LangSmith trace was not verified. |
| Tests for important services | **Met:** 116 tests passed, including the opt-in E2E test, after the corpus guard fix. |
| README, architecture, run guide | **Met:** README contains the diagram, commands, and engineering decisions. |
| Failure analysis | **Met:** concrete cases are recorded below. |

The specification also asks for token use or estimated cost, a demo video, and
deployment/local instructions among its broader evaluation plan and deliverables.
Token/cost data and a recorded video are not available. Local run instructions
exist; no public deployment was performed.

## Measured results and provenance

The clean evaluation ran against **12 seed jobs** in PostgreSQL. Qdrant was
in-memory for the evaluation; the E2E test separately used the real Qdrant service.
The reranker, router, embeddings, and SQL agent tool in the 39-case suite are
fakes. SQL correctness uses five fixed, validated queries on real PostgreSQL.
No personal data was used.

| Retrieval method | Recall@5 | MRR@10 | P50 / P95 latency |
|---|---:|---:|---:|
| Keyword | 1.000 | 0.917 | 2.39 / 3.80 ms |
| Dense | 0.958 | 0.833 | 0.32 / 0.66 ms |
| Hybrid | 1.000 | 0.958 | 2.69 / 4.28 ms |
| Reranked | 1.000 | 1.000 | 2.94 / 3.86 ms |

Other deterministic results: routing intent/tool selection/task success 12/12;
SQL executable and exact-result accuracy 5/5; CV scoring 5/5; error handling
5/5; overall operation error rate 0/75 and P95 5.67 ms. The system latency is
for 75 local operations, not 39 end-to-end provider calls. The fixed SQL queries
used the actual read-only database role, verified with privileges `t|f|f|f` for
SELECT/INSERT/UPDATE/DELETE on `jobs`.

The live smoke test used three synthetic questions, no retries, and a 12-second
per-call timeout. It produced **0/3 successful routes**: HTTP 503 model overload,
a read timeout, and another provider error. This is a failure observation, not
a routing accuracy estimate. The earlier `retrieval_results.json` and
`routing_results.json` lack enough run provenance to serve as current audit
evidence. `agent_review_results.json` separately records 0/6 routes from a
previous review; it was not rerun here.

## Commands and observed checks

- `.venv/bin/pytest -ra`: 114 passed, 1 E2E skipped without services.
- With audit PostgreSQL/Qdrant and `RUN_DOCKER_E2E=1`, `.venv/bin/pytest -ra`:
  **116 passed** after the corpus guard test was added.
- `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .`: passed.
- `alembic current`: `20260920_02 (head)` on the fresh audit database.
- `GET /health` and `GET /ready`: HTTP 200; API, PostgreSQL, and Qdrant healthy.
- `POST /search/keyword` returned seeded Computer Vision jobs via FastAPI.
- `python -m evaluation.evaluate_milestone7`: passed, 39 cases, 12 indexed jobs.
- `python -m evaluation.evaluate_retrieval`: stopped at missing `EMBEDDING_API_KEY`.
- `python -m evaluation.evaluate_agent` and `python -m evaluation.evaluate_routing`:
  stopped at missing model configuration in the key-free audit container.
- The E2E test imported a synthetic job, stored it in PostgreSQL, indexed an
  isolated real Qdrant collection, and produced a grounded agent answer using
  fake model components; it cleaned up its job and collection.
- A separate synthetic search run produced the sanitized local graph trace in
  `evaluation/agent_trace_example.json`: LangGraph, route, search, and answer
  completed with 6 matched start/end events and 5 job results.

The audit Compose override was temporary and used distinct project and volume
names plus localhost-only ports. Do not use `docker compose down -v` on the
regular project: it would remove its user data. To start the regular application,
use `docker compose up --build -d` after configuring the required keys in the
ignored `.env`; seed and index with the commands in README.

## Failure cases and limits

1. The previous deterministic report indexed 17 jobs, whereas the seed corpus
   contains 12. Extra jobs changed retrieval metrics. The evaluator now rejects
   a database containing anything beyond the seed jobs, so results cannot be
   silently compared across different corpora.
2. Live routing failed under provider overload/timeout. The graph cannot
   provide a live agent answer when routing fails; fake success does not prove
   live reliability.
3. Dense/hybrid indexing and search require an embedding key; without it the
   live evaluation stops explicitly. The current E2E proof uses fake vectors.
4. Import, CV extraction, and reranking through a real model were not verified
   in this audit. The CV tests check synthetic text and scoring, not PDF quality
   on personal documents.
5. Job import and Qdrant indexing are separate operations. Search can be stale
   until indexing completes. BM25 can miss aliases/paraphrases, and the bounded
   reranker can change recall and add provider latency.
6. A local complete graph trace exists; no remote LangSmith trace or
   token/cost accounting was observed. Do not claim either in the portfolio.

## Security review

The current tracked and non-ignored file list contains no `.env`, private key,
PDF CV, database file, log, or temporary file. `.env` exists locally and is
ignored by both Git and Docker build context. A scan for common API-key and
private-key formats found no candidate in the tracked source; example passwords
in `.env.example` and README are placeholders. This review is limited to the
current tree, not all Git history. Do not commit `.env`, real CVs, or generated
provider outputs containing personal data.

## GitHub and CV wording

**GitHub short description:** FastAPI career intelligence prototype combining
PostgreSQL job analytics, Qdrant-backed hybrid retrieval, SQL safety validation,
and LangGraph routing. Includes a 39-case deterministic evaluation and an
end-to-end integration test on real database services.

**CV bullet:** Built a containerized AI career intelligence prototype with
FastAPI, PostgreSQL, Qdrant, and LangGraph; validated read-only SQL execution,
compared four retrieval modes on a 12-job synthetic corpus, and tested an
import-to-answer flow using real database services and fake model providers.

These descriptions deliberately do not claim live model accuracy or latency.
Record the 2–3 minute demo using the checklist in `docs/DEMO.md` after verifying
the provider configuration you intend to show.
