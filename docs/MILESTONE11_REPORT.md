# Milestone 11 — DeepSeek chat, Google embeddings

Implemented on the existing uncommitted M9/M10 working tree. No changes to
`docs/PROJECT_SPEC.md`, `AGENTS.md`, the Crawl repository, database schemas,
embedding implementation, vector dimensions, or the user's `.env`.

## Changes in this milestone

- `app/config.py`: shared `DEEPSEEK_*` settings, HTTP(S) endpoint validation,
  nonempty default model, two allowed structured-output methods, empty component
  model overrides; removed old chat key and local inference settings.
- `app/llm.py`: one `ChatDeepSeek` builder; trimmed key/model checks; component
  timeouts/retries; explicit method and `include_raw`; JSON-mode schema guidance
  escaped for LangChain prompt templates.
- `app/ingestion/extractor.py`, `app/services/cv_extractor.py`,
  `app/graph/router.py`, `app/tools/sql_tool.py`, `app/retrieval/reranker.py`:
  shared DeepSeek key and optional component model; original primary prompts and
  validation remain; removed local-provider guidance. Single and batch extraction
  receive their respective schemas in JSON mode. Existing API exception handlers
  propagate the new missing-key message as HTTP 503.
- `.env.example`, `docker-compose.yml`: four DeepSeek variables; removed obsolete
  chat variables; retained Google embedding configuration.
- `pyproject.toml`, new `uv.lock`: selected `langchain-deepseek>=1.0,<2`, resolved
  to **1.1.1** with `langchain-core` **1.6.6**. Removed `langchain-ollama`.
  `langchain-openai` remains a transitive dependency of the official DeepSeek
  integration, not an application fallback.
- Added `scripts/check_deepseek.py`, deleted `scripts/check_gemini.py`: one request,
  zero client retries, optional method flag, model/method/timing/parsed result,
  sanitized failure output and nonzero exit status.
- `tests/test_llm_selection.py`: constructor arguments, both output methods,
  all five component overrides and prompts, missing/blank key, invalid settings,
  batch raw recovery including tool calls and JSON content with reasoning metadata.
- Updated `tests/test_agent_config.py`, `tests/test_reranker_config.py`,
  `tests/test_extractor.py`, `tests/test_embedding_config.py`,
  `tests/test_milestone10_extractor.py`, `tests/test_milestone10_ingestion.py`:
  provider/key names only; existing behavior assertions remain.
- `README.md`, `docs/PORTFOLIO_RELEASE.md`: current provider setup, connectivity
  command, privacy boundary and Google embedding separation. M10 request estimate
  retained. `docs/CURRENT_MILESTONE.md`: one whitespace-only fix in its Python
  example so the pre-existing format failure passes. This report is new.
- `.serena/memories/chat_provider.md`: durable provider configuration and boundaries.

## Technical decisions and concepts

The official [ChatDeepSeek integration](https://reference.langchain.com/python/langchain-deepseek/chat_models/ChatDeepSeek/with_structured_output)
supports function calling, JSON mode and raw envelopes. Its dependency resolver
accepted the existing core 1.x constraint, so a ChatOpenAI fallback was unnecessary.
The application allows only `function_calling` and `json_mode`; unsupported methods
are rejected by Settings. Pydantic still validates every final component result.
JSON mode instructs every component to return one object and includes its actual
schema; template braces are escaped so the schema creates no input variables.
Reasoning metadata is not used as structured evidence.

Provider selection and embedding representation are separate concerns: changing
chat does not change stored vectors or require re-indexing. All components share
one DeepSeek key, while their model override, timeout and retry policies remain
independent. Job extraction retains application-level retries, throttling,
deduplication, deterministic fast path, batch splitting and guards. Raw batch
responses allow recovery of valid indexed items after provider parser failures.

## Validation

Baseline before edits:

- `uv run --no-sync pytest`: **174 passed, 1 skipped**.
- `uv run --no-sync ruff check .`: passed.
- `uv run --no-sync ruff format --check .`: failed only on extra comment spacing
  in the Python example in `docs/CURRENT_MILESTONE.md`; fixed without changing scope.

After implementation:

- Focused provider/config/embedding/M10 checks: **101 passed**.
- Full `uv run --no-sync pytest`: **219 passed, 1 skipped**.
- `uv run --no-sync ruff check .`: passed.
- `uv run --no-sync ruff format --check .`: passed, **123 files** (including this report and provider memory).
- Required obsolete-provider scan of `app tests scripts`: no matches.
- Actual ChatDeepSeek runnable construction for both methods and batch schema:
  passed offline; no provider invocation.
- `uv run --no-sync python scripts/check_deepseek.py --help`: passed without API call.
- Dependency operations: `uv add 'langchain-deepseek>=1.0,<2' --no-sync`,
  `uv remove langchain-ollama --no-sync`, `uv sync --extra dev`: passed.
- `uv lock --check --offline`: passed (120 resolved packages).
- `git diff --check`: passed.
- `docker build -t ai-career-platform:milestone11 .`: passed (clean dependency installation and image export).
- `docker run --rm --network none ai-career-platform:milestone11 python -c ...`:
  API import and actual DeepSeek runnable construction passed without network access.

The initial plain `uv run` commands could not access the default cache within the
sandbox. Redirecting the cache then encountered sandbox DNS restrictions. Baseline
used approved access to the existing environment with `--no-sync`; subsequent
local checks used `UV_CACHE_DIR=/private/tmp/llm-uv-cache`. Dependency downloads and
Docker access used approved sandbox escalation. No check failure was suppressed.

## Run commands and remaining limits

Set `DEEPSEEK_API_KEY` in the ignored `.env`; set the separate Google
`EMBEDDING_API_KEY` for retrieval. Clear legacy component model values from the
user's `.env` or set valid DeepSeek overrides; empty means `DEEPSEEK_MODEL`.
Obsolete environment variables are ignored, but the retained component model
variables are still honored. Default chat model: `deepseek-v4-flash`.

```bash
uv sync --extra dev
uv run uvicorn app.main:app --reload
# Containerized alternative:
docker compose up --build -d
# Each invocation below makes one paid request; choose the needed method:
uv run python scripts/check_deepseek.py
uv run python scripts/check_deepseek.py --method json_mode
```

No paid smoke request, real-job import, or personal CV request was performed.
Those live checks are assigned to Claude by the milestone. Unit tests do not
prove live model availability, extraction quality, latency or cost. The skipped
E2E test requires explicitly enabled real PostgreSQL/Qdrant services. No migration
is required by M11. Job content and parsed CV text go to DeepSeek; embedding inputs
go to Google. Keys and personal content were not added to source or documentation.
