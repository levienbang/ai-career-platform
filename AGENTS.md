# AGENTS.md

## Project

This repository implements the AI Career Intelligence Platform.

Read `docs/PROJECT_SPEC.md` before making architectural decisions. That file is
the project roadmap, not an instruction to implement every milestone at once.

## Current milestone

Read `docs/CURRENT_MILESTONE.md` and implement it phase by phase. The user
authorized that file; it is overwritten for each new milestone. Where it
conflicts with `docs/PROJECT_SPEC.md`, the current milestone file wins for
that milestone only. Do not modify the Crawl repository (`../Crawl`).

## Working rules

- Implement only the milestone explicitly requested by the user.
- Do not implement features from later milestones without being asked.
- Inspect the relevant existing code before editing.
- Avoid broad repository scans when Serena can locate the needed symbols,
  references, or files more directly.
- Preserve existing user changes.
- Prefer simple, readable code over premature abstractions.
- Do not introduce a dependency unless the current milestone requires it.
- Keep business logic separate from API routes and framework-specific code.
- Each tool must work and be tested independently before integration into LangGraph.
- Do not build the LangGraph agent before the SQL and retrieval tools work independently.
- Do not modify `docs/PROJECT_SPEC.md` unless explicitly requested.
- Do not commit secrets, credentials, API keys, CV contents, or private data.
- Update `.env.example` whenever a new environment variable is introduced.
- Add or update tests for every behavior change.
- Run relevant tests after implementation.
- Report commands that could not be executed and explain why.


## Serena usage

Serena is available for semantic code navigation and project memory.

- Activate the current repository with Serena when starting a new task if it is
  not already active.
- Before broad repository exploration, inspect Serena's available project
  memories and read only the memories relevant to the current task.
- Prefer Serena semantic tools for:
  - symbol lookup,
  - symbol overview,
  - reference lookup,
  - implementation discovery,
  - code navigation.
- Do not read every Serena memory by default.
- Do not use Serena memory as a substitute for verifying current source code
  when implementation details matter.
- Update Serena memories only for durable project knowledge such as:
  - architectural decisions,
  - repository conventions,
  - important milestone state,
  - recurring commands,
  - important implementation constraints,
  - non-obvious lessons that are likely to matter in future sessions.
- Do not store raw logs, temporary debugging details, or short-lived implementation
  state in Serena memory.
- If a Serena memory conflicts with current source code or the project specification,
  treat the source code and `docs/PROJECT_SPEC.md` as authoritative.

## Technology constraints

Use the following stack unless the user explicitly changes it:

- Python
- FastAPI
- Pydantic
- PostgreSQL
- SQLAlchemy
- Alembic
- Qdrant
- LangChain
- LangGraph
- Pytest
- Docker Compose

Do not add Redis, Celery, Kafka, Kubernetes, Airflow, multi-agent systems,
fine-tuning, or application tracking unless explicitly requested.

## Implementation order

Follow this order:

1. PostgreSQL and data schema
2. Ingestion and structured extraction
3. Dense retrieval
4. Hybrid retrieval and reranking
5. Independent SQL and search tools
6. LangGraph routing
7. Minimal CV skill-gap analysis
8. Evaluation and deployment

Do not skip ahead unless explicitly requested.

## SQL safety

Any LLM-generated SQL must be treated as untrusted input.

- Only allow one read-only query.
- Only allow `SELECT` or `WITH ... SELECT`.
- Reject write and schema-changing statements.
- Use an allowlist of accessible tables and columns.
- Use a PostgreSQL user with read-only permissions.
- Apply row limits and statement timeouts.
- Never rely only on keyword matching for SQL validation.

## Quality requirements

Before declaring a milestone complete:

- Run relevant tests.
- Check formatting and linting when configured.
- Verify migrations when the milestone changes the database.
- Keep API schemas explicit.
- Include error handling for expected failures.
- Do not hide failing checks.
- Verify that new behavior fits the existing architecture rather than merely
  passing isolated tests.

## Tool usage

Use tools deliberately.

- Prefer Serena for semantic code navigation before broad filesystem searches.
- Prefer targeted file reads over reading entire large files.
- Prefer focused test execution before running the full test suite.
- Use framework or library documentation when current API behavior is uncertain.
- Do not introduce new tooling merely because it is available; use it only when
  it improves the current task.

## Final response

After each implementation, report:

- Files created or modified.
- Important technical decisions.
- Commands used to run the project.
- Tests executed and their actual results.
- Remaining limitations.
- Concepts the user should understand from the milestone.