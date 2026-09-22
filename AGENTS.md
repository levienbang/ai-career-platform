# AGENTS.md

## Project

This repository implements the AI Career Intelligence Platform.

Read `docs/PROJECT_SPEC.md` before making architectural decisions. That file is
the project roadmap, not an instruction to implement every milestone at once.

## Working rules

- Implement only the milestone explicitly requested by the user.
- Do not implement features from later milestones without being asked.
- Inspect the existing repository before editing.
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

## Final response

After each implementation, report:

- Files created or modified.
- Important technical decisions.
- Commands used to run the project.
- Tests executed and their actual results.
- Remaining limitations.
- Concepts the user should understand from the milestone.
