# Task Completion
- Run focused tests for changed behavior, then the broader relevant suite; default full suite command: `pytest`.
- Run `ruff check .` and `ruff format --check .` when available.
- If the database/schema changes, create/inspect the Alembic migration and verify `alembic upgrade head`.
- If retrieval behavior changes, run relevant retrieval tests; run `python -m evaluation.evaluate_retrieval` only with required PostgreSQL/Qdrant/provider setup.
- If dependencies, Dockerfile, Python version, or system dependencies change, rebuild the dev image; ordinary mounted Python edits do not require rebuilding.
- Report files changed, technical decisions, project commands, exact test results, remaining limitations, and milestone concepts.
- Report any command not executed and why; do not claim completion with hidden failures.