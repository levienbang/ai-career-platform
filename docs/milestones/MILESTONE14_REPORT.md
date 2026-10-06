# Milestone 14 — Skill taxonomy and demo data layout

## Result

- One reviewed taxonomy: `data/taxonomy/skill_aliases.json`, alongside
  `skill_alias_blocklist.json`. All aliases formerly combined by `load_seed_data()`
  are preserved, including categories and insertion order: 55 skills, 56 aliases.
- Demo corpus: `data/examples/demo_jobs.json`, containing only the `jobs` array
  (12 jobs). JSON/CSV import examples also moved into `data/examples/`.
- `scripts/sync_skills.py` replaces `scripts/seed.py --taxonomy-only`.
  `scripts/load_demo_jobs.py` synchronizes taxonomy and loads demo jobs atomically.
  The old script was removed without a compatibility alias.
- Persistence remains in `app/services/skill_taxonomy.py`. Existing taxonomy,
  merge, learning, review and normalization algorithms were not changed.
- `SKILL_DATA_DIR` defaults to `data/taxonomy` relative to the working directory;
  empty still selects the default. Maintenance scripts locate taxonomy relative
  to their own files.

## Files changed

- Moved/merged data: `data/taxonomy/{skill_aliases,skill_alias_blocklist}.json`,
  `data/examples/{demo_jobs,sample_jobs}.json`, `data/examples/sample_jobs.csv`.
  Removed the former top-level data files.
- New commands: `scripts/sync_skills.py`, `scripts/load_demo_jobs.py`;
  removed `scripts/seed.py`.
- Updated paths in `scripts/merge_skills.py`, `scripts/learn_skill_aliases.py`,
  `scripts/review_skill_decisions.py`.
- Updated `app/config.py`, `app/services/skill_taxonomy.py`, `.env.example`.
- Updated `evaluation/evaluate_milestone7.py`, `evaluation/evaluate_retrieval.py`,
  `evaluation/milestone7_cases.json`.
- Added `tests/test_demo_data.py`, `tests/fixtures/milestone14_baseline.json`;
  removed `tests/test_seed.py`. Updated import/loader/ingestion, M7 evaluation,
  M10 CV/setup, M12 taxonomy and M13 indexing tests.
- Added `scripts/README.md`; updated `README.md`, `docs/DEMO.md`,
  `docs/CURRENT_MILESTONE.md` and Serena `core`, `conventions`,
  `suggested_commands`.

Existing user changes in AGENTS.md and memories were preserved. No dependency,
schema, migration, provider, API, project specification or Crawl changes.
Dockerfile already copies the entire data directory and development Compose
mounts it, so their paths need no edits. Rebuild an existing production API image
to include the new bundled layout; a development bind mount sees it immediately.
Earlier milestone reports remain historical and unchanged.

## Validation

Before deleting the old loader, captured its merged taxonomy and database output
on an empty SQLite test database. The frozen per-table SHA-256 fingerprints cover
all columns in companies, jobs, skills, skill_aliases and job_skills, including
IDs, alias `source`, evidence and relationship keys. Only generated timestamps
(`created_at`, `updated_at`, `ingested_at`) are excluded. The new commands match
this baseline both with an explicit skill sync first and with demo loading alone.
The baseline contains 55 skills, 56 aliases, 12 companies, 12 jobs and 26 job links.

Tests also cover repeated sync/demo calls without further changes, sync without
jobs, rollback of the entire demo transaction on failure, existing extracted
synonym consolidation and job-link preservation, execution from another working
directory, default/empty/overridden taxonomy paths and the relocated M7 fixtures.

Executed:

- Focused pytest run across demo data, M10/M12/M13, M7 and ingestion: **49 passed**
  before the final configuration test was added.
- `.venv/bin/pytest`: **313 passed, 1 skipped** on the final source.
- `.venv/bin/ruff check .`: **passed**.
- `.venv/bin/ruff format --check .`: **143 files already formatted**.
- Both new commands with `--help`: **passed**, including the demo-only warning.
- `git diff --check`: **passed**.
- Reference search outside `docs/milestones/`: no obsolete command, flag or
  fixture references remain after closing the milestone.

An initial Ruff check found two lines over the limit in the relocated evaluation
paths; formatting fixed them before the successful final checks.

## Commands and limits

For real data:

```bash
alembic upgrade head
python -m scripts.sync_skills
# Import real jobs through POST /jobs/import
python -m scripts.index_jobs
```

For an isolated demo/evaluation DB, use `python -m scripts.load_demo_jobs` instead
of the real-data import. It also performs skill synchronization. Do not mix this
corpus with real Crawl jobs. `scripts/README.md` documents all scripts, their
services, write/dry-run modes and paid API calls.

The skipped test is Docker E2E, gated by `RUN_DOCKER_E2E=1` and requiring real
PostgreSQL/Qdrant. It was not enabled. The full M7 evaluator was not executed
against a separate PostgreSQL database; its fixture/corpus tests passed. No live
provider checks, paid embedding/learning runs, production data loads or Docker
image build were performed. Migration verification is not applicable because
the database schema did not change.

## Concepts

- Curated taxonomy is operational reference data; it can be synchronized on a
  populated DB. Demo jobs are an isolated evaluation corpus. Import examples are
  inputs to the ingestion API, not that evaluation corpus.
- Idempotence means repeated synchronization/loading adds no further records.
- A single transaction keeps demo loading and taxonomy sync all-or-nothing.
- File-path refactoring needs data-equivalence checks, including provenance and
  links, rather than only counting jobs.
