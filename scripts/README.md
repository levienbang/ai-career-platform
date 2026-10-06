# Maintenance scripts

Run from the repository root with `python -m scripts.<name>` (or prefix with
`docker compose exec api`). Use `--help` for options. The taxonomy scripts find
`data/taxonomy/` relative to their own files, independent of the working directory.
`__init__.py` only marks the package; it is not a command.

| Script | Purpose / when to run | Paid API calls | Default effects | Service called |
|---|---|---|---|---|
| `sync_skills.py` | Synchronize reviewed skills before imports or after taxonomy edits; safe on an existing DB | None | Writes skills/aliases and merges extracted synonyms; idempotent, no jobs | `skill_taxonomy.seed_taxonomy` |
| `load_demo_jobs.py` | Load 12 jobs on an isolated demo/evaluation DB; do not mix with real Crawl data | None | Synchronizes skills, then writes demo jobs; idempotent, one transaction | `skill_taxonomy.seed_taxonomy`, `skill_taxonomy.load_demo_jobs` |
| `index_jobs.py` | Index after imports or skill changes; `--full` embeds all jobs again | Google, except `--payload-only` | Writes Qdrant; skips unchanged jobs; payload mode only refreshes existing points | `QdrantJobIndex.index_all` / `refresh_payloads`, embedding provider |
| `merge_skills.py` | Preview or apply known synonym merges | None | Dry-run; `--apply` writes merges | `skill_taxonomy.plan_skill_merges` / `apply_skill_merges` |
| `learn_skill_aliases.py` | Propose alias decisions for extracted skills, review saved JSON, then apply | DeepSeek even in dry-run; none with `--from FILE` | Dry-run for DB, writes proposal JSON; `--apply` writes decisions/aliases/merges | `SkillAliasLearner.propose` / `apply` |
| `review_skill_decisions.py` | List or manually approve, keep, or revert learned decisions | None | Required subcommand: `list` reads; `approve`, `keep-new`, `revert` write immediately | `skill_review.approve_decision` / `keep_new_decision` / `revert_alias` |
| `check_deepseek.py` | Check provider connectivity and structured output when troubleshooting | DeepSeek, one request | Live API check; no DB writes or dry-run | `llm.build_structured_chat_model`, `structured_output_guidance` |

## Thứ tự dựng từ DB trống

1. `alembic upgrade head`
2. `python -m scripts.sync_skills`
3. Import real data via `POST /jobs/import` (see the root README; extraction may
   call DeepSeek). Example files: `data/examples/sample_jobs.json` and `.csv`.
4. `python -m scripts.index_jobs` (Google embedding calls and Qdrant writes).

For an isolated demo/evaluation DB, replace step 3 with
`python -m scripts.load_demo_jobs`. This command also includes step 2. The corpus
is `data/examples/demo_jobs.json`; the deterministic Milestone 7 evaluator needs
exactly these 12 jobs. `python -m evaluation.evaluate_milestone7` uses offline
provider fixtures, while the other live evaluators may call paid providers.
