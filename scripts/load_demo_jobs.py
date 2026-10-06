"""Load 12 jobs for demo/evaluation only; do not use alongside real Crawl data.

Synchronizes curated skills first, so it also works on an empty database.
Writes immediately, idempotently; does not call paid APIs.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from app.db.session import SessionLocal
from app.services.skill_taxonomy import (
    load_demo_jobs as persist_demo_jobs,
)
from app.services.skill_taxonomy import (
    load_skill_aliases,
    load_skill_blocklist,
    seed_taxonomy,
)

DATA_DIR = Path(__file__).parents[1] / "data" / "taxonomy"
DEMO_JOBS_FILE = Path(__file__).parents[1] / "data" / "examples" / "demo_jobs.json"


def read_demo_jobs() -> list[dict[str, Any]]:
    return json.loads(DEMO_JOBS_FILE.read_text(encoding="utf-8"))["jobs"]


def load_demo_jobs() -> None:
    jobs = read_demo_jobs()
    definitions = load_skill_aliases(DATA_DIR / "skill_aliases.json")
    blocklist = load_skill_blocklist(DATA_DIR / "skill_alias_blocklist.json")
    with SessionLocal.begin() as session:
        skills = seed_taxonomy(session, definitions, blocklist=blocklist)
        persist_demo_jobs(session, jobs, skills)
    print(f"Demo corpus is ready: {len(jobs)} jobs.")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    load_demo_jobs()
