"""Synchronize curated skills and aliases on an empty or existing database.

Writes immediately, idempotently; does not create jobs or call paid APIs.
"""

import argparse
from pathlib import Path

from app.db.session import SessionLocal
from app.services.skill_taxonomy import load_skill_aliases, load_skill_blocklist, seed_taxonomy

DATA_DIR = Path(__file__).parents[1] / "data" / "taxonomy"


def sync_skills() -> None:
    definitions = load_skill_aliases(DATA_DIR / "skill_aliases.json")
    blocklist = load_skill_blocklist(DATA_DIR / "skill_alias_blocklist.json")
    with SessionLocal.begin() as session:
        seed_taxonomy(session, definitions, blocklist=blocklist)
    print(f"Curated skills synchronized: {len(definitions)} groups.")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    sync_skills()
