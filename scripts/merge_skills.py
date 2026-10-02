"""Preview skill merges; persist only with --apply, in one transaction."""

import argparse
from pathlib import Path

from app.db.session import SessionLocal
from app.services.skill_taxonomy import (
    apply_skill_merges,
    load_skill_aliases,
    load_skill_blocklist,
    plan_skill_merges,
)


def merge_skills(*, apply: bool = False) -> int:
    with SessionLocal.begin() as session:
        data_dir = Path(__file__).parents[1] / "data"
        blocklist = load_skill_blocklist(data_dir / "skill_alias_blocklist.json")
        plan = plan_skill_merges(
            session, load_skill_aliases(data_dir / "skill_aliases.json"), blocklist=blocklist
        )
        print("Apply:" if apply else "Dry-run (no database changes):")
        for group in plan:
            print(
                f"{', '.join(group.removed_names)} -> {group.keep_name} "
                f"(keep id={group.keep_id}; {group.affected_job_skills} job_skills affected)"
            )
        if apply:
            apply_skill_merges(session, plan, blocklist=blocklist)
        print(f"{len(plan)} merge group(s)")
    if apply:
        print("Re-index with: python -m scripts.index_jobs (search documents contain skill names).")
    return len(plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write merges to the database")
    merge_skills(apply=parser.parse_args().apply)
