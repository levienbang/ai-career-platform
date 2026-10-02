"""List, approve, keep or revert skill learning decisions."""

import argparse
from pathlib import Path

from sqlalchemy import select

from app.config import Settings
from app.db.models import Skill, SkillDecision
from app.db.session import SessionLocal
from app.services.skill_review import approve_decision, keep_new_decision, revert_alias

DATA_DIR = Path(__file__).parents[1] / "data"


def review_skill_decisions(
    action: str,
    name: str | None = None,
    *,
    canonical: str | None = None,
    decision: str | None = None,
) -> None:
    with SessionLocal.begin() as session:
        if action == "list":
            query = select(SkillDecision).order_by(SkillDecision.id)
            if decision:
                query = query.where(SkillDecision.decision == decision)
            for item in session.scalars(query):
                target = session.get(Skill, item.skill_id) if item.skill_id else None
                print(
                    f"{item.name}\t{item.decision}\t{target.canonical_name if target else '-'}"
                    f"\t{item.confidence}\t{item.reason}"
                )
        elif action == "approve":
            approve_decision(session, name, canonical, settings=Settings(), data_dir=DATA_DIR)
        elif action == "keep-new":
            keep_new_decision(session, name)
        elif action == "revert":
            print(f"Reverted; {revert_alias(session, name)} job link(s) affected")
        else:
            raise ValueError("Unknown review action")
    if action != "list":
        print("Decision saved. Re-index changed jobs with: python scripts/index_jobs.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--decision", choices=["pending", "alias", "new"])
    approve = sub.add_parser("approve")
    approve.add_argument("name")
    approve.add_argument("--as", dest="canonical", required=True)
    for action in ("keep-new", "revert"):
        sub.add_parser(action).add_argument("name")
    args = parser.parse_args()
    review_skill_decisions(
        args.action,
        getattr(args, "name", None),
        canonical=getattr(args, "canonical", None),
        decision=getattr(args, "decision", None),
    )


if __name__ == "__main__":
    main()
