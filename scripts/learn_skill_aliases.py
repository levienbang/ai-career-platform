"""Classify existing extracted skills; dry-run unless --apply is supplied."""

import argparse
from pathlib import Path

from sqlalchemy import select

from app.config import Settings
from app.db.models import Skill
from app.db.session import SessionLocal
from app.services.skill_learning import SkillAliasLearner

DATA_DIR = Path(__file__).parents[1] / "data"


def learn_skill_aliases(
    *, apply: bool = False, settings: Settings | None = None, model=None
) -> int:
    with SessionLocal.begin() as session:
        names = list(
            session.scalars(
                select(Skill.canonical_name).where(Skill.origin == "extracted").order_by(Skill.id)
            )
        )
        learner = SkillAliasLearner(session, settings or Settings(), model=model, data_dir=DATA_DIR)
        proposals = learner.propose(names)
        print("Apply:" if apply else "Dry-run (no database changes; LLM calls still occur):")
        for item in proposals:
            target = session.get(Skill, item.skill_id) if item.skill_id else None
            print(
                f"{item.name} -> {item.decision}\t{target.canonical_name if target else '-'}"
                f"\t{item.confidence:.2f}\t{item.reason}"
            )
        if apply:
            learner.apply(proposals)
        print(f"{len(proposals)} decision(s); LLM calls={learner.llm_calls}")
    if apply:
        print("Re-index changed jobs with: python scripts/index_jobs.py")
    return learner.llm_calls


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write decisions and merge aliases")
    learn_skill_aliases(apply=parser.parse_args().apply)
