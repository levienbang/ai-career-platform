"""Classify existing extracted skills with the LLM; dry-run unless --apply is supplied.

A dry-run saves the LLM decisions to a JSON file. Review it, then apply that file
with --apply --from FILE so the LLM is not called a second time.
"""

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from app.config import Settings
from app.db.models import Skill, SkillDecision
from app.db.session import SessionLocal
from app.ingestion.normalizer import skill_match_key
from app.services.skill_learning import LearningProposal, SkillAliasLearner

DATA_DIR = Path(__file__).parents[1] / "data"
DEFAULT_PROPOSALS = Path("skill_learning_proposals.json")


def _print(session, proposals: list[LearningProposal]) -> None:
    for item in proposals:
        target = session.get(Skill, item.skill_id) if item.skill_id else None
        print(
            f"{item.name} -> {item.decision}\t{target.canonical_name if target else '-'}"
            f"\t{item.confidence:.2f}\t{item.reason}"
        )


def learn_skill_aliases(
    *,
    apply: bool = False,
    settings: Settings | None = None,
    model=None,
    save: Path | None = DEFAULT_PROPOSALS,
    source: Path | None = None,
) -> int:
    """Return the number of LLM calls made (0 when applying a saved file)."""
    settings = settings or Settings()
    with SessionLocal.begin() as session:
        learner = SkillAliasLearner(session, settings, model=model, data_dir=DATA_DIR)
        if source is not None:
            saved = json.loads(source.read_text(encoding="utf-8"))
            decided = set(session.scalars(select(SkillDecision.match_key)))
            proposals = [
                LearningProposal(**item)
                for item in saved["proposals"]
                if skill_match_key(item["name"]) not in decided
            ]
            print(f"Loaded {len(proposals)} undecided proposal(s) from {source}; no LLM calls.")
        else:
            names = list(
                session.scalars(
                    select(Skill.canonical_name)
                    .where(Skill.origin == "extracted")
                    .order_by(Skill.id)
                )
            )
            proposals = learner.propose(names)
        print("Apply:" if apply else "Dry-run (no database changes):")
        _print(session, proposals)
        if apply:
            learner.apply(proposals)
        elif save is not None and source is None:
            save.write_text(
                json.dumps(
                    {
                        "created_at": datetime.now(UTC).isoformat(),
                        "model": settings.deepseek_model,
                        "proposals": [asdict(item) for item in proposals],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Saved decisions to {save}; apply with: --apply --from {save}")
        print(f"{len(proposals)} decision(s); LLM calls={learner.llm_calls}")
    if apply:
        print("Refresh Qdrant with: python scripts/index_jobs.py --payload-only")
    return learner.llm_calls


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write decisions and merge aliases")
    parser.add_argument(
        "--from",
        dest="source",
        type=Path,
        help="Apply decisions saved by an earlier dry-run instead of calling the LLM",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=DEFAULT_PROPOSALS,
        help=f"Where a dry-run saves decisions (default: {DEFAULT_PROPOSALS})",
    )
    args = parser.parse_args()
    learn_skill_aliases(apply=args.apply, save=args.save, source=args.source)
