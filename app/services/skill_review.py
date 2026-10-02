"""Human review of persisted learning decisions; no network calls."""

import re
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import JobSkill, Skill, SkillAlias, SkillDecision
from app.ingestion.normalizer import SkillNormalizer, skill_lookup_key, skill_match_key
from app.services.skill_learning import LearningProposal, SkillAliasLearner


def find_decision(session: Session, name: str, *, pending: bool = False) -> SkillDecision:
    decision = session.scalar(
        select(SkillDecision).where(SkillDecision.match_key == skill_match_key(name))
    )
    if decision is None or (pending and decision.decision != "pending"):
        raise ValueError("No matching pending decision" if pending else "Decision not found")
    return decision


def approve_decision(
    session: Session, name: str, canonical: str, *, settings: Settings, data_dir: Path | None = None
) -> None:
    decision = find_decision(session, name, pending=True)
    target = SkillNormalizer(session).resolve(canonical)
    if target is None or target.id == decision.skill_id:
        raise ValueError("Select a different existing canonical skill")
    learner = SkillAliasLearner(session, settings, data_dir=data_dir)
    learner.apply_one(
        LearningProposal(decision.name, "alias", target.id, 1.0, None, "Approved by reviewer."),
        replace_pending=True,
    )


def keep_new_decision(session: Session, name: str) -> None:
    decision = find_decision(session, name, pending=True)
    decision.decision = "new"
    decision.reason = "Kept as a distinct skill by reviewer."
    session.flush()


def _mentions(text: str | None, name: str) -> bool:
    return bool(
        re.search(
            rf"(?<!\w){re.escape(skill_lookup_key(name))}(?!\w)", skill_lookup_key(text or "")
        )
    )


def revert_alias(session: Session, name: str) -> int:
    alias = next(
        (
            item
            for item in session.scalars(select(SkillAlias))
            if skill_match_key(item.alias) == skill_match_key(name)
        ),
        None,
    )
    if alias is None or alias.source != "llm":
        raise ValueError("Only aliases with source='llm' can be reverted")
    decision = find_decision(session, alias.alias)
    canonical = session.get(Skill, alias.skill_id)
    skill = session.scalar(select(Skill).where(Skill.canonical_name == alias.alias))
    if skill is None:
        skill = Skill(canonical_name=alias.alias, origin="extracted")
        session.add(skill)
        session.flush()
    links = list(session.scalars(select(JobSkill).where(JobSkill.skill_id == canonical.id)))
    affected = 0
    for link in links:
        if not _mentions(link.evidence_text, alias.alias):
            continue
        affected += 1
        existing = session.scalar(
            select(JobSkill).where(
                JobSkill.job_id == link.job_id,
                JobSkill.skill_id == skill.id,
                JobSkill.requirement_type == link.requirement_type,
            )
        )
        remaining_evidence = re.sub(
            rf"(?<!\w){re.escape(skill_lookup_key(alias.alias))}(?!\w)",
            "",
            skill_lookup_key(link.evidence_text or ""),
        ).strip(" ;")
        canonical_names = [
            canonical.canonical_name,
            *(other.alias for other in canonical.aliases if other.id != alias.id),
        ]
        if any(_mentions(remaining_evidence, wording) for wording in canonical_names):
            # Retain canonical evidence only when it occurs outside the reverted wording.
            link.evidence_text = remaining_evidence
            if existing is None:
                session.add(
                    JobSkill(
                        job_id=link.job_id,
                        skill_id=skill.id,
                        requirement_type=link.requirement_type,
                        importance=link.importance,
                        evidence_text=alias.alias,
                    )
                )
        elif existing is not None:
            session.delete(link)
        else:
            link.skill_id = skill.id
    session.delete(alias)
    decision.decision = "rejected"
    decision.skill_id = skill.id
    decision.reason = "LLM alias reverted by reviewer."
    session.flush()
    session.expire_all()
    return affected
