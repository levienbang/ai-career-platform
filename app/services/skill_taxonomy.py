"""Curated taxonomy and transactional skill consolidation, without API dependencies."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import JobSkill, RequirementType, Skill, SkillAlias, SkillDecision
from app.db.repositories import SkillRepository
from app.ingestion.normalizer import skill_match_key


def load_skill_aliases(path: Path | None = None) -> dict[str, Any]:
    path = path or get_settings().skill_data_dir / "skill_aliases.json"
    try:
        definitions = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Skill taxonomy file not found: {path.resolve()}. "
            "Set SKILL_DATA_DIR to the directory containing skill_aliases.json."
        ) from error
    owners: dict[str, str] = {}
    for canonical, definition in definitions.items():
        for name in [canonical, *definition["aliases"]]:
            key = skill_match_key(name)
            if not key or owners.setdefault(key, canonical) != canonical:
                raise ValueError(f"Conflicting skill alias: {name}")
    return definitions


def load_skill_blocklist(path: Path | None = None) -> set[frozenset[str]]:
    path = path or get_settings().skill_data_dir / "skill_alias_blocklist.json"
    try:
        pairs = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Skill blocklist file not found: {path.resolve()}. Set SKILL_DATA_DIR."
        ) from error
    if not isinstance(pairs, list) or any(
        not isinstance(pair, list)
        or len(pair) != 2
        or not all(isinstance(name, str) and skill_match_key(name) for name in pair)
        for pair in pairs
    ):
        raise ValueError(f"Invalid skill blocklist: {path}")
    return {frozenset(skill_match_key(name) for name in pair) for pair in pairs}


def blocked_skill_pair(first: str, second: str, blocklist: set[frozenset[str]]) -> bool:
    return frozenset((skill_match_key(first), skill_match_key(second))) in blocklist


@dataclass(frozen=True)
class SkillMergeGroup:
    keep_id: int
    keep_name: str
    removed_ids: tuple[int, ...]
    removed_names: tuple[str, ...]
    affected_job_skills: int


def plan_skill_merges(
    session: Session,
    definitions: dict[str, Any],
    *,
    blocklist: set[frozenset[str]] | None = None,
) -> list[SkillMergeGroup]:
    skills = SkillRepository(session).list_with_aliases()
    blocklist = load_skill_blocklist() if blocklist is None else blocklist
    parents = {skill.id: skill.id for skill in skills}

    def root(skill_id: int) -> int:
        while parents[skill_id] != skill_id:
            parents[skill_id] = parents[parents[skill_id]]
            skill_id = parents[skill_id]
        return skill_id

    canonical_keys = {
        skill_match_key(name): canonical
        for canonical, definition in definitions.items()
        for name in [canonical, *definition["aliases"]]
    }
    seen: dict[str, int] = {}
    for skill in skills:
        for name in [skill.canonical_name, *(alias.alias for alias in skill.aliases)]:
            key = skill_match_key(name)
            key = skill_match_key(canonical_keys.get(key, key))
            if key in seen:
                first, second = root(skill.id), root(seen[key])
                members = [item for item in skills if root(item.id) in (first, second)]
                names = [
                    name
                    for item in members
                    for name in [item.canonical_name, *(alias.alias for alias in item.aliases)]
                ]
                if not any(blocked_skill_pair(a, b, blocklist) for a in names for b in names):
                    parents[first] = second
            else:
                seen[key] = skill.id
    groups: dict[int, list[Skill]] = {}
    for skill in skills:
        groups.setdefault(root(skill.id), []).append(skill)
    counts = Counter(session.scalars(select(JobSkill.skill_id)))
    plan = []
    for members in groups.values():
        keep = min(
            members,
            key=lambda skill: (
                skill.canonical_name not in definitions,
                skill.origin != "curated",
                -counts[skill.id],
                skill.id,
            ),
        )
        display_name = canonical_keys.get(skill_match_key(keep.canonical_name), keep.canonical_name)
        if any(skill.canonical_name == display_name and skill not in members for skill in skills):
            display_name = keep.canonical_name
        if len(members) < 2 and display_name == keep.canonical_name:
            continue
        removed = sorted((skill for skill in members if skill.id != keep.id), key=lambda s: s.id)
        plan.append(
            SkillMergeGroup(
                keep.id,
                display_name,
                tuple(skill.id for skill in removed),
                tuple(skill.canonical_name for skill in removed),
                sum(counts[skill.id] for skill in members),
            )
        )
    return sorted(plan, key=lambda group: group.keep_id)


def apply_skill_merges(
    session: Session,
    plan: list[SkillMergeGroup],
    *,
    source: str = "merge",
    confidence: float | None = None,
    reason: str | None = None,
    blocklist: set[frozenset[str]] | None = None,
) -> None:
    """Flush unique collisions before reassignment; caller owns the transaction."""
    blocklist = load_skill_blocklist() if blocklist is None else blocklist
    for group in plan:
        member_ids = (group.keep_id, *group.removed_ids)
        names = list(session.scalars(select(Skill.canonical_name).where(Skill.id.in_(member_ids))))
        names.extend(
            session.scalars(select(SkillAlias.alias).where(SkillAlias.skill_id.in_(member_ids)))
        )
        names.append(group.keep_name)
        if any(blocked_skill_pair(a, b, blocklist) for a in names for b in names):
            raise ValueError("Skill merge blocked by skill_alias_blocklist.json")
        survivor = session.get(Skill, group.keep_id)
        old_name = survivor.canonical_name
        removed_names = list(group.removed_names)
        if (
            old_name != group.keep_name
            and session.scalar(select(Skill.id).where(Skill.canonical_name == group.keep_name))
            is None
        ):
            survivor.canonical_name = group.keep_name
            removed_names.append(old_name)
        links = list(
            session.scalars(
                select(JobSkill).where(JobSkill.skill_id.in_(member_ids)).order_by(JobSkill.id)
            )
        )
        required_jobs = {
            link.job_id for link in links if link.requirement_type == RequirementType.REQUIRED
        }
        # Prefer existing survivor links for equal requirements, then stable smallest ID.
        links.sort(
            key=lambda link: (
                link.requirement_type != RequirementType.REQUIRED,
                link.skill_id != group.keep_id,
                link.id,
            )
        )
        retained: list[JobSkill] = []
        seen: set[tuple[int, RequirementType]] = set()
        for link in links:
            key = (link.job_id, link.requirement_type)
            if key in seen or (
                link.job_id in required_jobs and link.requirement_type == RequirementType.PREFERRED
            ):
                target = next((item for item in retained if item.job_id == link.job_id), None)
                if target and link.evidence_text:
                    target.evidence_text = "; ".join(
                        dict.fromkeys(filter(None, [target.evidence_text, link.evidence_text]))
                    )
                session.delete(link)
            else:
                seen.add(key)
                retained.append(link)
        session.flush()
        for link in retained:
            link.skill_id = group.keep_id
        session.flush()
        session.execute(
            update(SkillAlias)
            .where(SkillAlias.skill_id.in_(group.removed_ids))
            .values(skill_id=group.keep_id)
        )
        session.execute(
            update(SkillDecision)
            .where(SkillDecision.skill_id.in_(group.removed_ids))
            .values(skill_id=group.keep_id)
        )
        for name in removed_names:
            if session.scalar(select(SkillAlias).where(SkillAlias.alias == name)) is None:
                alias_source = (
                    "merge" if name == old_name and old_name != group.keep_name else source
                )
                session.add(
                    SkillAlias(
                        skill_id=group.keep_id,
                        alias=name,
                        source=alias_source,
                        confidence=confidence if alias_source == "llm" else None,
                        reason=reason if alias_source == "llm" else None,
                    )
                )
        session.flush()
        # Explicitly moved all references; avoid ORM orphan cascades on stale collections.
        session.execute(delete(Skill).where(Skill.id.in_(group.removed_ids)))
    session.flush()
    session.expire_all()


def seed_taxonomy(
    session: Session,
    definitions: dict[str, Any],
    *,
    blocklist: set[frozenset[str]] | None = None,
) -> dict[str, Skill]:
    for name, definition in definitions.items():
        skill = session.scalar(select(Skill).where(Skill.canonical_name == name))
        if skill is None:
            skill = Skill(canonical_name=name, category=definition["category"], origin="curated")
            session.add(skill)
        else:
            skill.origin = "curated"
            skill.category = definition["category"]
    session.flush()
    apply_skill_merges(
        session, plan_skill_merges(session, definitions, blocklist=blocklist), blocklist=blocklist
    )
    skills = {skill.canonical_name: skill for skill in session.scalars(select(Skill))}
    for name, definition in definitions.items():
        for alias in definition["aliases"]:
            existing = session.scalar(select(SkillAlias).where(SkillAlias.alias == alias))
            if existing is None:
                session.add(SkillAlias(skill_id=skills[name].id, alias=alias, source="curated"))
            elif existing.skill_id != skills[name].id:
                raise ValueError(f"Conflicting database skill alias: {alias}")
            else:
                existing.source = "curated"
                existing.confidence = None
                existing.reason = None
    session.flush()
    return {name: skills[name] for name in definitions}
