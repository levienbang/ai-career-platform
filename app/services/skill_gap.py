import re
from collections import Counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Job, JobSkill, RequirementType
from app.db.repositories import JobRepository
from app.ingestion.normalizer import SkillNormalizer, skill_lookup_key
from app.schemas.cv import CVExtraction, CVGapReport, GapSkill, JobGap, RecognizedSkill


class TargetJobError(ValueError):
    pass


def _job_gap(job: Job, available_ids: set[int]) -> JobGap:
    required: set[int] = set()
    preferred: set[int] = set()
    names: dict[int, str] = {}
    for item in job.skills:
        names[item.skill_id] = item.skill.canonical_name
        if item.requirement_type == RequirementType.REQUIRED:
            required.add(item.skill_id)
        elif item.requirement_type == RequirementType.PREFERRED:
            preferred.add(item.skill_id)
    preferred -= required
    matched = (required | preferred) & available_ids
    required_coverage = len(required & available_ids) / len(required) if required else None
    preferred_coverage = len(preferred & available_ids) / len(preferred) if preferred else None
    weights = [(0.8, required_coverage), (0.2, preferred_coverage)]
    denominator = sum(weight for weight, value in weights if value is not None)
    fit = (
        sum(weight * value for weight, value in weights if value is not None) / denominator
        if denominator
        else 0.0
    )
    return JobGap(
        job_id=job.id,
        title=job.title,
        matched_skills=sorted(names[skill_id] for skill_id in matched),
        missing_required_skills=sorted(names[skill_id] for skill_id in required - available_ids),
        missing_preferred_skills=sorted(names[skill_id] for skill_id in preferred - available_ids),
        required_coverage=required_coverage,
        preferred_coverage=preferred_coverage,
        fit_score=round(fit, 4),
    )


def analyze_skill_gap(
    session: Session, profile: CVExtraction, cv_text: str, job_ids: list[int]
) -> CVGapReport:
    unique_ids = list(dict.fromkeys(job_ids))
    if not unique_ids:
        raise TargetJobError("At least one target job is required")
    jobs = JobRepository(session).get_many(unique_ids)
    missing_ids = set(unique_ids) - {job.id for job in jobs}
    if missing_ids:
        raise TargetJobError(f"Target job not found: {min(missing_ids)}")

    recognized, unknown = recognize_cv_skills(session, profile, cv_text)
    gaps = [_job_gap(job, set(recognized)) for job in jobs]
    required_counts = Counter(skill for gap in gaps for skill in gap.missing_required_skills)
    preferred_counts = Counter(skill for gap in gaps for skill in gap.missing_preferred_skills)
    return CVGapReport(
        profile=profile,
        recognized_skills=sorted(recognized.values(), key=lambda item: item.skill),
        unrecognized_skills=sorted(unknown),
        jobs=gaps,
        missing_required_skills=[
            GapSkill(skill=name, frequency=count / len(jobs))
            for name, count in sorted(required_counts.items())
        ],
        missing_preferred_skills=[
            GapSkill(skill=name, frequency=count / len(jobs))
            for name, count in sorted(preferred_counts.items())
        ],
    )


def recognize_cv_skills(
    session: Session, profile: CVExtraction, cv_text: str
) -> tuple[dict[int, RecognizedSkill], set[str]]:
    normalizer = SkillNormalizer(session)
    recognized: dict[int, RecognizedSkill] = {}
    unknown: set[str] = set()
    market_skill_ids = set(session.scalars(select(JobSkill.skill_id).distinct()))
    normalized_text = skill_lookup_key(cv_text)
    for claim in profile.skills:
        evidence = claim.evidence.strip()
        normalized_evidence = skill_lookup_key(evidence)
        if not re.search(rf"(?<!\w){re.escape(normalized_evidence)}(?!\w)", normalized_text):
            continue
        skill = normalizer.resolve(claim.name)
        if skill is None or skill.id not in market_skill_ids:
            if claim.name.casefold() in evidence.casefold():
                unknown.add(claim.name)
        elif normalizer.evidence_mentions(skill, evidence, name=claim.name):
            recognized.setdefault(
                skill.id, RecognizedSkill(skill=skill.canonical_name, evidence=evidence)
            )

    return recognized, unknown
