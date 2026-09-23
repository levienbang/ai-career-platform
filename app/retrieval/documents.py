from dataclasses import dataclass
from typing import Any

from app.db.models import Job, RequirementType


@dataclass(frozen=True)
class SearchDocument:
    job_id: int
    title: str
    text: str
    payload: dict[str, Any]


def _skill_names(job: Job, requirement_type: RequirementType) -> list[str]:
    return sorted(
        {
            job_skill.skill.canonical_name
            for job_skill in job.skills
            if job_skill.requirement_type == requirement_type
        },
        key=str.casefold,
    )


def build_search_document(job: Job) -> SearchDocument:
    if job.id is None:
        raise ValueError("Job must be persisted before creating a search document")

    required_skills = _skill_names(job, RequirementType.REQUIRED)
    preferred_skills = _skill_names(job, RequirementType.PREFERRED)
    experience = (
        f"{job.experience_years_min} years"
        if job.experience_years_min is not None
        else "Not specified"
    )
    fields = [
        f"Title: {job.title}",
        f"Location: {job.location or 'Not specified'}",
        f"Employment type: {job.employment_type or 'Not specified'}",
        f"Minimum experience: {experience}",
        f"Required skills: {', '.join(required_skills) or 'Not specified'}",
        f"Preferred skills: {', '.join(preferred_skills) or 'Not specified'}",
        f"Description: {job.description}",
    ]
    return SearchDocument(
        job_id=job.id,
        title=job.title,
        text="\n".join(fields),
        payload={
            "job_id": job.id,
            "title": job.title,
            "company": job.company.name,
            "location": job.location,
            "employment_type": job.employment_type,
            "experience_years_min": job.experience_years_min,
            "required_skills": required_skills,
            "preferred_skills": preferred_skills,
            "content_hash": job.content_hash,
            "search_text": "\n".join(fields),
        },
    )
