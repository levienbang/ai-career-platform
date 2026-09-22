from sqlalchemy.orm import Session

from app.db.models import Job
from app.db.repositories import JobRepository
from app.schemas.jobs import JobResponse, JobSkillResponse


class JobNotFoundError(Exception):
    pass


def _to_response(job: Job) -> JobResponse:
    return JobResponse.model_validate(
        {
            "id": job.id,
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "employment_type": job.employment_type,
            "description": job.description,
            "experience_years_min": job.experience_years_min,
            "source_url": job.source_url,
            "posted_at": job.posted_at,
            "ingested_at": job.ingested_at,
            "skills": [
                JobSkillResponse(
                    skill=job_skill.skill.canonical_name,
                    requirement_type=job_skill.requirement_type,
                    importance=job_skill.importance,
                    evidence_text=job_skill.evidence_text,
                )
                for job_skill in job.skills
            ],
        }
    )


def list_jobs(session: Session, *, offset: int, limit: int) -> list[JobResponse]:
    return [_to_response(job) for job in JobRepository(session).list(offset=offset, limit=limit)]


def get_job(session: Session, job_id: int) -> JobResponse:
    job = JobRepository(session).get(job_id)
    if job is None:
        raise JobNotFoundError(job_id)
    return _to_response(job)
