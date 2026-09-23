from hashlib import sha256

from sqlalchemy import select

from app.db.models import Company, Job, JobSkill, RequirementType, Skill


def add_job(
    session,
    *,
    title: str,
    description: str,
    company_name: str,
    source_url: str,
    location: str | None = None,
    employment_type: str | None = "Full-time",
    experience_years_min: int | None = None,
    required_skills: tuple[str, ...] = (),
    preferred_skills: tuple[str, ...] = (),
) -> Job:
    company = Company(name=company_name, location=location)
    job = Job(
        title=title,
        company=company,
        location=location,
        employment_type=employment_type,
        experience_years_min=experience_years_min,
        description=description,
        source_url=source_url,
        content_hash=sha256(source_url.encode()).hexdigest(),
    )
    for skill_name in required_skills:
        skill = session.scalar(select(Skill).where(Skill.canonical_name == skill_name))
        job.skills.append(
            JobSkill(
                skill=skill or Skill(canonical_name=skill_name),
                requirement_type=RequirementType.REQUIRED,
            )
        )
    for skill_name in preferred_skills:
        skill = session.scalar(select(Skill).where(Skill.canonical_name == skill_name))
        job.skills.append(
            JobSkill(
                skill=skill or Skill(canonical_name=skill_name),
                requirement_type=RequirementType.PREFERRED,
            )
        )
    session.add(job)
    session.commit()
    return job
