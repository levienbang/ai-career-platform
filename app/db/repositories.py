from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.db.models import Company, Job, JobSkill, Skill


class CompanyRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_name(self, name: str) -> Company | None:
        return self.session.scalar(select(Company).where(func.lower(Company.name) == name.lower()))

    def add(self, company: Company) -> Company:
        self.session.add(company)
        return company


class JobRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list(self, *, offset: int = 0, limit: int = 20) -> list[Job]:
        statement = (
            select(Job)
            .options(
                selectinload(Job.company),
                selectinload(Job.skills).selectinload(JobSkill.skill),
            )
            .order_by(Job.id)
            .offset(offset)
            .limit(limit)
        )
        return list(self.session.scalars(statement))

    def list_all(self) -> list[Job]:
        statement = (
            select(Job)
            .options(
                selectinload(Job.company),
                selectinload(Job.skills).selectinload(JobSkill.skill),
            )
            .order_by(Job.id)
        )
        return list(self.session.scalars(statement))

    def get_by_source_urls(self, source_urls: list[str]) -> list[Job]:
        if not source_urls:
            return []
        statement = select(Job).where(Job.source_url.in_(source_urls)).order_by(Job.id)
        return list(self.session.scalars(statement))

    def get(self, job_id: int) -> Job | None:
        statement = (
            select(Job)
            .where(Job.id == job_id)
            .options(
                selectinload(Job.company),
                selectinload(Job.skills).selectinload(JobSkill.skill),
            )
        )
        return self.session.scalar(statement)

    def get_many(self, job_ids: list[int]) -> list[Job]:
        statement = (
            select(Job)
            .where(Job.id.in_(job_ids))
            .options(selectinload(Job.skills).selectinload(JobSkill.skill))
            .order_by(Job.id)
        )
        return list(self.session.scalars(statement))

    def get_by_source_url(self, source_url: str) -> Job | None:
        return self.session.scalar(select(Job).where(Job.source_url == source_url))

    def get_by_content_hash(self, content_hash: str) -> Job | None:
        return self.session.scalar(select(Job).where(Job.content_hash == content_hash))

    def get_by_raw_hash(self, raw_hash: str) -> Job | None:
        return self.session.scalar(select(Job).where(Job.raw_hash == raw_hash))

    def add(self, job: Job) -> Job:
        self.session.add(job)
        return job


class SkillRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list_with_aliases(self) -> list[Skill]:
        statement = select(Skill).options(selectinload(Skill.aliases)).order_by(Skill.id)
        return list(self.session.scalars(statement))
