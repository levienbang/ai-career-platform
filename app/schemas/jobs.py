from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from app.db.models import RequirementType


class CompanySummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str


class JobSkillResponse(BaseModel):
    skill: str
    requirement_type: RequirementType
    importance: Decimal | None
    evidence_text: str | None


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    company: CompanySummary
    location: str | None
    employment_type: str | None
    description: str
    experience_years_min: int | None
    source_url: str | None
    posted_at: datetime | None
    ingested_at: datetime
    skills: list[JobSkillResponse]
