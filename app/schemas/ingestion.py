from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


def _clean_skill_values(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("skills must be a list of strings")

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ValueError("each skill must be a string")
        skill = " ".join(item.split())
        if not skill:
            continue
        key = skill.casefold()
        if key not in seen:
            seen.add(key)
            cleaned.append(skill)
    return cleaned


class RawJobRecord(BaseModel):
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    description: str = Field(min_length=1)
    title: str | None = None
    company: str | None = None
    location: str | None = None
    employment_type: str | None = None
    experience_years_min: int | str | None = None
    required_skills: list[str] | str | None = None
    preferred_skills: list[str] | str | None = None
    source_url: HttpUrl | None = None


class JobExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1)
    company: str | None = None
    location: str | None = None
    employment_type: str | None = None
    experience_years_min: int | None = Field(default=None, ge=0)
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    description: str = Field(min_length=1)
    source_url: HttpUrl | None = None

    @field_validator("company", mode="before")
    @classmethod
    def empty_company_is_none(cls, value: Any) -> Any:
        return None if isinstance(value, str) and not value.strip() else value

    _validate_required_skills = field_validator("required_skills", mode="before")(
        _clean_skill_values
    )
    _validate_preferred_skills = field_validator("preferred_skills", mode="before")(
        _clean_skill_values
    )


class JobBatchItem(JobExtraction):
    record_index: int = Field(ge=1)


class JobBatchExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[JobBatchItem]


class ImportErrorDetail(BaseModel):
    record: int
    error: str


class ImportResult(BaseModel):
    processed: int
    inserted: int
    skipped_duplicates: int
    failed: int
    errors: list[ImportErrorDetail] = Field(default_factory=list)
