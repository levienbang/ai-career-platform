from dataclasses import dataclass


@dataclass(frozen=True)
class SearchHit:
    job_id: int
    score: float
    title: str
    company: str | None
    location: str | None
    employment_type: str | None
    experience_years_min: int | None
    required_skills: list[str]
    preferred_skills: list[str]
    document_text: str = ""
