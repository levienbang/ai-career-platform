from pydantic import BaseModel, Field


class CVSkillClaim(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    evidence: str = Field(min_length=1, max_length=300)


class CVExtraction(BaseModel):
    summary: str | None = None
    experience: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    skills: list[CVSkillClaim] = Field(default_factory=list)


class RecognizedSkill(BaseModel):
    skill: str
    evidence: str


class GapSkill(BaseModel):
    skill: str
    frequency: float = Field(ge=0, le=1)


class JobGap(BaseModel):
    job_id: int
    title: str
    matched_skills: list[str]
    missing_required_skills: list[str]
    missing_preferred_skills: list[str]
    required_coverage: float | None = Field(default=None, ge=0, le=1)
    preferred_coverage: float | None = Field(default=None, ge=0, le=1)
    fit_score: float = Field(ge=0, le=1)


class CVGapReport(BaseModel):
    profile: CVExtraction
    recognized_skills: list[RecognizedSkill]
    unrecognized_skills: list[str]
    jobs: list[JobGap]
    missing_required_skills: list[GapSkill]
    missing_preferred_skills: list[GapSkill]


class CVMatchedJob(BaseModel):
    job_id: int
    title: str
    company: str | None
    location: str | None
    final_score: float
    semantic_score: float
    skill_score: float = Field(ge=0, le=1)
    matched_skills: list[str]
    missing_required_skills: list[str]
    missing_preferred_skills: list[str]


class CVMatchReport(BaseModel):
    recognized_skills: list[RecognizedSkill]
    unrecognized_skills: list[str]
    query: str
    jobs: list[CVMatchedJob]
