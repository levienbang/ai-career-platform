from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    job_id: int
    score: float
    title: str
    company: str
    location: str | None
    employment_type: str | None
    experience_years_min: int | None
    required_skills: list[str]
    preferred_skills: list[str]


class SearchResponse(BaseModel):
    method: Literal["keyword", "dense", "hybrid", "reranked"]
    query: str
    latency_ms: float = Field(ge=0)
    results: list[SearchResult]


class IndexRequest(BaseModel):
    job_id: int | None = Field(default=None, ge=1)


class IndexResponse(BaseModel):
    collection: str
    indexed: int
    deleted: int
