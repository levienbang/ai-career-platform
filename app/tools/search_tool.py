from typing import Protocol

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.retrieval.types import SearchHit


class SearchBackend(Protocol):
    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]: ...


class SearchToolInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=20)


class SearchToolJob(BaseModel):
    job_id: int
    score: float
    title: str
    company: str
    location: str | None
    employment_type: str | None
    experience_years_min: int | None
    required_skills: list[str]
    preferred_skills: list[str]
    evidence: str


class SearchToolResult(BaseModel):
    query: str
    jobs: list[SearchToolJob]


class HybridJobSearchTool:
    """Thin tool wrapper around the existing reranked hybrid retrieval service."""

    def __init__(self, backend: SearchBackend) -> None:
        self.backend = backend

    def invoke(self, query: str, limit: int = 5) -> SearchToolResult:
        hits = self.backend.search(query, limit=limit)
        return SearchToolResult(
            query=query,
            jobs=[
                SearchToolJob.model_validate({**hit.__dict__, "evidence": hit.document_text})
                for hit in hits
            ],
        )

    def as_langchain_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.invoke,
            name="hybrid_job_search",
            description="Find relevant jobs with existing hybrid retrieval and reranking.",
            args_schema=SearchToolInput,
        )
