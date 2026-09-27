from app.retrieval.types import SearchHit
from app.tools.search_tool import HybridJobSearchTool


class FakeSearchBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        self.calls.append((query, limit))
        return [
            SearchHit(
                job_id=7,
                score=0.9,
                title="Computer Vision Intern",
                company="Vision Lab",
                location="Hanoi",
                employment_type="Internship",
                experience_years_min=0,
                required_skills=["Python", "Computer Vision"],
                preferred_skills=[],
            )
        ]


def test_search_tool_wraps_existing_backend() -> None:
    backend = FakeSearchBackend()
    tool = HybridJobSearchTool(backend)

    result = tool.invoke("image internship", limit=3)

    assert backend.calls == [("image internship", 3)]
    assert result.jobs[0].job_id == 7


def test_search_langchain_wrapper_uses_structured_input() -> None:
    wrapped = HybridJobSearchTool(FakeSearchBackend()).as_langchain_tool()

    result = wrapped.invoke({"query": "vision", "limit": 1})

    assert result.jobs[0].title == "Computer Vision Intern"
