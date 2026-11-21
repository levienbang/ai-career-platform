from app.retrieval.keyword import KeywordSearchService
from tests.factories import add_job


def test_keyword_search_ranks_exact_terms(db_session) -> None:
    vision_job = add_job(
        db_session,
        title="Computer Vision Intern",
        description="Analyze images with Python.",
        company_name="Vision",
        source_url="https://example.com/keyword-vision",
        required_skills=("Computer Vision", "Python"),
    )
    add_job(
        db_session,
        title="Backend Engineer",
        description="Build APIs backed by PostgreSQL.",
        company_name="Backend",
        source_url="https://example.com/keyword-backend",
        required_skills=("PostgreSQL",),
    )

    results = KeywordSearchService(db_session).search("computer vision images", limit=5)

    assert results[0].job_id == vision_job.id
    assert results[0].score > 0


def test_keyword_search_returns_empty_for_unmatched_query(db_session) -> None:
    add_job(
        db_session,
        title="Backend Engineer",
        description="Build APIs.",
        company_name="Backend",
        source_url="https://example.com/no-match",
    )

    assert KeywordSearchService(db_session).search("quantum chemistry") == []
