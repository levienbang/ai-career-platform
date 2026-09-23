from app.retrieval.documents import build_search_document
from tests.factories import add_job


def test_search_document_uses_retrieval_fields_and_metadata(db_session) -> None:
    job = add_job(
        db_session,
        title="Computer Vision Intern",
        description="Prototype image recognition models.",
        company_name="Example Vision",
        source_url="https://example.com/document-test",
        location="Hanoi",
        employment_type="Internship",
        experience_years_min=0,
        required_skills=("Python", "Computer Vision"),
        preferred_skills=("Docker",),
    )

    document = build_search_document(job)

    assert "Title: Computer Vision Intern" in document.text
    assert "Required skills: Computer Vision, Python" in document.text
    assert "Preferred skills: Docker" in document.text
    assert "Description: Prototype image recognition models." in document.text
    assert "Example Vision" not in document.text
    assert document.payload["job_id"] == job.id
    assert document.payload["company"] == "Example Vision"
    assert document.payload["content_hash"] == job.content_hash
