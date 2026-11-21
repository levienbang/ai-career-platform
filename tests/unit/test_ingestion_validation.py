import pytest
from pydantic import ValidationError

from app.ingestion.cleaner import clean_job_record
from app.schemas.ingestion import JobExtraction, RawJobRecord


def test_cleaner_normalizes_whitespace_aliases_and_missing_values() -> None:
    cleaned = clean_job_record(
        {
            " job_description ": "  Build\nPython\tservices.  ",
            "company_name": " Example   Company ",
            "location": "N/A",
        }
    )

    assert cleaned == {
        "description": "Build Python services.",
        "company": "Example Company",
        "location": None,
    }


def test_raw_record_requires_description() -> None:
    with pytest.raises(ValidationError):
        RawJobRecord.model_validate({"title": "AI Engineer"})


def test_extraction_rejects_invalid_types_and_negative_experience() -> None:
    with pytest.raises(ValidationError):
        JobExtraction.model_validate(
            {
                "title": "AI Engineer",
                "company": "Example",
                "experience_years_min": -1,
                "required_skills": "Python",
                "preferred_skills": [],
                "description": "Build AI systems.",
            }
        )
