import json
from pathlib import Path

from sqlalchemy import func, select

from app.db.models import Company, Job, JobSkill
from app.ingestion.pipeline import JobIngestionPipeline

DATA_FILE = Path(__file__).parents[1] / "data" / "sample_jobs.json"


def load_fixture() -> list[dict[str, object]]:
    return json.loads(DATA_FILE.read_text())


def test_pipeline_handles_batch_and_database_duplicates(
    db_session, taxonomy, fake_extractor
) -> None:
    pipeline = JobIngestionPipeline(db_session, fake_extractor)

    first = pipeline.import_records(load_fixture())
    second = pipeline.import_records(load_fixture())

    assert first.model_dump() == {
        "processed": 9,
        "inserted": 6,
        "skipped_duplicates": 2,
        "failed": 1,
        "errors": [
            {
                "record": 3,
                "error": "Structured extraction failed",
            }
        ],
    }
    assert second.inserted == 0
    assert second.skipped_duplicates == 8
    assert second.failed == 1
    assert db_session.scalar(select(func.count()).select_from(Job)) == 6
    assert db_session.scalar(select(func.count()).select_from(Company)) == 6
    # Skills mentioned only in structured fields lack source-text evidence.
    assert db_session.scalar(select(func.count()).select_from(JobSkill)) == 3


def test_pipeline_deduplicates_content_without_source_url(
    db_session, taxonomy, fake_extractor
) -> None:
    record = {
        "title": "Computer Vision Intern",
        "company": "Vision Studio",
        "description": "Build CV prototypes.",
        "required_skills": ["cv"],
        "preferred_skills": [],
    }

    result = JobIngestionPipeline(db_session, fake_extractor).import_records([record, record])

    assert result.inserted == 1
    assert result.skipped_duplicates == 1
