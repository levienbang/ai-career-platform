from pathlib import Path

import pytest

from app.ingestion.loader import JobLoadError, load_job_records

DATA_DIR = Path(__file__).parents[1] / "data"


def test_load_json_records() -> None:
    records = load_job_records("jobs.json", (DATA_DIR / "sample_jobs.json").read_bytes())

    assert len(records) == 9
    assert records[0]["required_skills"] == ["Python", "postgres"]


def test_load_csv_records() -> None:
    records = load_job_records("jobs.csv", (DATA_DIR / "sample_jobs.csv").read_bytes())

    assert len(records) == 4
    assert records[0]["required_skills"] == "Python;postgres"


def test_loader_rejects_unsupported_file_type() -> None:
    with pytest.raises(JobLoadError, match="Only .csv and .json"):
        load_job_records("jobs.txt", b"description")
