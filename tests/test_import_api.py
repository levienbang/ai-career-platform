from pathlib import Path

from app.api.jobs import get_job_extractor

DATA_FILE = Path(__file__).parents[1] / "data" / "sample_jobs.json"


def test_import_endpoint_and_reimport(client, taxonomy, fake_extractor) -> None:
    client.app.dependency_overrides[get_job_extractor] = lambda: fake_extractor

    first = client.post(
        "/jobs/import",
        files={"file": ("sample_jobs.json", DATA_FILE.read_bytes(), "application/json")},
    )
    second = client.post(
        "/jobs/import",
        files={"file": ("sample_jobs.json", DATA_FILE.read_bytes(), "application/json")},
    )

    assert first.status_code == 200
    assert first.json()["inserted"] == 2
    assert first.json()["skipped_duplicates"] == 1
    assert first.json()["failed"] == 1
    assert second.status_code == 200
    assert second.json()["inserted"] == 0
    assert second.json()["skipped_duplicates"] == 3


def test_import_endpoint_rejects_unsupported_files(client, fake_extractor) -> None:
    client.app.dependency_overrides[get_job_extractor] = lambda: fake_extractor

    response = client.post(
        "/jobs/import", files={"file": ("jobs.txt", b"not supported", "text/plain")}
    )

    assert response.status_code == 400
