import pytest

from app.ingestion.cleaner import clean_job_record
from app.ingestion.pipeline import build_raw_record_hash
from app.schemas.ingestion import RawJobRecord


def test_raw_hash_cleaning_and_key_order():
    first = {"title": " AI  Engineer ", "description": " Python  work ", "company": "null"}
    second = {"company": None, "description": "Python work", "title": "AI Engineer"}
    assert build_raw_record_hash(RawJobRecord(**clean_job_record(first))) == build_raw_record_hash(
        RawJobRecord(**second)
    )
    assert build_raw_record_hash(RawJobRecord(**second, source_record_hash="a" * 64)) == "a" * 64
    for invalid in ["A" * 64, "xyz", None]:
        record = RawJobRecord(**second, source_record_hash=invalid)
        value = build_raw_record_hash(record)
        assert len(value) == 64 and value != invalid
        assert value == build_raw_record_hash(record)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        ("", []),
        ("['SQL', 'Python', 'sql']", ["SQL", "Python"]),
        ([" Python ", 2, "python", ""], ["Python"]),
        ("SQL;Python\nDocker", ["SQL", "Python", "Docker"]),
        ("[1, 2", ["[1", "2"]),
        ("__import__('os')", ["__import__('os')"]),
        (["x" * 101, "SQL"], ["SQL"]),
        ("[1, 2]", []),
    ],
)
def test_parse_skill_list(value, expected):
    from app.ingestion.structured import parse_skill_list

    assert parse_skill_list(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("3 năm", 3),
        ("6 tháng", 0),
        ("18 tháng", 1),
        ("Không yêu cầu", 0),
        ("2 years", 2),
        ("3 YRS", 3),
        ("18 months", 1),
        ("no experience", 0),
        ("unknown", None),
        (4, 4),
        (-1, None),
        (None, None),
        (True, None),
    ],
)
def test_parse_experience(value, expected):
    from app.ingestion.structured import parse_experience_years

    assert parse_experience_years(value) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("llm_batch_size", 0),
        ("llm_batch_size", 26),
        ("llm_requests_per_minute", -1),
        ("llm_requests_per_minute", 1001),
    ],
)
def test_ingestion_settings_bounds(field, value):
    from pydantic import ValidationError

    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(**{field: value}, _env_file=None)
