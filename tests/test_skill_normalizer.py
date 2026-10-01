import pytest

from app.db.models import RequirementType
from app.ingestion.normalizer import SkillNormalizer, UnknownSkillsError


def test_skill_normalizer_matches_aliases_case_insensitively(db_session, taxonomy) -> None:
    normalized = SkillNormalizer(db_session).normalize(
        ["POSTGRES", "python"], ["postgre sql", "FAST API"]
    )

    assert [(item.skill.canonical_name, item.requirement_type) for item in normalized] == [
        ("PostgreSQL", RequirementType.REQUIRED),
        ("Python", RequirementType.REQUIRED),
        ("FastAPI", RequirementType.PREFERRED),
    ]
    assert normalized[0].evidence_text == "POSTGRES"


def test_skill_normalizer_rejects_unknown_taxonomy_entries(db_session, taxonomy) -> None:
    with pytest.raises(UnknownSkillsError, match="Unknown Framework"):
        SkillNormalizer(db_session).normalize(
            ["Unknown Framework"], [], evidence_text="Unknown Framework", create_missing=False
        )
