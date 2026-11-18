import pytest

from app.db.models import RequirementType
from app.ingestion.cleaner import normalize_text
from app.ingestion.normalizer import SkillNormalizer, UnknownSkillsError, mentions


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


def test_normalize_text_unifies_unicode_and_whitespace() -> None:
    assert normalize_text("  Ｍachine\u00a0\n\tLearning ") == "Machine Learning"


@pytest.mark.parametrize(
    ("text", "phrase", "expected"),
    [
        ("Built APIs with Node.js and C++", "node.js", True),
        ("Built APIs with Node.js and C++", "c++", True),
        ("Machine\n  Learning pipelines", "machine learning", True),
        ("Five years of JavaScript", "Java", False),
        ("Worked at Google", "Go", False),
        (None, "Python", False),
    ],
)
def test_mentions_requires_whole_words(text, phrase, expected) -> None:
    assert mentions(text, phrase) is expected
