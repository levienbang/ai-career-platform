import pytest

from app.ingestion.cleaner import normalize_text
from app.ingestion.normalizer import mentions


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
