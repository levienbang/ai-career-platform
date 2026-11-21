import pytest

from app.ingestion.normalizer import skill_match_key
from app.ingestion.structured import parse_skill_list
from app.skills.taxonomy import (
    load_skill_aliases,
)


@pytest.mark.parametrize(
    "name,key",
    [
        ("Node.js", "nodejs"),
        ("NodeJS", "nodejs"),
        ("node js", "nodejs"),
        ("Power BI", "powerbi"),
        ("PowerBI", "powerbi"),
        ("PL/SQL", "plsql"),
        ("PLSQL", "plsql"),
        ("C++", "c++"),
        ("C#", "c#"),
        ("C", "c"),
        (".NET", "net"),
        ("Scikit-learn", "scikitlearn"),
        (" Ｐｏｗｅｒ＿ＢＩ\t", "powerbi"),
    ],
)
def test_match_key(name, key):
    assert skill_match_key(name) == key


def test_alias_file_has_unique_owners():
    definitions = load_skill_aliases()
    assert 40 <= len(definitions) <= 60
    canonical_keys = [skill_match_key(name) for name in definitions]
    assert len(set(canonical_keys)) == len(canonical_keys)
    owners = {}
    for canonical, definition in definitions.items():
        assert definition["category"] is None or isinstance(definition["category"], str)
        for name in [canonical, *definition["aliases"]]:
            assert isinstance(name, str)
            key = skill_match_key(name)
            assert owners.setdefault(key, canonical) == canonical
            if key in {"r", "c", "go"}:
                assert name == canonical


@pytest.mark.parametrize("as_string", [False, True])
def test_parenthesized_skill_enumeration(as_string):
    phrase = "Thư viện học máy (Scikit-learn, TensorFlow, Keras, PyTorch)"
    value = phrase + "; Docker" if as_string else [phrase, "Docker", "Keras"]
    assert parse_skill_list(value) == ["Scikit-learn", "TensorFlow", "Keras", "PyTorch", "Docker"]
    assert parse_skill_list(["SQL (advanced)", "C++"]) == ["SQL (advanced)", "C++"]
