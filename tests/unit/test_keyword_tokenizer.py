import pytest

from app.retrieval.keyword import tokenize, vietnamese_compounds


@pytest.mark.parametrize(
    ("text", "compounds"),
    [
        # Regex only sees syllables: "dữ", "liệu" could come from anywhere in a job.
        ("cơ sở dữ liệu", ["cơ_sở", "dữ_liệu"]),
        ("phát triển ứng dụng di động", ["phát_triển", "ứng_dụng", "di_động"]),
        ("kiểm thử phần mềm tự động", ["kiểm_thử", "phần_mềm", "tự_động"]),
        ("xử lý ngôn ngữ tự nhiên", ["xử_lý", "ngôn_ngữ", "tự_nhiên"]),
        ("thị giác máy tính", ["thị_giác", "máy_tính"]),
    ],
)
def test_pyvi_finds_vietnamese_compounds_that_regex_splits(text, compounds) -> None:
    regex_tokens = tokenize(text, "regex")

    assert all("_" not in token for token in regex_tokens)
    assert vietnamese_compounds(text) == compounds
    assert tokenize(text, "pyvi") == regex_tokens + compounds


def test_regex_keeps_technical_names_and_drops_sentence_dots() -> None:
    tokens = tokenize("Docker. Kubernetes, C++, C# và Node.js, ASP.NET.", "regex")

    assert tokens == ["docker", "kubernetes", "c++", "c#", "và", "node.js", "asp.net"]


def test_pyvi_mode_keeps_technical_names_whole() -> None:
    # pyvi alone would emit "C + +" and "Node . js"; the regex tokens stay intact.
    tokens = tokenize("Lập trình viên C++ và Node.js", "pyvi")

    assert {"c++", "node.js", "lập_trình_viên"} <= set(tokens)
    assert "+" not in tokens


def test_unknown_tokenizer_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported keyword tokenizer"):
        tokenize("Python", "spacy")  # type: ignore[arg-type]
