import pytest

from app.retrieval.keyword import KeywordSearchService, tokenize, vietnamese_compounds
from tests.factories import add_job


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


def test_pyvi_ranks_the_real_phrase_above_scattered_syllables(db_session) -> None:
    mobile = add_job(
        db_session,
        title="Mobile Developer",
        description="Phát triển ứng dụng di động bằng Flutter cho khách hàng doanh nghiệp.",
        company_name="Mobile",
        source_url="https://example.com/tokenizer-mobile",
    )
    recruiter = add_job(
        db_session,
        title="Recruiter",
        description="Tuyển dụng ứng viên, di chuyển linh động.",
        company_name="People",
        source_url="https://example.com/tokenizer-recruiter",
    )
    query = "ứng dụng di động"

    regex_top = KeywordSearchService(db_session, tokenizer="regex").search(query)[0]
    pyvi_top = KeywordSearchService(db_session, tokenizer="pyvi").search(query)[0]

    # Same four syllables, shorter text: plain BM25 prefers the recruiter job.
    assert regex_top.job_id == recruiter.id
    assert pyvi_top.job_id == mobile.id
