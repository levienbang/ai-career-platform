from app.retrieval.keyword import KeywordSearchService
from tests.factories import add_job


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
