"""Download the local cross-encoder reranker and rerank three sample jobs once.

The model lands in the Hugging Face cache (~/.cache/huggingface, ~2.3 GB for
bge-reranker-v2-m3); later runs load it from disk. Never run by the test suite.
"""

import argparse
import sys
import time

from app.config import Settings
from app.retrieval.reranker import CrossEncoderJobReranker, RerankerConfigurationError
from app.retrieval.types import SearchHit

SAMPLE_QUERY = "kỹ sư xử lý ngôn ngữ tự nhiên xây chatbot"
SAMPLE_JOBS = [
    ("Android Developer", "Phát triển ứng dụng di động bằng Kotlin."),
    ("NLP Engineer", "Xây dựng chatbot và mô hình ngôn ngữ lớn (LLM) cho khách hàng."),
    ("Data Analyst", "Làm báo cáo Power BI và phân tích số liệu kinh doanh."),
]


def _hit(job_id: int, title: str, description: str) -> SearchHit:
    return SearchHit(
        job_id=job_id,
        score=0.0,
        title=title,
        company=None,
        location=None,
        employment_type=None,
        experience_years_min=None,
        required_skills=[],
        preferred_skills=[],
        document_text=f"Title: {title}\nDescription: {description}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Override RERANKER_MODEL")
    args = parser.parse_args()
    settings = Settings(**({"reranker_model": args.model} if args.model else {}))

    started = time.perf_counter()
    try:
        reranker = CrossEncoderJobReranker(settings)
    except RerankerConfigurationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"model={settings.reranker_model} loaded in {time.perf_counter() - started:.1f}s")

    candidates = [_hit(index, *job) for index, job in enumerate(SAMPLE_JOBS, start=1)]
    started = time.perf_counter()
    ranked = reranker.rerank(SAMPLE_QUERY, candidates)
    print(f"query={SAMPLE_QUERY!r} reranked in {(time.perf_counter() - started) * 1000:.0f}ms")
    for rank, hit in enumerate(ranked, start=1):
        print(f"{rank}. {hit.title:<18} score={hit.score:.4f}")
    if ranked[0].title != "NLP Engineer":
        print("error: the NLP job should rank first", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
