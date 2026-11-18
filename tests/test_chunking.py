import pytest

from app.config import Settings
from app.retrieval.chunking import chunk_search_document, split_sentences
from app.retrieval.reranker import CrossEncoderJobReranker
from app.retrieval.types import SearchHit

HEADER = "Title: NLP Engineer\nLocation: Hà Nội\nRequired skills: Python, LLM"


def _document(description: str) -> str:
    return f"{HEADER}\nDescription: {description}"


def _sentences(count: int, words: int = 10) -> str:
    return " ".join(
        " ".join(f"s{index}w{word}" for word in range(words)) + "." for index in range(count)
    )


def test_split_sentences_keeps_technical_dots() -> None:
    text = "Dùng Node.js và ASP.NET. Yêu cầu: Python; Docker! Có thưởng?"

    assert split_sentences(text) == [
        "Dùng Node.js và ASP.NET.",
        "Yêu cầu: Python;",
        "Docker!",
        "Có thưởng?",
    ]


def test_short_document_stays_one_unchanged_chunk() -> None:
    document = _document(_sentences(3))

    assert chunk_search_document(document, max_words=50) == [document]


def test_document_without_description_marker_is_not_split() -> None:
    assert chunk_search_document("Title: X " + "word " * 500, max_words=50) == [
        "Title: X " + "word " * 500
    ]


def test_every_chunk_repeats_the_header_and_respects_size() -> None:
    chunks = chunk_search_document(
        _document(_sentences(12)), max_words=40, overlap_words=10, max_chunks=10
    )

    assert len(chunks) > 1
    for chunk in chunks:
        header, _, body = chunk.partition("\nDescription: ")
        assert header == HEADER
        assert len(body.split()) <= 40


def test_chunks_overlap_by_whole_sentences() -> None:
    chunks = chunk_search_document(
        _document(_sentences(8)), max_words=40, overlap_words=10, max_chunks=10
    )
    first_body = chunks[0].partition("\nDescription: ")[2]
    second_body = chunks[1].partition("\nDescription: ")[2]

    # 4 ten-word sentences fill the first chunk; the last one is carried over.
    assert first_body.split(". ")[-1] == second_body.split(". ")[0] + "."


def test_overlong_sentence_is_cut_into_word_slices() -> None:
    chunks = chunk_search_document(
        _document(" ".join(f"w{i}" for i in range(95))), max_words=40, overlap_words=0
    )

    assert [len(chunk.partition("\nDescription: ")[2].split()) for chunk in chunks] == [40, 40, 15]


def test_max_chunks_caps_the_cost() -> None:
    chunks = chunk_search_document(
        _document(_sentences(50)), max_words=40, overlap_words=10, max_chunks=3
    )

    assert len(chunks) == 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_words": 0}, "positive"),
        ({"max_chunks": 0}, "positive"),
        ({"max_words": 40, "overlap_words": 40}, "overlap_words"),
    ],
)
def test_invalid_chunk_settings_are_rejected(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        chunk_search_document(_document("x"), **kwargs)


def test_settings_reject_overlap_not_below_chunk_size() -> None:
    with pytest.raises(ValueError, match="RERANKER_CHUNK_OVERLAP_WORDS"):
        Settings(_env_file=None, reranker_chunk_words=40, reranker_chunk_overlap_words=40)


class MatchCountModel:
    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs, **kwargs):
        self.pairs.extend(pairs)
        return [float(passage.count("chatbot")) for _query, passage in pairs]


def _hit(job_id: int, description: str) -> SearchHit:
    return SearchHit(
        job_id=job_id,
        score=0.0,
        title=f"Job {job_id}",
        company=None,
        location=None,
        employment_type=None,
        experience_years_min=None,
        required_skills=[],
        preferred_skills=[],
        document_text=_document(description),
    )


def test_chunked_cross_encoder_scores_a_job_by_its_best_chunk() -> None:
    # The relevant requirement sits past the first window, where truncation would cut it.
    late = _hit(1, _sentences(10) + " chatbot chatbot chatbot.")
    early = _hit(2, "chatbot. " + _sentences(10))
    model = MatchCountModel()
    settings = Settings(
        _env_file=None,
        reranker_chunk_words=40,
        reranker_chunk_overlap_words=10,
        reranker_max_chunks=10,
    )

    ranked = CrossEncoderJobReranker(settings, model=model).rerank("chatbot", [early, late])

    assert [hit.job_id for hit in ranked] == [1, 2]
    assert ranked[0].score == 3.0
    assert len(model.pairs) > 2
