"""Split a job search document into reranker-sized chunks.

Job descriptions are stored with whitespace collapsed (no line breaks survive
ingestion), so chunks follow sentence boundaries rather than headings. Every chunk
repeats the document header (title, location, skills) so a chunk taken from the
middle of a description still says which job it belongs to.
"""

import re

DESCRIPTION_MARKER = "\nDescription: "
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")


def split_sentences(text: str) -> list[str]:
    return [sentence for sentence in _SENTENCE_END.split(text.strip()) if sentence]


def _windows(sentences: list[str], max_words: int, overlap_words: int) -> list[str]:
    # A sentence longer than the window is cut into word slices first.
    pieces: list[list[str]] = []
    for sentence in sentences:
        words = sentence.split()
        for start in range(0, len(words), max_words):
            pieces.append(words[start : start + max_words])

    chunks: list[str] = []
    current: list[list[str]] = []
    for piece in pieces:
        if current and sum(map(len, current)) + len(piece) > max_words:
            chunks.append(" ".join(word for part in current for word in part))
            # Carry whole trailing sentences, up to overlap_words, into the next chunk.
            carried: list[list[str]] = []
            for part in reversed(current):
                if sum(map(len, carried)) + len(part) > overlap_words:
                    break
                carried.insert(0, part)
            current = carried
        current.append(piece)
    if current:
        chunks.append(" ".join(word for part in current for word in part))
    return chunks


def chunk_search_document(
    document_text: str,
    *,
    max_words: int = 200,
    overlap_words: int = 40,
    max_chunks: int = 4,
) -> list[str]:
    """Return header-prefixed chunks; a short document stays one chunk, unchanged."""
    if max_words < 1 or max_chunks < 1:
        raise ValueError("max_words and max_chunks must be positive")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be between 0 and max_words - 1")

    header, marker, description = document_text.partition(DESCRIPTION_MARKER)
    if not marker or len(description.split()) <= max_words:
        return [document_text]
    windows = _windows(split_sentences(description), max_words, overlap_words)
    return [f"{header}{DESCRIPTION_MARKER}{window}" for window in windows[:max_chunks]]
