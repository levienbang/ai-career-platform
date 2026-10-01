import math
import re
from collections import Counter

from sqlalchemy.orm import Session

from app.db.repositories import JobRepository
from app.retrieval.documents import SearchDocument, build_search_document
from app.retrieval.types import SearchHit

TOKEN_PATTERN = re.compile(r"[\w+#.]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_PATTERN.findall(text)]


def _to_hit(document: SearchDocument, score: float) -> SearchHit:
    payload = document.payload
    return SearchHit(
        job_id=document.job_id,
        score=score,
        title=document.title,
        company=payload.get("company"),
        location=payload["location"],
        employment_type=payload["employment_type"],
        experience_years_min=payload["experience_years_min"],
        required_skills=list(payload["required_skills"]),
        preferred_skills=list(payload["preferred_skills"]),
        document_text=document.text,
    )


class KeywordSearchService:
    def __init__(self, session: Session, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.session = session
        self.k1 = k1
        self.b = b

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        documents = [build_search_document(job) for job in JobRepository(self.session).list_all()]
        if not documents:
            return []

        corpus_tokens = [tokenize(document.text) for document in documents]
        average_length = sum(map(len, corpus_tokens)) / len(corpus_tokens)
        document_frequency = Counter(token for tokens in corpus_tokens for token in set(tokens))
        query_counts = Counter(query_tokens)
        scored: list[tuple[float, SearchDocument]] = []

        for document, tokens in zip(documents, corpus_tokens, strict=True):
            frequencies = Counter(tokens)
            score = 0.0
            for token, query_frequency in query_counts.items():
                term_frequency = frequencies[token]
                if term_frequency == 0:
                    continue
                frequency = document_frequency[token]
                inverse_document_frequency = math.log(
                    1 + (len(documents) - frequency + 0.5) / (frequency + 0.5)
                )
                length_normalization = 1 - self.b + self.b * len(tokens) / average_length
                score += (
                    query_frequency
                    * inverse_document_frequency
                    * (term_frequency * (self.k1 + 1))
                    / (term_frequency + self.k1 * length_normalization)
                )
            if score > 0:
                scored.append((score, document))

        scored.sort(key=lambda item: (-item[0], item[1].job_id))
        return [_to_hit(document, score) for score, document in scored[:limit]]
