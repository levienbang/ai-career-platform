from typing import Any

from qdrant_client import QdrantClient

from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.qdrant import SearchIndexNotReadyError, VectorStoreUnavailableError
from app.retrieval.types import SearchHit


class DenseSearchService:
    def __init__(
        self,
        client: QdrantClient,
        embedder: EmbeddingProvider,
        collection_name: str,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.collection_name = collection_name

    def _collection_exists(self) -> bool:
        try:
            return self.client.collection_exists(self.collection_name)
        except Exception as error:
            raise VectorStoreUnavailableError("Qdrant is unavailable") from error

    @staticmethod
    def _hit_from_payload(payload: dict[str, Any], score: float) -> SearchHit:
        return SearchHit(
            job_id=int(payload["job_id"]),
            score=score,
            title=str(payload["title"]),
            company=str(payload["company"]),
            location=payload.get("location"),
            employment_type=payload.get("employment_type"),
            experience_years_min=payload.get("experience_years_min"),
            required_skills=list(payload.get("required_skills", [])),
            preferred_skills=list(payload.get("preferred_skills", [])),
            document_text=str(payload.get("search_text", "")),
        )

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        if not self._collection_exists():
            raise SearchIndexNotReadyError(
                f"Qdrant collection '{self.collection_name}' does not exist; index jobs first"
            )
        vector = self.embedder.embed_query(query)
        if len(vector) != self.embedder.dimensions:
            raise ValueError("Embedding provider returned a query vector with wrong dimensions")
        try:
            points = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            ).points
        except Exception as error:
            raise VectorStoreUnavailableError("Dense search failed") from error
        return [
            self._hit_from_payload(dict(point.payload or {}), float(point.score))
            for point in points
        ]
