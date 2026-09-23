from dataclasses import dataclass

from qdrant_client import QdrantClient, models
from sqlalchemy.orm import Session

from app.db.repositories import JobRepository
from app.retrieval.documents import SearchDocument, build_search_document
from app.retrieval.embeddings import EmbeddingProvider


class VectorStoreUnavailableError(RuntimeError):
    pass


class SearchIndexNotReadyError(RuntimeError):
    pass


class CollectionConfigurationError(RuntimeError):
    pass


class IndexedJobNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class IndexResult:
    indexed: int
    deleted: int


def build_qdrant_client(url: str, timeout: float) -> QdrantClient:
    return QdrantClient(url=url, timeout=timeout)


class QdrantJobIndex:
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

    def ensure_collection(self) -> None:
        if not self._collection_exists():
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedder.dimensions,
                        distance=models.Distance.COSINE,
                    ),
                )
                return
            except Exception as error:
                raise VectorStoreUnavailableError(
                    "Could not create the Qdrant collection"
                ) from error

        try:
            vectors = self.client.get_collection(self.collection_name).config.params.vectors
            size = vectors.size if isinstance(vectors, models.VectorParams) else None
        except Exception as error:
            raise VectorStoreUnavailableError("Could not inspect the Qdrant collection") from error
        if size != self.embedder.dimensions:
            raise CollectionConfigurationError(
                f"Collection '{self.collection_name}' uses vector size {size}; "
                f"configured embedding size is {self.embedder.dimensions}. "
                "Rebuild the collection before changing embedding models or dimensions."
            )

    @staticmethod
    def _validate_vectors(
        documents: list[SearchDocument], vectors: list[list[float]], dimensions: int
    ) -> None:
        if len(vectors) != len(documents):
            raise ValueError("Embedding provider returned the wrong number of vectors")
        if any(len(vector) != dimensions for vector in vectors):
            raise ValueError("Embedding provider returned a vector with the wrong dimensions")

    def _upsert_documents(self, documents: list[SearchDocument]) -> int:
        if not documents:
            return 0
        vectors = self.embedder.embed_documents(
            [document.text for document in documents],
            [document.title for document in documents],
        )
        self._validate_vectors(documents, vectors, self.embedder.dimensions)
        points = [
            models.PointStruct(
                id=document.job_id,
                vector=vector,
                payload=document.payload,
            )
            for document, vector in zip(documents, vectors, strict=True)
        ]
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
        except Exception as error:
            raise VectorStoreUnavailableError("Could not upsert jobs into Qdrant") from error
        return len(points)

    def _indexed_point_ids(self) -> set[int]:
        point_ids: set[int] = set()
        offset = None
        try:
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=256,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                point_ids.update(int(point.id) for point in points)
                if offset is None:
                    break
        except Exception as error:
            raise VectorStoreUnavailableError("Could not inspect indexed jobs") from error
        return point_ids

    def index_all(self, session: Session) -> IndexResult:
        self.ensure_collection()
        jobs = JobRepository(session).list_all()
        documents = [build_search_document(job) for job in jobs]
        indexed = self._upsert_documents(documents)

        database_ids = {document.job_id for document in documents}
        stale_ids = sorted(self._indexed_point_ids() - database_ids)
        if stale_ids:
            try:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=stale_ids),
                    wait=True,
                )
            except Exception as error:
                raise VectorStoreUnavailableError(
                    "Could not remove stale jobs from Qdrant"
                ) from error
        return IndexResult(indexed=indexed, deleted=len(stale_ids))

    def index_job(self, session: Session, job_id: int) -> IndexResult:
        job = JobRepository(session).get(job_id)
        if job is None:
            raise IndexedJobNotFoundError(job_id)
        self.ensure_collection()
        indexed = self._upsert_documents([build_search_document(job)])
        return IndexResult(indexed=indexed, deleted=0)
