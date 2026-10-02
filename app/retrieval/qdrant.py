import time
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256

from qdrant_client import QdrantClient, models
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.repositories import JobRepository
from app.ingestion.extractor import RequestThrottle
from app.retrieval.documents import SearchDocument, build_search_document
from app.retrieval.embeddings import EmbeddingProvider, EmbeddingServiceError


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
    skipped: int = 0
    refreshed: int = 0


def document_hash(document: SearchDocument) -> str:
    return sha256(document.text.encode("utf-8")).hexdigest()


def embedding_retry_details(error: Exception) -> tuple[bool, float | None]:
    """Inspect wrapped provider errors without logging payloads or credentials."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        code = getattr(current, "status_code", None) or getattr(response, "status_code", None)
        code = code or getattr(current, "code", None)
        if callable(code):
            code = code()
        try:
            status = int(code)
        except (TypeError, ValueError):
            status = 429 if "RESOURCE_EXHAUSTED" in str(current) else 0
        if status == 429 or 500 <= status <= 599:
            headers = getattr(response, "headers", None) or getattr(current, "headers", {})
            try:
                delay = max(0.0, float(headers.get("retry-after")))
            except (TypeError, ValueError):
                delay = None
            return True, delay
        current = current.__cause__ or current.__context__
    return False, None


def build_qdrant_client(url: str, timeout: float) -> QdrantClient:
    return QdrantClient(url=url, timeout=timeout)


class QdrantJobIndex:
    def __init__(
        self,
        client: QdrantClient,
        embedder: EmbeddingProvider | None,
        collection_name: str,
        *,
        settings: Settings | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.collection_name = collection_name
        self.settings = settings or get_settings()
        self.sleep = sleep
        self.progress = progress
        self.throttle = RequestThrottle(
            self.settings.embedding_requests_per_minute, clock=clock, sleep=sleep
        )

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
        for attempt in range(self.settings.embedding_max_retries + 1):
            self.throttle.wait()
            try:
                vectors = self.embedder.embed_documents(
                    [document.text for document in documents],
                    [document.title for document in documents],
                )
                break
            except Exception as error:
                retryable, retry_after = embedding_retry_details(error)
                if not retryable or attempt == self.settings.embedding_max_retries:
                    raise EmbeddingServiceError("Document embedding request failed") from error
                self.sleep(
                    retry_after
                    if retry_after is not None
                    else min(120, self.settings.embedding_retry_base_seconds * 2**attempt)
                )
        self._validate_vectors(documents, vectors, self.embedder.dimensions)
        points = [
            models.PointStruct(
                id=document.job_id,
                vector=vector,
                payload={**document.payload, "document_hash": document_hash(document)},
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

    def _indexed_documents(self) -> dict[int, str | None]:
        hashes: dict[int, str | None] = {}
        offset = None
        try:
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=256,
                    offset=offset,
                    with_payload=["document_hash"],
                    with_vectors=False,
                )
                hashes.update(
                    (int(point.id), (point.payload or {}).get("document_hash")) for point in points
                )
                if offset is None:
                    break
        except Exception as error:
            raise VectorStoreUnavailableError("Could not inspect indexed jobs") from error
        return hashes

    def index_all(self, session: Session, *, full: bool = False) -> IndexResult:
        self.ensure_collection()
        jobs = JobRepository(session).list_all()
        documents = [build_search_document(job) for job in jobs]
        previous = self._indexed_documents()
        changed = [
            document
            for document in documents
            if full or previous.get(document.job_id) != document_hash(document)
        ]
        indexed = 0
        for start in range(0, len(changed), self.settings.embedding_batch_size):
            batch = changed[start : start + self.settings.embedding_batch_size]
            try:
                indexed += self._upsert_documents(batch)
            except EmbeddingServiceError as error:
                raise EmbeddingServiceError(
                    f"Embedding failed after indexing {indexed}/{len(changed)} jobs; "
                    "completed batches are saved. Run indexing again to resume."
                ) from error
            if self.progress:
                self.progress(indexed, len(changed))

        database_ids = {document.job_id for document in documents}
        stale_ids = sorted(previous.keys() - database_ids)
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
        return IndexResult(
            indexed=indexed, deleted=len(stale_ids), skipped=len(documents) - len(changed)
        )

    def refresh_payloads(self, session: Session) -> IndexResult:
        """Rewrite display payloads and hashes of indexed jobs without embedding.

        Use after changes that keep a job's meaning, such as renamed or merged
        skills or location casing: Qdrant shows the new values and the next
        incremental index does not re-embed those jobs. Jobs not yet in Qdrant are
        left for index_all.
        """
        if not self._collection_exists():
            raise SearchIndexNotReadyError(
                f"Qdrant collection '{self.collection_name}' does not exist; index jobs first"
            )
        indexed = self._indexed_documents()
        refreshed = 0
        try:
            for job in JobRepository(session).list_all():
                if job.id not in indexed:
                    continue
                document = build_search_document(job)
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={**document.payload, "document_hash": document_hash(document)},
                    points=[job.id],
                    wait=False,
                )
                refreshed += 1
        except Exception as error:
            raise VectorStoreUnavailableError("Could not refresh Qdrant payloads") from error
        return IndexResult(indexed=0, deleted=0, refreshed=refreshed)

    def index_job(self, session: Session, job_id: int) -> IndexResult:
        job = JobRepository(session).get(job_id)
        if job is None:
            raise IndexedJobNotFoundError(job_id)
        self.ensure_collection()
        indexed = self._upsert_documents([build_search_document(job)])
        return IndexResult(indexed=indexed, deleted=0)
