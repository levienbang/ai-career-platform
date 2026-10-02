from time import perf_counter
from typing import Annotated, Literal, NoReturn

from fastapi import APIRouter, Depends, HTTPException, status
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.session import get_db
from app.retrieval.dense import DenseSearchService
from app.retrieval.embeddings import (
    EmbeddingConfigurationError,
    EmbeddingProvider,
    EmbeddingServiceError,
    build_embedding_provider,
)
from app.retrieval.factory import build_hybrid_search_service, build_reranked_search_service
from app.retrieval.keyword import KeywordSearchService
from app.retrieval.qdrant import (
    CollectionConfigurationError,
    IndexedJobNotFoundError,
    QdrantJobIndex,
    SearchIndexNotReadyError,
    VectorStoreUnavailableError,
    build_qdrant_client,
)
from app.retrieval.reranker import (
    JobReranker,
    RerankerConfigurationError,
    RerankerInputError,
    RerankerResponseError,
    RerankerServiceError,
    build_reranker,
)
from app.retrieval.types import SearchHit
from app.schemas.search import (
    IndexRequest,
    IndexResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

router = APIRouter(prefix="/search", tags=["search"])
DbSession = Annotated[Session, Depends(get_db)]


def get_embedding_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> EmbeddingProvider:
    try:
        return build_embedding_provider(settings)
    except EmbeddingConfigurationError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error


def get_vector_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> QdrantClient:
    return build_qdrant_client(settings.qdrant_url, settings.qdrant_timeout_seconds)


def get_job_reranker(
    settings: Annotated[Settings, Depends(get_settings)],
) -> JobReranker:
    try:
        return build_reranker(settings)
    except RerankerConfigurationError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error


EmbeddingDependency = Annotated[EmbeddingProvider, Depends(get_embedding_provider)]
QdrantDependency = Annotated[QdrantClient, Depends(get_vector_client)]
SettingsDependency = Annotated[Settings, Depends(get_settings)]
RerankerDependency = Annotated[JobReranker, Depends(get_job_reranker)]


def _response(
    method: Literal["keyword", "dense", "hybrid", "reranked"],
    request: SearchRequest,
    hits: list[SearchHit],
    started_at: float,
) -> SearchResponse:
    return SearchResponse(
        method=method,
        query=request.query,
        latency_ms=(perf_counter() - started_at) * 1000,
        results=[SearchResult.model_validate(hit.__dict__) for hit in hits],
    )


def _raise_retrieval_http_error(error: Exception) -> NoReturn:
    if isinstance(error, IndexedJobNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Job not found"
        ) from error
    if isinstance(error, SearchIndexNotReadyError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error
    if isinstance(error, CollectionConfigurationError):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    if isinstance(error, (EmbeddingServiceError, VectorStoreUnavailableError)):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error
    if isinstance(error, RerankerInputError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error
    if isinstance(error, (RerankerServiceError, RerankerResponseError)):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error
    raise error


@router.post("/keyword", response_model=SearchResponse)
def keyword_search(request: SearchRequest, db: DbSession) -> SearchResponse:
    started_at = perf_counter()
    hits = KeywordSearchService(db).search(request.query, limit=request.limit)
    return _response("keyword", request, hits, started_at)


@router.post("/semantic", response_model=SearchResponse)
def dense_search(
    request: SearchRequest,
    embedder: EmbeddingDependency,
    client: QdrantDependency,
    settings: SettingsDependency,
) -> SearchResponse:
    started_at = perf_counter()
    try:
        hits = DenseSearchService(client, embedder, settings.qdrant_collection).search(
            request.query, limit=request.limit
        )
    except Exception as error:
        _raise_retrieval_http_error(error)
    return _response("dense", request, hits, started_at)


@router.post("/hybrid", response_model=SearchResponse)
def hybrid_search(
    request: SearchRequest,
    db: DbSession,
    embedder: EmbeddingDependency,
    client: QdrantDependency,
    settings: SettingsDependency,
) -> SearchResponse:
    started_at = perf_counter()
    try:
        hits = build_hybrid_search_service(db, embedder, client, settings).search(
            request.query, limit=request.limit
        )
    except Exception as error:
        _raise_retrieval_http_error(error)
    return _response("hybrid", request, hits, started_at)


@router.post("/reranked", response_model=SearchResponse)
def reranked_search(
    request: SearchRequest,
    db: DbSession,
    embedder: EmbeddingDependency,
    client: QdrantDependency,
    settings: SettingsDependency,
    reranker: RerankerDependency,
) -> SearchResponse:
    started_at = perf_counter()
    try:
        hits = build_reranked_search_service(db, embedder, client, reranker, settings).search(
            request.query, limit=request.limit
        )
    except Exception as error:
        _raise_retrieval_http_error(error)
    return _response("reranked", request, hits, started_at)


@router.post("/index", response_model=IndexResponse)
def index_jobs(
    request: IndexRequest,
    db: DbSession,
    embedder: EmbeddingDependency,
    client: QdrantDependency,
    settings: SettingsDependency,
) -> IndexResponse:
    index = QdrantJobIndex(client, embedder, settings.qdrant_collection, settings=settings)
    try:
        result = (
            index.index_job(db, request.job_id)
            if request.job_id is not None
            else index.index_all(db)
        )
    except Exception as error:
        _raise_retrieval_http_error(error)
    return IndexResponse(
        collection=settings.qdrant_collection,
        indexed=result.indexed,
        deleted=result.deleted,
    )
