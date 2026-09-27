from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from app.config import Settings
from app.retrieval.dense import DenseSearchService
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.hybrid import HybridSearchService
from app.retrieval.keyword import KeywordSearchService
from app.retrieval.reranker import JobReranker, RerankedSearchService


def build_hybrid_search_service(
    db: Session,
    embedder: EmbeddingProvider,
    client: QdrantClient,
    settings: Settings,
) -> HybridSearchService:
    return HybridSearchService(
        KeywordSearchService(db),
        DenseSearchService(client, embedder, settings.qdrant_collection),
        candidate_limit=settings.hybrid_candidate_limit,
        rrf_k=settings.rrf_k,
    )


def build_reranked_search_service(
    db: Session,
    embedder: EmbeddingProvider,
    client: QdrantClient,
    reranker: JobReranker,
    settings: Settings,
) -> RerankedSearchService:
    return RerankedSearchService(
        build_hybrid_search_service(db, embedder, client, settings),
        reranker,
        candidate_limit=settings.reranker_max_candidates,
    )
