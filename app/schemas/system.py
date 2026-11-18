from pydantic import BaseModel


class ReadinessResponse(BaseModel):
    status: str
    postgres: str
    qdrant: str
    # Local cross-encoder warm-up: not_used, loading, loaded or failed. Informational only:
    # search still works while it loads (the first reranked request waits).
    reranker: str = "not_used"
