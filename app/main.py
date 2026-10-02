import logging
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from qdrant_client import QdrantClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.agent import router as agent_router
from app.api.cv import router as cv_router
from app.api.jobs import router as jobs_router
from app.api.search import router as search_router
from app.config import Settings, get_settings
from app.db.session import get_db
from app.retrieval.qdrant import build_qdrant_client
from app.schemas.system import ReadinessResponse

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")

app = FastAPI(title="AI Career Intelligence Platform", version="0.1.0")
app.include_router(jobs_router)
app.include_router(search_router)
app.include_router(agent_router)
app.include_router(cv_router)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


def get_readiness_qdrant(
    settings: Annotated[Settings, Depends(get_settings)],
) -> QdrantClient:
    return build_qdrant_client(settings.qdrant_url, settings.qdrant_timeout_seconds)


@app.get("/ready", tags=["system"], response_model=ReadinessResponse)
def ready(
    db: Annotated[Session, Depends(get_db)],
    qdrant: Annotated[QdrantClient, Depends(get_readiness_qdrant)],
) -> ReadinessResponse:
    try:
        db.execute(text("SELECT 1"))
        qdrant.get_collections()
    except Exception as error:
        raise HTTPException(status_code=503, detail="Dependency unavailable") from error
    return ReadinessResponse(status="ready", postgres="ok", qdrant="ok")
