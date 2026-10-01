from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from app.api.search import get_embedding_provider, get_vector_client
from app.config import Settings, get_settings
from app.db.session import get_db
from app.retrieval.dense import DenseSearchService
from app.retrieval.embeddings import EmbeddingProvider, EmbeddingServiceError
from app.retrieval.qdrant import SearchIndexNotReadyError, VectorStoreUnavailableError
from app.schemas.cv import CVGapReport, CVMatchReport
from app.services.cv_extractor import CVExtractorError, LangChainCVExtractor
from app.services.cv_match import CVMatchService
from app.services.cv_pdf import (
    CVPDFError,
    DoclingPDFTextLoader,
    PDFParserUnavailableError,
    validate_cv_pdf,
)
from app.services.skill_gap import TargetJobError
from app.tools.cv_tool import CVSkillGapTool

router = APIRouter(prefix="/cv", tags=["cv"])


def get_cv_tool(
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> CVSkillGapTool:
    try:
        return CVSkillGapTool(db, DoclingPDFTextLoader(), LangChainCVExtractor(settings))
    except CVExtractorError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def get_cv_match_service(
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    client: Annotated[QdrantClient, Depends(get_vector_client)],
) -> CVMatchService:
    try:
        return CVMatchService(
            db,
            DoclingPDFTextLoader(),
            LangChainCVExtractor(settings),
            DenseSearchService(client, embedder, settings.qdrant_collection),
            settings,
        )
    except CVExtractorError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/upload", response_model=CVGapReport)
async def upload_cv(
    file: Annotated[UploadFile, File()],
    job_ids: Annotated[list[int], Form()],
    settings: Annotated[Settings, Depends(get_settings)],
    tool: Annotated[CVSkillGapTool, Depends(get_cv_tool)],
) -> CVGapReport:
    content = await file.read(settings.max_cv_bytes + 1)
    await file.close()
    try:
        validate_cv_pdf(
            content,
            filename=file.filename or "",
            content_type=file.content_type or "",
            max_bytes=settings.max_cv_bytes,
        )
        if not job_ids or len(job_ids) > 20 or any(job_id < 1 for job_id in job_ids):
            raise HTTPException(status_code=422, detail="Provide 1 to 20 valid target job IDs")
        # PDF parsing and LLM extraction block; keep them off the event loop.
        return await run_in_threadpool(
            tool.invoke, content, job_ids, max_pages=settings.max_cv_pages
        )
    except CVPDFError as error:
        code = (
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            if len(content) > settings.max_cv_bytes
            else 400
        )
        raise HTTPException(status_code=code, detail=str(error)) from error
    except TargetJobError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except CVExtractorError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except PDFParserUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/match", response_model=CVMatchReport)
async def match_cv(
    file: Annotated[UploadFile, File()],
    settings: Annotated[Settings, Depends(get_settings)],
    service: Annotated[CVMatchService, Depends(get_cv_match_service)],
    limit: Annotated[int, Form(ge=1, le=20)] = 10,
) -> CVMatchReport:
    content = await file.read(settings.max_cv_bytes + 1)
    await file.close()
    try:
        validate_cv_pdf(
            content,
            filename=file.filename or "",
            content_type=file.content_type or "",
            max_bytes=settings.max_cv_bytes,
        )
        return await run_in_threadpool(
            service.invoke, content, limit=limit, max_pages=settings.max_cv_pages
        )
    except CVPDFError as error:
        code = 413 if len(content) > settings.max_cv_bytes else 400
        raise HTTPException(status_code=code, detail=str(error)) from error
    except (CVExtractorError, PDFParserUnavailableError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (SearchIndexNotReadyError, EmbeddingServiceError, VectorStoreUnavailableError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
