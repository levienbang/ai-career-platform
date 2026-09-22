from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.session import get_db
from app.ingestion.extractor import (
    ExtractorConfigurationError,
    StructuredJobExtractor,
    build_job_extractor,
)
from app.ingestion.loader import JobLoadError, load_job_records
from app.ingestion.pipeline import JobIngestionPipeline
from app.schemas.ingestion import ImportResult
from app.schemas.jobs import JobResponse
from app.services.jobs import JobNotFoundError, get_job, list_jobs

router = APIRouter(prefix="/jobs", tags=["jobs"])
DbSession = Annotated[Session, Depends(get_db)]


def get_job_extractor(
    settings: Annotated[Settings, Depends(get_settings)],
) -> StructuredJobExtractor:
    try:
        return build_job_extractor(settings)
    except ExtractorConfigurationError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error


JobExtractor = Annotated[StructuredJobExtractor, Depends(get_job_extractor)]


@router.post("/import", response_model=ImportResult)
async def import_jobs(
    db: DbSession,
    extractor: JobExtractor,
    settings: Annotated[Settings, Depends(get_settings)],
    file: Annotated[UploadFile, File()],
) -> ImportResult:
    content = await file.read(settings.max_import_bytes + 1)
    await file.close()
    if len(content) > settings.max_import_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Import file exceeds {settings.max_import_bytes} bytes",
        )
    try:
        records = load_job_records(file.filename or "", content)
    except JobLoadError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    return JobIngestionPipeline(db, extractor).import_records(records)


@router.get("", response_model=list[JobResponse])
def read_jobs(
    db: DbSession,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[JobResponse]:
    return list_jobs(db, offset=offset, limit=limit)


@router.get("/{job_id}", response_model=JobResponse)
def read_job(job_id: int, db: DbSession) -> JobResponse:
    try:
        return get_job(db, job_id)
    except JobNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Job not found"
        ) from error
