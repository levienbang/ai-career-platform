from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.session import get_db
from app.schemas.cv import CVGapReport
from app.services.cv_extractor import CVExtractorError, LangChainCVExtractor
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
        return tool.invoke(content, job_ids, max_pages=settings.max_cv_pages)
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
