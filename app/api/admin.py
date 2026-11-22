"""Admin endpoints behind the chat UI's admin buttons, replacing manual scripts.

Every route needs the admin key (X-API-Key) once ADMIN_API_KEY is set. Import and
index already live in /jobs/import and /search/index; this router adds statistics,
skill-decision review and a one-request LLM check.
"""

import logging
from time import perf_counter
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.models import Company, Job, Skill, SkillDecision
from app.db.session import get_db
from app.fallbacks import llm_fallbacks, reranker_fallbacks
from app.llm import check_structured_output
from app.retrieval.qdrant import build_qdrant_client
from app.security import AdminKey, paid_call_limiter
from app.skills.review import approve_decision, keep_new_decision, revert_alias

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[AdminKey])
DbSession = Annotated[Session, Depends(get_db)]
SettingsDependency = Annotated[Settings, Depends(get_settings)]
DecisionName = Literal["pending", "alias", "new", "rejected"]


class AdminStats(BaseModel):
    jobs: int
    companies: int
    skills: int
    skill_decisions: dict[str, int]
    indexed_jobs: int | None = Field(description="Qdrant points; null if Qdrant is down")
    paid_calls_today: int = Field(description="Rate-limited paid requests since 00:00 UTC")
    reranker_fallbacks: dict[str, int] = Field(description="Since the API process started")
    llm_fallbacks: dict[str, int] = Field(
        default_factory=dict, description="DeepSeek calls served by the backup model"
    )


class SkillDecisionItem(BaseModel):
    name: str
    decision: str
    skill: str | None
    confidence: float | None
    reason: str | None


class SkillDecisionPage(BaseModel):
    total: int
    items: list[SkillDecisionItem]


class DecisionNameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class ApproveRequest(DecisionNameRequest):
    canonical: str = Field(min_length=1, max_length=255)


class ReviewResult(BaseModel):
    name: str
    decision: str
    affected_job_links: int | None = None
    reindex_needed: bool = True


class LLMCheckResponse(BaseModel):
    ok: bool
    model: str
    latency_ms: float


@router.get("/stats", response_model=AdminStats)
def stats(db: DbSession, settings: SettingsDependency) -> AdminStats:
    decisions = dict(
        db.execute(
            select(SkillDecision.decision, func.count()).group_by(SkillDecision.decision)
        ).all()
    )
    indexed: int | None = None
    try:
        client = build_qdrant_client(settings.qdrant_url, settings.qdrant_timeout_seconds)
        try:
            indexed = client.count(settings.qdrant_collection, exact=True).count
        finally:
            client.close()
    except Exception as error:
        logger.warning("Qdrant count failed (%s)", type(error).__name__)
    return AdminStats(
        jobs=db.scalar(select(func.count()).select_from(Job)) or 0,
        companies=db.scalar(select(func.count()).select_from(Company)) or 0,
        skills=db.scalar(select(func.count()).select_from(Skill)) or 0,
        skill_decisions=decisions,
        indexed_jobs=indexed,
        paid_calls_today=paid_call_limiter.calls_today(),
        reranker_fallbacks=reranker_fallbacks.snapshot(),
        llm_fallbacks=llm_fallbacks.snapshot(),
    )


@router.get("/skill-decisions", response_model=SkillDecisionPage)
def list_skill_decisions(
    db: DbSession,
    decision: DecisionName = "pending",
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> SkillDecisionPage:
    where = SkillDecision.decision == decision
    total = db.scalar(select(func.count()).select_from(SkillDecision).where(where)) or 0
    rows = db.execute(
        select(SkillDecision, Skill.canonical_name)
        .outerjoin(Skill, Skill.id == SkillDecision.skill_id)
        .where(where)
        .order_by(SkillDecision.confidence.desc().nulls_last(), SkillDecision.id)
        .limit(limit)
        .offset(offset)
    ).all()
    return SkillDecisionPage(
        total=total,
        items=[
            SkillDecisionItem(
                name=item.name,
                decision=item.decision,
                skill=skill_name,
                confidence=float(item.confidence) if item.confidence is not None else None,
                reason=item.reason,
            )
            for item, skill_name in rows
        ],
    )


def _review(db: Session, action) -> object:
    try:
        result = action()
        db.commit()
        return result
    except ValueError as error:
        db.rollback()
        code = (
            status.HTTP_404_NOT_FOUND
            if "not found" in str(error).lower() or "no matching" in str(error).lower()
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=code, detail=str(error)) from error
    except Exception:
        db.rollback()
        raise


@router.post("/skill-decisions/approve", response_model=ReviewResult)
def approve(request: ApproveRequest, db: DbSession, settings: SettingsDependency) -> ReviewResult:
    """Merge a pending name into an existing canonical skill (as an alias)."""
    _review(db, lambda: approve_decision(db, request.name, request.canonical, settings=settings))
    return ReviewResult(name=request.name, decision="alias")


@router.post("/skill-decisions/keep-new", response_model=ReviewResult)
def keep_new(request: DecisionNameRequest, db: DbSession) -> ReviewResult:
    """Keep a pending name as its own skill."""
    _review(db, lambda: keep_new_decision(db, request.name))
    return ReviewResult(name=request.name, decision="new", reindex_needed=False)


@router.post("/skill-decisions/revert", response_model=ReviewResult)
def revert(request: DecisionNameRequest, db: DbSession) -> ReviewResult:
    """Undo an alias the LLM created; only source='llm' aliases can be reverted."""
    affected = _review(db, lambda: revert_alias(db, request.name))
    return ReviewResult(name=request.name, decision="rejected", affected_job_links=affected)


@router.post("/check-llm", response_model=LLMCheckResponse)
async def check_llm(settings: SettingsDependency) -> LLMCheckResponse:
    """Spend one DeepSeek request to confirm the key, network and structured output."""
    started = perf_counter()
    try:
        model, result = await run_in_threadpool(check_structured_output, settings)
    except Exception as error:
        # Provider errors may echo request headers; return only the error type.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM check failed ({type(error).__name__})",
        ) from error
    return LLMCheckResponse(ok=result.ok, model=model, latency_ms=(perf_counter() - started) * 1000)
