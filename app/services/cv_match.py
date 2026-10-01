"""Match a CV to indexed jobs using semantic and skill evidence."""

from typing import Protocol

from sqlalchemy.orm import Session

from app.config import Settings
from app.db.repositories import JobRepository
from app.retrieval.types import SearchHit
from app.schemas.cv import CVExtraction, CVMatchedJob, CVMatchReport, RecognizedSkill
from app.services.cv_extractor import StructuredCVExtractor
from app.services.cv_pdf import PDFTextLoader
from app.services.skill_gap import _job_gap, recognize_cv_skills


class DenseSearchBackend(Protocol):
    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]: ...


def build_cv_query(profile: CVExtraction, recognized: dict[int, RecognizedSkill]) -> str:
    parts = [
        profile.summary or "",
        ", ".join(item.skill for item in recognized.values()),
        "\n".join(profile.projects),
        "\n".join(profile.experience),
        "\n".join(profile.education),
    ]
    return "\n".join(part.strip() for part in parts if part.strip())[:2000]


class CVMatchService:
    def __init__(
        self,
        session: Session,
        loader: PDFTextLoader,
        extractor: StructuredCVExtractor,
        dense_search: DenseSearchBackend,
        settings: Settings,
    ) -> None:
        self.session = session
        self.loader = loader
        self.extractor = extractor
        self.dense_search = dense_search
        self.settings = settings

    def invoke(self, content: bytes, *, limit: int, max_pages: int) -> CVMatchReport:
        cv_text = self.loader.load(content, max_pages=max_pages)
        profile = self.extractor.extract(cv_text)
        recognized, unknown = recognize_cv_skills(self.session, profile, cv_text)
        query = build_cv_query(profile, recognized)
        hits = self.dense_search.search(query, limit=self.settings.cv_match_candidates)
        jobs_by_id = {
            job.id: job
            for job in JobRepository(self.session).get_many([hit.job_id for hit in hits])
        }
        weight = self.settings.cv_match_semantic_weight
        matches: list[CVMatchedJob] = []
        for hit in hits:
            job = jobs_by_id.get(hit.job_id)
            if job is None:
                continue
            gap = _job_gap(job, set(recognized))
            matches.append(
                CVMatchedJob(
                    job_id=job.id,
                    title=job.title,
                    company=job.company.name if job.company else None,
                    location=job.location,
                    final_score=weight * hit.score + (1 - weight) * gap.fit_score,
                    semantic_score=hit.score,
                    skill_score=gap.fit_score,
                    matched_skills=gap.matched_skills,
                    missing_required_skills=gap.missing_required_skills,
                    missing_preferred_skills=gap.missing_preferred_skills,
                )
            )
        matches.sort(key=lambda item: (-item.final_score, -item.semantic_score, item.job_id))
        return CVMatchReport(
            recognized_skills=sorted(recognized.values(), key=lambda item: item.skill),
            unrecognized_skills=sorted(unknown),
            query=query,
            jobs=matches[:limit],
        )
