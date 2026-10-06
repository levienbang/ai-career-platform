from sqlalchemy.orm import Session

from app.cv.extractor import StructuredCVExtractor
from app.cv.gap import analyze_skill_gap
from app.cv.pdf import PDFTextLoader
from app.schemas.cv import CVGapReport


class CVSkillGapTool:
    def __init__(
        self, session: Session, loader: PDFTextLoader, extractor: StructuredCVExtractor
    ) -> None:
        self.session = session
        self.loader = loader
        self.extractor = extractor

    def invoke(self, content: bytes, job_ids: list[int], *, max_pages: int) -> CVGapReport:
        text = self.loader.load(content, max_pages=max_pages)
        profile = self.extractor.extract(text)
        return analyze_skill_gap(self.session, profile, text, job_ids)
