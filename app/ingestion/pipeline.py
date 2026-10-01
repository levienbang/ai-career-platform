import hashlib
import json
import logging
import re

from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.models import Company, Job, JobSkill
from app.db.repositories import CompanyRepository, JobRepository
from app.ingestion.cleaner import clean_job_record
from app.ingestion.extractor import ExtractionError, StructuredJobExtractor
from app.ingestion.normalizer import SkillNormalizer, UnknownSkillsError
from app.ingestion.structured import extract_structured_record
from app.schemas.ingestion import (
    ImportErrorDetail,
    ImportResult,
    JobExtraction,
    RawJobRecord,
)

logger = logging.getLogger(__name__)


class DuplicateJobError(ValueError):
    pass


def build_raw_record_hash(source: RawJobRecord) -> str:
    supplied = getattr(source, "source_record_hash", None)
    if isinstance(supplied, str) and re.fullmatch(r"[0-9a-f]{64}", supplied):
        return supplied
    cleaned = RawJobRecord.model_validate(clean_job_record(source.model_dump(mode="json")))
    encoded = json.dumps(
        cleaned.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_content_hash(job: JobExtraction) -> str:
    payload = {
        "title": job.title.casefold(),
        "company": job.company.casefold() if job.company else None,
        "location": job.location.casefold() if job.location else None,
        "employment_type": job.employment_type.casefold() if job.employment_type else None,
        "experience_years_min": job.experience_years_min,
        "description": " ".join(job.description.split()),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class JobIngestionPipeline:
    def __init__(
        self, session: Session, extractor: StructuredJobExtractor, settings: Settings | None = None
    ) -> None:
        self.session = session
        self.extractor = extractor
        self.settings = settings or get_settings()

    def import_records(self, records: list[dict[str, object]]) -> ImportResult:
        result = ImportResult(processed=len(records), inserted=0, skipped_duplicates=0, failed=0)
        seen_urls: set[str] = set()
        seen_hashes: set[str] = set()
        seen_raw_hashes: set[str] = set()
        prepared: list[tuple[int, RawJobRecord, str]] = []
        extracted_records: dict[int, JobExtraction] = {}
        pending: list[tuple[int, RawJobRecord]] = []
        deterministic = 0
        llm_calls = 0
        skipped_before_llm = 0

        with self.session.begin():
            company_repository = CompanyRepository(self.session)
            job_repository = JobRepository(self.session)

            for index, raw_record in enumerate(records, start=1):
                try:
                    source = RawJobRecord.model_validate(clean_job_record(raw_record))
                    raw_hash = build_raw_record_hash(source)
                    url = str(source.source_url) if source.source_url else None
                    if (
                        raw_hash in seen_raw_hashes
                        or job_repository.get_by_raw_hash(raw_hash) is not None
                        or (url and (url in seen_urls or job_repository.get_by_source_url(url)))
                    ):
                        result.skipped_duplicates += 1
                        skipped_before_llm += 1
                        continue
                    seen_raw_hashes.add(raw_hash)
                    if url:
                        seen_urls.add(url)
                    prepared.append((index, source, raw_hash))
                except ValidationError as error:
                    result.failed += 1
                    result.errors.append(
                        ImportErrorDetail(record=index, error=self._safe_error_message(error))
                    )

            for index, source, _raw_hash in prepared:
                try:
                    extracted = (
                        extract_structured_record(source)
                        if self.settings.llm_structured_fast_path
                        else None
                    )
                    if extracted is None:
                        pending.append((index, source))
                    else:
                        extracted_records[index] = extracted
                        deterministic += 1
                except ValidationError as error:
                    result.failed += 1
                    result.errors.append(
                        ImportErrorDetail(record=index, error=self._safe_error_message(error))
                    )

            for start in range(0, len(pending), self.settings.llm_batch_size):
                group = pending[start : start + self.settings.llm_batch_size]
                calls_before = getattr(self.extractor, "llm_calls", 0)
                try:
                    outputs = self.extractor.extract_many([source for _, source in group])
                except ExtractionError:
                    outputs = [None] * len(group)
                llm_calls += getattr(self.extractor, "llm_calls", calls_before + 1) - calls_before
                for (index, _source), output in zip(group, outputs, strict=True):
                    if output is None:
                        result.failed += 1
                        result.errors.append(
                            ImportErrorDetail(record=index, error="Structured extraction failed")
                        )
                    else:
                        try:
                            extracted_records[index] = JobExtraction.model_validate(output)
                        except ValidationError as error:
                            result.failed += 1
                            result.errors.append(
                                ImportErrorDetail(
                                    record=index, error=self._safe_error_message(error)
                                )
                            )

            seen_urls.clear()
            for index, source, raw_hash in prepared:
                if index not in extracted_records:
                    continue
                try:
                    with self.session.begin_nested():
                        extracted = extracted_records[index]
                        source_url = extracted.source_url or source.source_url
                        source_url_text = str(source_url) if source_url else None

                        content_hash = build_content_hash(extracted)
                        if self._is_duplicate(
                            job_repository,
                            source_url_text,
                            content_hash,
                            seen_urls,
                            seen_hashes,
                        ):
                            raise DuplicateJobError

                        normalized_skills = SkillNormalizer(self.session).normalize(
                            extracted.required_skills,
                            extracted.preferred_skills,
                            evidence_text=(
                                f"{source.title or ''} {source.description} "
                                f"{getattr(source, 'technical_skills', None) or ''}"
                            ),
                        )
                        company = None
                        if extracted.company is not None:
                            company = company_repository.get_by_name(extracted.company)
                            if company is None:
                                company = company_repository.add(
                                    Company(name=extracted.company, location=extracted.location)
                                )
                                self.session.flush()

                        job = Job(
                            title=extracted.title,
                            company=company,
                            location=extracted.location,
                            employment_type=extracted.employment_type,
                            description=extracted.description,
                            experience_years_min=extracted.experience_years_min,
                            source_url=source_url_text,
                            content_hash=content_hash,
                            raw_hash=raw_hash,
                        )
                        job.skills = [
                            JobSkill(
                                skill=item.skill,
                                requirement_type=item.requirement_type,
                                evidence_text=item.evidence_text,
                            )
                            for item in normalized_skills
                        ]
                        job_repository.add(job)
                        self.session.flush()

                    if source_url_text:
                        seen_urls.add(source_url_text)
                    seen_hashes.add(content_hash)
                    result.inserted += 1
                except DuplicateJobError:
                    result.skipped_duplicates += 1
                except (
                    ValidationError,
                    ExtractionError,
                    UnknownSkillsError,
                    IntegrityError,
                ) as error:
                    result.failed += 1
                    result.errors.append(
                        ImportErrorDetail(record=index, error=self._safe_error_message(error))
                    )

        result.errors.sort(key=lambda item: item.record)
        logger.info(
            "processed=%d skipped_before_llm=%d deterministic=%d "
            "llm_records=%d llm_calls=%d failed=%d",
            result.processed,
            skipped_before_llm,
            deterministic,
            len(pending),
            llm_calls,
            result.failed,
        )
        return result

    @staticmethod
    def _is_duplicate(
        repository: JobRepository,
        source_url: str | None,
        content_hash: str,
        seen_urls: set[str],
        seen_hashes: set[str],
    ) -> bool:
        if source_url and (
            source_url in seen_urls or repository.get_by_source_url(source_url) is not None
        ):
            return True
        return (
            content_hash in seen_hashes or repository.get_by_content_hash(content_hash) is not None
        )

    @staticmethod
    def _safe_error_message(error: Exception) -> str:
        if isinstance(error, ValidationError):
            first = error.errors(include_url=False)[0]
            location = ".".join(str(part) for part in first["loc"])
            return f"Validation failed for {location}: {first['msg']}"
        if isinstance(error, IntegrityError):
            return "Database constraint rejected the record"
        return str(error)
