import hashlib
import json

from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import Company, Job, JobSkill
from app.db.repositories import CompanyRepository, JobRepository
from app.ingestion.cleaner import clean_job_record
from app.ingestion.extractor import ExtractionError, StructuredJobExtractor
from app.ingestion.normalizer import SkillNormalizer, UnknownSkillsError
from app.schemas.ingestion import (
    ImportErrorDetail,
    ImportResult,
    JobExtraction,
    RawJobRecord,
)


class DuplicateJobError(ValueError):
    pass


def build_content_hash(job: JobExtraction) -> str:
    payload = {
        "title": job.title.casefold(),
        "company": job.company.casefold(),
        "location": job.location.casefold() if job.location else None,
        "employment_type": job.employment_type.casefold() if job.employment_type else None,
        "experience_years_min": job.experience_years_min,
        "description": " ".join(job.description.split()),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class JobIngestionPipeline:
    def __init__(self, session: Session, extractor: StructuredJobExtractor) -> None:
        self.session = session
        self.extractor = extractor

    def import_records(self, records: list[dict[str, object]]) -> ImportResult:
        result = ImportResult(processed=len(records), inserted=0, skipped_duplicates=0, failed=0)
        seen_urls: set[str] = set()
        seen_hashes: set[str] = set()

        with self.session.begin():
            normalizer = SkillNormalizer(self.session)
            company_repository = CompanyRepository(self.session)
            job_repository = JobRepository(self.session)

            for index, raw_record in enumerate(records, start=1):
                try:
                    with self.session.begin_nested():
                        cleaned = clean_job_record(raw_record)
                        source = RawJobRecord.model_validate(cleaned)
                        extracted = JobExtraction.model_validate(self.extractor.extract(source))
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

                        normalized_skills = normalizer.normalize(
                            extracted.required_skills, extracted.preferred_skills
                        )
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
