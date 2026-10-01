"""Use structured source fields without a model request."""

import ast
import re

from app.ingestion.normalizer import skill_lookup_key
from app.schemas.ingestion import JobExtraction, RawJobRecord


def parse_skill_list(value: object) -> list[str]:
    if isinstance(value, str):
        parsed = None
        if value.strip().startswith("["):
            try:
                parsed = ast.literal_eval(value.strip())
            except (ValueError, SyntaxError, TypeError, RecursionError):
                pass
        value = parsed if isinstance(parsed, (list, tuple)) else re.split(r"[,;\n](?![^()]*\))", value)
    if not isinstance(value, (list, tuple)):
        return []
    skills: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        enumeration = re.fullmatch(r"[^()]+\(([^()]*,[^()]*)\)\s*", item.strip())
        values = enumeration[1].split(",") if enumeration else [item]
        for value in values:
            skill = value.strip()
            key = skill_lookup_key(skill)
            if skill and len(skill) <= 100 and key not in seen:
                skills.append(skill)
                seen.add(key)
    return skills


def parse_experience_years(value: object) -> int | None:
    if type(value) is int:
        return value if value >= 0 else None
    if not isinstance(value, str):
        return None
    if value.strip().casefold() in {"không yêu cầu", "no experience"}:
        return 0
    years = re.match(r"^\s*(\d+)\s*(?:năm|years?|yrs?)\b", value, re.IGNORECASE)
    if years:
        return int(years[1])
    months = re.match(r"^\s*(\d+)\s*(?:tháng|months?)\b", value, re.IGNORECASE)
    return int(months[1]) // 12 if months else None


def extract_structured_record(source: RawJobRecord) -> JobExtraction | None:
    required = parse_skill_list(source.required_skills) or parse_skill_list(
        getattr(source, "technical_skills", None)
    )
    preferred = parse_skill_list(source.preferred_skills)
    if not source.title or not source.description or not (required or preferred):
        return None
    experience = source.experience_years_min
    if type(experience) is not int:
        experience = parse_experience_years(getattr(source, "experience", None))
    return JobExtraction(
        title=source.title,
        company=source.company,
        location=source.location,
        employment_type=source.employment_type,
        description=source.description,
        source_url=source.source_url,
        experience_years_min=experience,
        required_skills=required,
        preferred_skills=preferred,
    )
