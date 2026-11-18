import re
import unicodedata
from typing import Any

MISSING_VALUES = {"", "n/a", "na", "none", "null"}
FIELD_ALIASES = {
    "job_description": "description",
    "description_text": "description",
    "raw_description": "description",
    "company_name": "company",
    "min_experience": "experience_years_min",
}


def normalize_text(value: str) -> str:
    """NFKC-normalize and collapse every whitespace run to one space."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def clean_text(value: str) -> str | None:
    normalized = normalize_text(value)
    if normalized.casefold() in MISSING_VALUES:
        return None
    return normalized


def _clean_value(value: Any) -> Any:
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := _clean_value(item)) is not None]
    return value


def clean_job_record(record: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for raw_key, raw_value in record.items():
        key = clean_text(str(raw_key))
        if key is None:
            continue
        canonical_key = FIELD_ALIASES.get(key.casefold(), key)
        if canonical_key in {
            "required_skills",
            "preferred_skills",
            "technical_skills",
        } and isinstance(raw_value, str):
            # Newlines delimit source skills; ordinary text whitespace is still collapsed.
            cleaned[canonical_key] = (
                "\n".join(line for value in raw_value.splitlines() if (line := clean_text(value)))
                or None
            )
        else:
            cleaned[canonical_key] = _clean_value(raw_value)
    return cleaned
