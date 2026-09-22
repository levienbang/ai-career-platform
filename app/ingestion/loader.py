import csv
import io
import json
from pathlib import Path
from typing import Any


class JobLoadError(ValueError):
    pass


def _ensure_record_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict) and "jobs" in value:
        value = value["jobs"]
    if not isinstance(value, list):
        raise JobLoadError("JSON must contain a list of job objects or a 'jobs' list")
    if not all(isinstance(item, dict) for item in value):
        raise JobLoadError("Every job record must be a JSON object")
    return value


def load_job_records(filename: str, content: bytes) -> list[dict[str, Any]]:
    suffix = Path(filename).suffix.casefold()
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise JobLoadError("Import files must use UTF-8 encoding") from error

    if suffix == ".json":
        try:
            return _ensure_record_list(json.loads(text))
        except json.JSONDecodeError as error:
            raise JobLoadError(f"Invalid JSON: {error.msg}") from error

    if suffix == ".csv":
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise JobLoadError("CSV must contain a header row")
        return [dict(row) for row in reader]

    raise JobLoadError("Only .csv and .json files are supported")
