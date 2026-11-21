"""Async client for the AI Career Platform API. No AI logic lives in the chat app."""

import os
from typing import Any

import httpx

API_URL = os.getenv("CAREER_API_URL", "http://localhost:8000").rstrip("/")
# Sent only on admin calls, and only for a user who logged in as admin.
ADMIN_API_KEY = os.getenv("CAREER_ADMIN_API_KEY", "")
TIMEOUT = httpx.Timeout(300.0, connect=5.0)


def _error_text(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = response.text[:300]
    return f"HTTP {response.status_code}: {detail}"


async def call(
    method: str, path: str, *, admin: bool = False, **kwargs: Any
) -> tuple[Any | None, str | None]:
    """Return (json, None) on success or (None, readable error)."""
    headers = {"X-API-Key": ADMIN_API_KEY} if admin and ADMIN_API_KEY else {}
    try:
        async with httpx.AsyncClient(base_url=API_URL, timeout=TIMEOUT) as client:
            response = await client.request(method, path, headers=headers, **kwargs)
    except httpx.TimeoutException:
        return None, f"Hết thời gian chờ API ({path})."
    except httpx.HTTPError as error:
        return None, f"Không kết nối được API tại {API_URL}: {type(error).__name__}"
    if response.is_error:
        return None, _error_text(response)
    return response.json(), None


async def ask_agent(question: str) -> tuple[Any | None, str | None]:
    return await call("POST", "/agent/query", json={"question": question})


async def match_cv(name: str, content: bytes, limit: int = 10):
    return await call(
        "POST",
        "/cv/match",
        files={"file": (name, content, "application/pdf")},
        data={"limit": str(limit)},
    )


async def skill_gap(name: str, content: bytes, job_ids: list[int]):
    return await call(
        "POST",
        "/cv/upload",
        files={"file": (name, content, "application/pdf")},
        data={"job_ids": [str(job_id) for job_id in job_ids]},
    )
