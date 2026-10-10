"""Guards for a public deployment: an admin key for write routes and a limit on
routes that spend paid LLM or embedding calls.

Both are off unless configured, so local development keeps working unchanged.
"""

import hmac
import math
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from threading import Lock
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from app.config import Settings, get_settings


def require_admin_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """Protect routes that write data or spend quota in bulk (import, index)."""
    expected = settings.admin_api_key
    if not expected:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key",
        )


class PaidCallLimiter:
    """In-memory sliding minute window per client plus one shared daily cap.

    One API process only; several workers or servers would need a shared store.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        today: Callable[[], str] = lambda: datetime.now(UTC).date().isoformat(),
    ) -> None:
        self._clock = clock
        self._today = today
        self._lock = Lock()
        self._recent: dict[str, deque[float]] = {}
        self._day = ""
        self._day_count = 0

    def check(self, client: str, *, per_minute: int, per_day: int) -> int | None:
        """Record one call; return seconds to wait when a limit is exceeded."""
        with self._lock:
            now = self._clock()
            day = self._today()
            if day != self._day:
                self._day, self._day_count = day, 0
            if per_day and self._day_count >= per_day:
                return 3600
            if per_minute:
                calls = self._recent.setdefault(client, deque())
                while calls and now - calls[0] >= 60:
                    calls.popleft()
                if len(calls) >= per_minute:
                    return max(1, math.ceil(60 - (now - calls[0])))
                calls.append(now)
            self._day_count += 1
            return None

    def reset(self) -> None:
        with self._lock:
            self._recent.clear()
            self._day, self._day_count = "", 0


paid_call_limiter = PaidCallLimiter()


def limit_paid_calls(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """Throttle routes that call DeepSeek or the embedding API on each request."""
    per_minute = settings.public_rate_limit_per_minute
    per_day = settings.public_daily_limit
    if not per_minute and not per_day:
        return
    client = request.client.host if request.client else "unknown"
    wait = paid_call_limiter.check(client, per_minute=per_minute, per_day=per_day)
    if wait is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Demo request limit reached; try again later",
            headers={"Retry-After": str(wait)},
        )


AdminKey = Depends(require_admin_key)
PaidCallLimit = Depends(limit_paid_calls)
