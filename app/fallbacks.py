"""Fallback counters since the API process started, shown in the admin stats."""

import threading


class FallbackCounter:
    """Fallbacks keyed by what ran instead."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    def record(self, used: str) -> None:
        with self._lock:
            self._counts[used] = self._counts.get(used, 0) + 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()


reranker_fallbacks = FallbackCounter()
llm_fallbacks = FallbackCounter()
