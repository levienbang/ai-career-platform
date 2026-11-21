import pytest

from app.config import Settings
from app.graph.builder import AgentRunResult
from app.security import PaidCallLimiter


class FakeAgent:
    def invoke(self, question: str) -> AgentRunResult:
        return AgentRunResult(question=question, route="sql", answer="ok", latency_ms=1)


def test_limiter_window_and_daily_cap():
    now = [0.0]
    day = ["2026-10-10"]
    limiter = PaidCallLimiter(clock=lambda: now[0], today=lambda: day[0])
    assert limiter.check_and_record("a", per_minute=1, per_day=3) is None
    assert limiter.check_and_record("a", per_minute=1, per_day=3) == 60
    assert limiter.check_and_record("b", per_minute=1, per_day=3) is None  # other client
    now[0] = 61
    assert limiter.check_and_record("a", per_minute=1, per_day=3) is None
    assert limiter.check_and_record("c", per_minute=1, per_day=3) == 3600  # shared daily cap
    day[0] = "2026-10-11"
    assert limiter.check_and_record("c", per_minute=1, per_day=3) is None


def test_production_refuses_to_start_without_guards():
    with pytest.raises(ValueError, match="ADMIN_API_KEY.*PUBLIC_DAILY_LIMIT"):
        Settings(_env_file=None, app_env="production")
    settings = Settings(
        _env_file=None,
        app_env="production",
        admin_api_key="secret",
        public_rate_limit_per_minute=5,
        public_daily_limit=200,
    )
    assert settings.app_env == "production"
