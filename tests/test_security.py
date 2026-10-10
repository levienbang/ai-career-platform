import pytest

from app.api.agent import get_career_agent
from app.config import Settings, get_settings
from app.graph.builder import AgentRunResult
from app.security import PaidCallLimiter, paid_call_limiter


class FakeAgent:
    def invoke(self, question: str) -> AgentRunResult:
        return AgentRunResult(question=question, route="sql", answer="ok", latency_ms=1)


@pytest.fixture
def configure(client):
    paid_call_limiter.reset()

    def apply(**values):
        settings = Settings(_env_file=None, **values)
        client.app.dependency_overrides[get_settings] = lambda: settings
        client.app.dependency_overrides[get_career_agent] = lambda: FakeAgent()
        return client

    yield apply
    paid_call_limiter.reset()


def test_import_is_open_when_no_admin_key_is_configured(configure):
    client = configure()
    # Past the guard: later checks (no file, no model key in tests) answer instead.
    assert client.post("/jobs/import").status_code != 401


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_import_and_index_require_the_admin_key(configure, headers):
    client = configure(admin_api_key="secret")
    assert client.post("/jobs/import", headers=headers).status_code == 401
    assert client.post("/search/index", json={}, headers=headers).status_code == 401


def test_correct_admin_key_passes_the_guard(configure):
    client = configure(admin_api_key="secret")
    response = client.post("/jobs/import", headers={"X-API-Key": "secret"})
    assert response.status_code != 401


def test_paid_routes_are_limited_per_minute(configure):
    client = configure(public_rate_limit_per_minute=2)
    body = {"question": "Có bao nhiêu job?"}
    assert client.post("/agent/query", json=body).status_code == 200
    assert client.post("/agent/query", json=body).status_code == 200
    limited = client.post("/agent/query", json=body)
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1
    # Free routes stay unlimited.
    assert client.post("/search/keyword", json={"query": "python"}).status_code == 200


def test_paid_routes_are_unlimited_by_default(configure):
    client = configure()
    for _ in range(5):
        response = client.post("/agent/query", json={"question": "Có bao nhiêu job?"})
        assert response.status_code == 200


def test_limiter_window_and_daily_cap():
    now = [0.0]
    day = ["2026-10-10"]
    limiter = PaidCallLimiter(clock=lambda: now[0], today=lambda: day[0])
    assert limiter.check("a", per_minute=1, per_day=3) is None
    assert limiter.check("a", per_minute=1, per_day=3) == 60
    assert limiter.check("b", per_minute=1, per_day=3) is None  # other client
    now[0] = 61
    assert limiter.check("a", per_minute=1, per_day=3) is None
    assert limiter.check("c", per_minute=1, per_day=3) == 3600  # shared daily cap
    day[0] = "2026-10-11"
    assert limiter.check("c", per_minute=1, per_day=3) is None


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
