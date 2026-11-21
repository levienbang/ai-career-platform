from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.api import admin as admin_module
from app.config import Settings, get_settings
from app.db.models import Job, JobSkill, RequirementType, Skill, SkillAlias, SkillDecision
from app.ingestion.normalizer import skill_match_key
from app.llm import LLMCheckResult
from app.retrieval.reranker import reranker_fallbacks
from app.security import paid_call_limiter


@pytest.fixture(autouse=True)
def offline_qdrant(monkeypatch):
    # Never touch a developer's running Qdrant; the count is faked per test.
    class FakeQdrant:
        def count(self, collection, exact):
            return SimpleNamespace(count=7)

        def close(self):
            pass

    monkeypatch.setattr(admin_module, "build_qdrant_client", lambda *args: FakeQdrant())
    reranker_fallbacks.reset()
    paid_call_limiter.reset()
    yield
    reranker_fallbacks.reset()
    paid_call_limiter.reset()


def _with_admin_key(client):
    settings = Settings(_env_file=None, admin_api_key="admin-secret")
    client.app.dependency_overrides[get_settings] = lambda: settings
    return {"X-API-Key": "admin-secret"}


def _decision(session, name, decision="pending", *, skill=None, confidence=None):
    session.add(
        SkillDecision(
            match_key=skill_match_key(name),
            name=name,
            decision=decision,
            skill_id=skill.id if skill else None,
            confidence=Decimal(str(confidence)) if confidence is not None else None,
            reason="test",
        )
    )


def _job_with_skill(session, skill, evidence, number):
    job = Job(title=f"Job {number}", description=evidence, content_hash=f"{number:064x}")
    job.skills = [
        JobSkill(skill=skill, requirement_type=RequirementType.REQUIRED, evidence_text=evidence)
    ]
    session.add(job)
    return job


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/admin/stats"),
        ("get", "/admin/skill-decisions"),
        ("post", "/admin/skill-decisions/keep-new"),
        ("post", "/admin/check-llm"),
    ],
)
def test_admin_routes_require_the_admin_key(client, method, path) -> None:
    headers = _with_admin_key(client)

    assert getattr(client, method)(path).status_code == 401
    assert getattr(client, method)(path, headers={"X-API-Key": "wrong"}).status_code == 401
    # With the right key the request reaches the route (validation may still reject it).
    assert getattr(client, method)(path, headers=headers).status_code != 401


def test_stats_counts_data_paid_calls_and_fallbacks(client, db_session) -> None:
    headers = _with_admin_key(client)
    python = Skill(canonical_name="Python")
    db_session.add(python)
    db_session.flush()
    _job_with_skill(db_session, python, "Python", 1)
    _decision(db_session, "Py", "alias", skill=python)
    _decision(db_session, "Pyth0n")
    db_session.commit()
    paid_call_limiter.check_and_record("1.2.3.4", per_minute=0, per_day=100)
    reranker_fallbacks.record("cross_encoder")
    reranker_fallbacks.record("cross_encoder")

    body = client.get("/admin/stats", headers=headers).json()

    assert body["jobs"] == 1
    assert body["skills"] == 1
    assert body["skill_decisions"] == {"alias": 1, "pending": 1}
    assert body["indexed_jobs"] == 7
    assert body["paid_calls_today"] == 1
    assert body["reranker_fallbacks"] == {"cross_encoder": 2}


def test_stats_survive_qdrant_being_down(client, monkeypatch) -> None:
    headers = _with_admin_key(client)

    def down(*args):
        raise ConnectionError("qdrant down")

    monkeypatch.setattr(admin_module, "build_qdrant_client", down)

    response = client.get("/admin/stats", headers=headers)

    assert response.status_code == 200
    assert response.json()["indexed_jobs"] is None


def test_pending_decisions_are_paged_most_confident_first(client, db_session) -> None:
    headers = _with_admin_key(client)
    for name, confidence in [("Low", 0.5), ("High", 0.85), ("Unknown", None), ("Mid", 0.7)]:
        _decision(db_session, name, confidence=confidence)
    _decision(db_session, "Done", "new")
    db_session.commit()

    first = client.get("/admin/skill-decisions?limit=2", headers=headers).json()
    second = client.get("/admin/skill-decisions?limit=2&offset=2", headers=headers).json()

    assert first["total"] == 4
    assert [item["name"] for item in first["items"]] == ["High", "Mid"]
    assert [item["name"] for item in second["items"]] == ["Low", "Unknown"]


def test_review_buttons_approve_keep_and_revert(client, db_session) -> None:
    headers = _with_admin_key(client)
    canonical = Skill(canonical_name="Scikit-learn")
    pending = Skill(canonical_name="Scikit learn lib", origin="extracted")
    distinct = Skill(canonical_name="Separate tool", origin="extracted")
    db_session.add_all([canonical, pending, distinct])
    db_session.flush()
    job = _job_with_skill(db_session, pending, "Scikit learn lib", 1)
    _decision(db_session, "Scikit learn lib", skill=pending)
    _decision(db_session, "Separate tool", skill=distinct)
    db_session.commit()

    approved = client.post(
        "/admin/skill-decisions/approve",
        json={"name": "Scikit learn lib", "canonical": "Scikit-learn"},
        headers=headers,
    )
    kept = client.post(
        "/admin/skill-decisions/keep-new", json={"name": "Separate tool"}, headers=headers
    )

    assert approved.status_code == 200
    assert approved.json() == {
        "name": "Scikit learn lib",
        "decision": "alias",
        "affected_job_links": None,
        "reindex_needed": True,
    }
    assert kept.json()["decision"] == "new"
    db_session.expire_all()
    assert job.skills[0].skill_id == canonical.id

    # Approval by a reviewer stores the alias with source 'llm', so it can be undone.
    reverted = client.post(
        "/admin/skill-decisions/revert", json={"name": "Scikit learn lib"}, headers=headers
    )

    assert reverted.status_code == 200
    assert reverted.json()["affected_job_links"] == 1
    db_session.expire_all()
    assert job.skills[0].skill.canonical_name == "Scikit learn lib"
    assert db_session.query(SkillAlias).count() == 0


def test_review_errors_are_reported_and_rolled_back(client, db_session) -> None:
    headers = _with_admin_key(client)
    curated = Skill(canonical_name="Docker")
    db_session.add(curated)
    db_session.flush()
    db_session.add(SkillAlias(skill_id=curated.id, alias="docker engine", source="curated"))
    db_session.commit()

    missing = client.post(
        "/admin/skill-decisions/keep-new", json={"name": "Nothing"}, headers=headers
    )
    curated_alias = client.post(
        "/admin/skill-decisions/revert", json={"name": "docker engine"}, headers=headers
    )

    assert missing.status_code == 404
    assert curated_alias.status_code == 400
    assert "source='llm'" in curated_alias.json()["detail"]
    assert db_session.query(SkillAlias).count() == 1


def test_check_llm_reports_model_and_hides_provider_errors(client, monkeypatch) -> None:
    headers = _with_admin_key(client)
    monkeypatch.setattr(
        admin_module,
        "check_structured_output",
        lambda settings: ("deepseek-test", LLMCheckResult(ok=True, language="English")),
    )

    ok = client.post("/admin/check-llm", headers=headers)

    assert ok.status_code == 200
    assert ok.json()["ok"] is True
    assert ok.json()["model"] == "deepseek-test"

    def failing(settings):
        raise RuntimeError("401 Authorization: Bearer sk-leaked")

    monkeypatch.setattr(admin_module, "check_structured_output", failing)
    failed = client.post("/admin/check-llm", headers=headers)

    assert failed.status_code == 503
    assert failed.json() == {"detail": "LLM check failed (RuntimeError)"}
