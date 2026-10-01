from io import BytesIO

from app.api.cv import get_cv_match_service
from app.config import Settings
from app.db.models import Job, JobSkill, RequirementType
from app.main import app
from app.retrieval.qdrant import SearchIndexNotReadyError
from app.retrieval.types import SearchHit
from app.schemas.cv import CVExtraction, CVSkillClaim, RecognizedSkill
from app.services.cv_match import CVMatchService, build_cv_query
from app.services.skill_gap import recognize_cv_skills


class FakeLoader:
    def load(self, content, *, max_pages):
        assert content.startswith(b"%PDF-") and max_pages == 5
        return "Python and Node.js projects"


class FakeExtractor:
    def extract(self, text):
        return CVExtraction(
            summary="AI developer",
            projects=["Built a vision application"],
            skills=[
                CVSkillClaim(name="Python", evidence="Python"),
                CVSkillClaim(name="Jetson Nano", evidence="Jetson Nano"),
            ],
        )


class FakeDense:
    def __init__(self, scores):
        self.scores = scores

    def search(self, query, *, limit):
        assert query.startswith("AI developer\nPython\n") and limit == 50
        return [
            SearchHit(
                job_id=job_id,
                score=score,
                title="AI",
                company=None,
                location=None,
                employment_type=None,
                experience_years_min=7,
                required_skills=[],
                preferred_skills=[],
            )
            for job_id, score in self.scores
        ]


def test_build_cv_query_order_and_limit():
    profile = CVExtraction(summary="Summary", projects=["Project one", "Project two"])
    recognized = {1: RecognizedSkill(skill="Python", evidence="Python")}
    assert build_cv_query(profile, recognized) == "Summary\nPython\nProject one\nProject two"
    assert len(build_cv_query(CVExtraction(summary="a" * 2100), {})) == 2000


def test_recognize_shared_and_match_scoring(db_session, taxonomy):
    jobs = [
        Job(title=f"AI {i}", description="Python", content_hash=str(i) * 64) for i in range(1, 4)
    ]
    jobs[0].skills.append(
        JobSkill(skill=taxonomy["Python"], requirement_type=RequirementType.REQUIRED)
    )
    jobs[1].skills.append(
        JobSkill(skill=taxonomy["Docker"], requirement_type=RequirementType.REQUIRED)
    )
    db_session.add_all(jobs)
    db_session.commit()
    profile = FakeExtractor().extract("")
    recognized, unknown = recognize_cv_skills(db_session, profile, "Python and Jetson Nano")
    assert [item.skill for item in recognized.values()] == ["Python"]
    assert unknown == {"Jetson Nano"}
    no_market_profile = CVExtraction(skills=[CVSkillClaim(name="FastAPI", evidence="FastAPI")])
    _, no_market_unknown = recognize_cv_skills(db_session, no_market_profile, "FastAPI")
    assert no_market_unknown == {"FastAPI"}
    dense = FakeDense([(jobs[2].id, 0.5), (jobs[1].id, 0.5), (jobs[0].id, 0.4)])
    report = CVMatchService(db_session, FakeLoader(), FakeExtractor(), dense, Settings()).invoke(
        b"%PDF-test", limit=3, max_pages=5
    )
    assert [item.job_id for item in report.jobs] == [jobs[0].id, jobs[1].id, jobs[2].id]
    assert report.jobs[0].final_score == 0.64
    assert report.jobs[1].skill_score == 0
    assert report.jobs[2].skill_score == 0
    assert report.jobs[0].company is None
    assert report.jobs[0].model_dump().get("experience_years_min") is None


def test_cv_match_api_success_validation_and_index_error(client, db_session, taxonomy):
    job = Job(title="AI", description="Python", content_hash="z" * 64)
    job.skills.append(JobSkill(skill=taxonomy["Python"], requirement_type=RequirementType.REQUIRED))
    db_session.add(job)
    db_session.commit()
    service = CVMatchService(
        db_session, FakeLoader(), FakeExtractor(), FakeDense([(job.id, 0.8)]), Settings()
    )
    app.dependency_overrides[get_cv_match_service] = lambda: service
    try:
        payload = {"file": ("cv.pdf", BytesIO(b"%PDF-test"), "application/pdf")}
        response = client.post("/cv/match", files=payload)
        assert response.status_code == 200
        assert response.json()["jobs"][0]["company"] is None
        assert client.post("/cv/match", files=payload, data={"limit": "21"}).status_code == 422
        assert (
            client.post("/cv/match", files={"file": ("cv.txt", b"bad", "text/plain")}).status_code
            == 400
        )
        service.dense_search = type(
            "Unavailable",
            (),
            {
                "search": lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    SearchIndexNotReadyError("index jobs first")
                )
            },
        )()
        failed = client.post("/cv/match", files=payload)
        assert failed.status_code == 503
        assert "index jobs first" in failed.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_cv_match_service, None)
