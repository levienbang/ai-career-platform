from io import BytesIO

import pytest

from app.api.cv import get_cv_tool
from app.cv.gap import TargetJobError, analyze_skill_gap
from app.db.models import Company, Job, JobSkill, RequirementType
from app.main import app
from app.schemas.cv import CVExtraction, CVSkillClaim
from app.tools.cv_tool import CVSkillGapTool


class FakePDFLoader:
    def load(self, content: bytes, *, max_pages: int) -> str:
        assert content.startswith(b"%PDF-")
        assert max_pages == 5
        return "Built services with Python and Postgres. Also used MysteryML."


class FakeCVExtractor:
    def extract(self, text: str) -> CVExtraction:
        assert "Python" in text
        return CVExtraction(
            summary="Built services",
            skills=[
                CVSkillClaim(name="python", evidence="Python"),
                CVSkillClaim(name="Postgres", evidence="Postgres"),
                CVSkillClaim(name="Docker", evidence="Docker"),
                CVSkillClaim(name="MysteryML", evidence="MysteryML"),
            ],
        )


def add_target_job(db_session, taxonomy, number, required, preferred):
    job = Job(
        title=f"AI Engineer {number}",
        company=Company(name=f"Company {number}"),
        description="Test job",
        content_hash=str(number) * 64,
    )
    for skill_name in required:
        job.skills.append(
            JobSkill(skill=taxonomy[skill_name], requirement_type=RequirementType.REQUIRED)
        )
    for skill_name in preferred:
        job.skills.append(
            JobSkill(skill=taxonomy[skill_name], requirement_type=RequirementType.PREFERRED)
        )
    db_session.add(job)
    db_session.commit()
    return job


def test_alias_evidence_unknown_and_weighted_gap(db_session, taxonomy):
    first = add_target_job(db_session, taxonomy, 1, ["Python", "Docker"], ["PostgreSQL"])
    second = add_target_job(db_session, taxonomy, 2, ["PostgreSQL"], ["FastAPI"])
    report = CVSkillGapTool(db_session, FakePDFLoader(), FakeCVExtractor()).invoke(
        b"%PDF-fixture", [first.id, second.id], max_pages=5
    )

    assert [item.skill for item in report.recognized_skills] == ["PostgreSQL", "Python"]
    assert report.unrecognized_skills == ["MysteryML"]
    assert report.jobs[0].matched_skills == ["PostgreSQL", "Python"]
    assert report.jobs[0].missing_required_skills == ["Docker"]
    assert report.jobs[0].fit_score == 0.6  # 0.8 * 1/2 + 0.2 * 1
    assert report.jobs[1].fit_score == 0.8
    assert [(item.skill, item.frequency) for item in report.missing_required_skills] == [
        ("Docker", 0.5)
    ]
    assert [(item.skill, item.frequency) for item in report.missing_preferred_skills] == [
        ("FastAPI", 0.5)
    ]


def test_missing_skills_are_ranked_by_frequency(db_session, taxonomy):
    jobs = [
        add_target_job(db_session, taxonomy, 1, ["FastAPI"], []),
        add_target_job(db_session, taxonomy, 2, ["FastAPI"], []),
        add_target_job(db_session, taxonomy, 3, ["Docker"], []),
    ]
    report = CVSkillGapTool(db_session, FakePDFLoader(), FakeCVExtractor()).invoke(
        b"%PDF-fixture", [job.id for job in jobs], max_pages=5
    )

    # FastAPI is missing more often, so it outranks the alphabetically earlier Docker.
    assert [item.skill for item in report.missing_required_skills] == ["FastAPI", "Docker"]


def test_missing_cv_data_and_no_job_requirements(db_session, taxonomy):
    job = add_target_job(db_session, taxonomy, 3, [], [])
    report = analyze_skill_gap(db_session, CVExtraction(), "No skills listed", [job.id])
    assert report.recognized_skills == []
    assert report.jobs[0].fit_score == 0
    assert report.jobs[0].required_coverage is None


def test_hallucinated_or_unquoted_skill_is_rejected(db_session, taxonomy):
    job = add_target_job(db_session, taxonomy, 4, ["Python"], [])
    profile = CVExtraction(
        skills=[
            CVSkillClaim(name="Python", evidence="Python"),
            CVSkillClaim(name="Docker", evidence="Python"),
        ]
    )
    report = analyze_skill_gap(db_session, profile, "Python", [job.id])
    assert [item.skill for item in report.recognized_skills] == ["Python"]
    assert report.jobs[0].fit_score == 1


def test_partial_word_is_not_skill_evidence(db_session, taxonomy):
    job = add_target_job(db_session, taxonomy, 6, ["Python"], [])
    profile = CVExtraction(skills=[CVSkillClaim(name="Python", evidence="Python")])
    report = analyze_skill_gap(db_session, profile, "Pythonista", [job.id])
    assert report.recognized_skills == []
    assert report.jobs[0].missing_required_skills == ["Python"]


def test_missing_job_is_error(db_session):
    with pytest.raises(TargetJobError, match="not found"):
        analyze_skill_gap(db_session, CVExtraction(), "", [999])


def test_cv_api_uses_tool_and_does_not_persist(client, db_session, taxonomy):
    job = add_target_job(db_session, taxonomy, 5, ["Python"], [])
    app.dependency_overrides[get_cv_tool] = lambda: CVSkillGapTool(
        db_session, FakePDFLoader(), FakeCVExtractor()
    )
    try:
        response = client.post(
            "/cv/upload",
            data={"job_ids": str(job.id)},
            files={"file": ("cv.pdf", BytesIO(b"%PDF-fixture"), "application/pdf")},
        )
        assert response.status_code == 200
        assert response.json()["jobs"][0]["fit_score"] == 1
        assert (
            client.post(
                "/cv/upload",
                data={"job_ids": "999"},
                files={"file": ("cv.pdf", b"%PDF-fixture", "application/pdf")},
            ).status_code
            == 404
        )
    finally:
        app.dependency_overrides.pop(get_cv_tool, None)
