from app.db.models import Company, Job, JobSkill, RequirementType, Skill


def add_job(db_session) -> Job:
    company = Company(name="Test Company")
    python = Skill(canonical_name="Python", category="programming")
    job = Job(
        title="AI Engineer Intern",
        company=company,
        location="Ho Chi Minh City",
        employment_type="Internship",
        description="Build AI services with Python.",
        experience_years_min=0,
        content_hash="a" * 64,
    )
    job.skills.append(
        JobSkill(
            skill=python,
            requirement_type=RequirementType.REQUIRED,
            evidence_text="Python",
        )
    )
    db_session.add(job)
    db_session.commit()
    return job


def test_list_jobs_returns_database_records(client, db_session) -> None:
    job = add_job(db_session)

    response = client.get("/jobs")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": job.id,
            "title": "AI Engineer Intern",
            "company": {"id": job.company_id, "name": "Test Company"},
            "location": "Ho Chi Minh City",
            "employment_type": "Internship",
            "description": "Build AI services with Python.",
            "experience_years_min": 0,
            "source_url": None,
            "posted_at": None,
            "ingested_at": job.ingested_at.isoformat().replace("+00:00", "Z"),
            "skills": [
                {
                    "skill": "Python",
                    "requirement_type": "required",
                    "importance": None,
                    "evidence_text": "Python",
                }
            ],
        }
    ]


def test_get_job_returns_404_for_unknown_id(client) -> None:
    response = client.get("/jobs/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Job not found"}


def test_list_jobs_validates_pagination(client) -> None:
    response = client.get("/jobs?limit=101")

    assert response.status_code == 422
