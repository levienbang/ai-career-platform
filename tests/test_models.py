import pytest
from sqlalchemy.exc import IntegrityError

from app.db.models import Company, Job


def test_job_requires_non_negative_experience(db_session) -> None:
    job = Job(
        title="Invalid Job",
        company=Company(name="Example"),
        description="Invalid experience requirement.",
        experience_years_min=-1,
        content_hash="b" * 64,
    )
    db_session.add(job)

    with pytest.raises(IntegrityError):
        db_session.commit()
