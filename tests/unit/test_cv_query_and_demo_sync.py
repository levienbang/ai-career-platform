from app.cv.match import build_cv_query
from app.schemas.cv import CVExtraction, RecognizedSkill


def test_cv_query_order_limit_and_no_raw_contacts():
    profile = CVExtraction(
        summary="AI summary",
        projects=["Vision project"],
        experience=["Python internship"],
        education=["Computer Science"],
    )
    recognized = {1: RecognizedSkill(skill="Python", evidence="python@example.com 0901234567")}
    assert build_cv_query(profile, recognized) == (
        "AI summary\nPython\nVision project\nPython internship\nComputer Science"
    )
    query = build_cv_query(profile.model_copy(update={"summary": "x" * 2500}), recognized)
    assert len(query) == 2000
    assert "python@example.com" not in query and "0901234567" not in query
    assert build_cv_query(CVExtraction(experience=["Video processing"], education=["BSc"]), {}) == (
        "Video processing\nBSc"
    )
