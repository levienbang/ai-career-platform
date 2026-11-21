from app.cv.match import build_cv_query
from app.schemas.cv import CVExtraction, RecognizedSkill


def test_build_cv_query_order_and_limit():
    profile = CVExtraction(summary="Summary", projects=["Project one", "Project two"])
    recognized = {1: RecognizedSkill(skill="Python", evidence="Python")}
    assert build_cv_query(profile, recognized) == "Summary\nPython\nProject one\nProject two"
    assert len(build_cv_query(CVExtraction(summary="a" * 2100), {})) == 2000
