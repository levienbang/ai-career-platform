from types import SimpleNamespace

from sqlalchemy import func, select

from app.db.models import Company, Job, Skill
from app.ingestion.extractor import LangChainJobExtractor
from app.ingestion.pipeline import JobIngestionPipeline, build_content_hash
from app.schemas.ingestion import JobExtraction, RawJobRecord


def test_optional_company_schema_and_extractor_guard():
    assert JobExtraction(title="AI", description="Python", company="").company is None
    assert JobExtraction(title="AI", description="Python", company=None).company is None
    extractor = LangChainJobExtractor.__new__(LangChainJobExtractor)
    extractor._prompt = SimpleNamespace(invoke=lambda _: None)
    extractor._model = SimpleNamespace(
        invoke=lambda _: JobExtraction(title="AI", company="ACME", description="Python")
    )
    extractor._max_retries = 0
    assert extractor.extract(RawJobRecord(description="Python")).company is None
    assert (
        extractor.extract(RawJobRecord(description="Python", company="Real Company")).company
        == "Real Company"
    )


def test_content_hash_keeps_company_hash_and_handles_null():
    with_company = JobExtraction(
        title="AI Engineer", company="Acme", description="Build Python services."
    )
    assert (
        build_content_hash(with_company)
        == "3e92df96802c4e6090fd245b76faafaae789420f199122f89d87106951a6250c"
    )
    without_company = with_company.model_copy(update={"company": None})
    assert (
        build_content_hash(without_company)
        == "886e699039c24e2d0f15dbb2ff7e45d736c00d60cffbd26b88416c3ed2d6ec17"
    )
    assert build_content_hash(without_company) != build_content_hash(with_company)


def test_vietjobs_batch_optional_company_and_open_taxonomy(db_session, taxonomy, fake_extractor):
    records = [
        {
            "title": f"Kỹ sư AI {number}",
            "company": None,
            "description": f"Description: Vị trí {number}. Requirements: Node.js và Python. "
            "Technical skills: Node.js.",
            "required_skills": ["Node.js", "Python", "Invented Skill"],
            "source_url": f"https://example.com/jobs/{number}",
        }
        for number in range(3)
    ]
    records[1]["required_skills"] = ["node.js", "Python"]
    pipeline = JobIngestionPipeline(db_session, fake_extractor)
    first = pipeline.import_records(records)
    second = pipeline.import_records(records)
    assert (first.inserted, first.failed, second.skipped_duplicates) == (3, 0, 3)
    assert db_session.scalar(select(func.count()).select_from(Company)) == 0
    assert all(job.company_id is None for job in db_session.scalars(select(Job)))
    extracted = list(db_session.scalars(select(Skill).where(Skill.origin == "extracted")))
    assert [(skill.canonical_name, skill.origin) for skill in extracted] == [
        ("Node.js", "extracted")
    ]
    assert db_session.scalar(select(Skill).where(Skill.canonical_name == "Invented Skill")) is None


def test_search_document_and_job_response_keep_null_company(db_session, client, fake_reranker):
    from app.retrieval.dense import DenseSearchService
    from app.retrieval.documents import build_search_document
    from app.retrieval.keyword import KeywordSearchService
    from app.retrieval.reranker import RerankedSearchService
    from app.services.jobs import get_job
    from app.tools.search_tool import RerankedJobSearchTool

    job = Job(title="AI Engineer", description="Python work", content_hash="a" * 64)
    db_session.add(job)
    db_session.commit()
    document = build_search_document(job)
    assert "Minimum experience" not in document.text
    assert document.payload["company"] is None
    hit = KeywordSearchService(db_session).search("Python")[0]
    assert hit.company is None
    assert DenseSearchService._hit_from_payload(document.payload, 0.8).company is None
    backend = SimpleNamespace(search=lambda query, *, limit: [hit])
    assert RerankedJobSearchTool(backend).invoke("Python").jobs[0].company is None
    assert (
        RerankedSearchService(backend, fake_reranker, candidate_limit=1)
        .search("Python", limit=1)[0]
        .company
        is None
    )
    assert get_job(db_session, job.id).company is None
    assert client.get(f"/jobs/{job.id}").json()["company"] is None
