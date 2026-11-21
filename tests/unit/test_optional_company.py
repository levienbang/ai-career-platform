from types import SimpleNamespace

from app.ingestion.extractor import LangChainJobExtractor
from app.ingestion.pipeline import build_content_hash
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
