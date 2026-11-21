import sys
from types import ModuleType, SimpleNamespace

import pytest

from app.cv.pdf import CVPDFError, DoclingPDFTextLoader, validate_cv_pdf


@pytest.mark.parametrize(
    ("content", "filename", "mime"),
    [(b"", "cv.pdf", "application/pdf"), (b"%PDF-x", "cv.txt", "text/plain")],
)
def test_reject_invalid_pdf(content, filename, mime):
    with pytest.raises(CVPDFError):
        validate_cv_pdf(content, filename=filename, content_type=mime, max_bytes=100)


def test_reject_oversized_pdf():
    with pytest.raises(CVPDFError, match="size"):
        validate_cv_pdf(
            b"%PDF-" + b"x" * 101,
            filename="cv.pdf",
            content_type="application/pdf",
            max_bytes=100,
        )


def test_docling_loader_enforces_page_limit_and_rejects_no_text(monkeypatch):
    calls = []

    class FakeDocumentStream:
        def __init__(self, *, name, stream):
            assert name == "cv.pdf"
            assert stream.read() == b"%PDF-test"

    class FakeConverter:
        def __init__(self, *, allowed_formats, format_options):
            assert allowed_formats == ["pdf"]
            assert isinstance(format_options["pdf"], FakeNativePdfFormatOption)

        def convert(self, source, *, max_num_pages, max_file_size):
            assert isinstance(source, FakeDocumentStream)
            calls.append((max_num_pages, max_file_size))
            return SimpleNamespace(document=SimpleNamespace(export_to_markdown=lambda: "  "))

    docling = ModuleType("docling")
    datamodel = ModuleType("docling.datamodel")
    base_models = ModuleType("docling.datamodel.base_models")
    base_models.DocumentStream = FakeDocumentStream
    base_models.InputFormat = SimpleNamespace(PDF="pdf")

    class FakeNativePdfFormatOption:
        pass

    converter = ModuleType("docling.document_converter")
    converter.DocumentConverter = FakeConverter
    converter.NativePdfFormatOption = FakeNativePdfFormatOption
    for name, module in {
        "docling": docling,
        "docling.datamodel": datamodel,
        "docling.datamodel.base_models": base_models,
        "docling.document_converter": converter,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    with pytest.raises(CVPDFError, match="no extractable text"):
        DoclingPDFTextLoader().load(b"%PDF-test", max_pages=5)
    assert calls == [(5, 9)]
