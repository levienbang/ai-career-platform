import sys
from io import BytesIO

import pytest

from app.cv.pdf import (
    CVPDFError,
    DoclingPDFTextLoader,
    EmptyPDFTextError,
    MacOCRPDFTextLoader,
    OCRFallbackPDFTextLoader,
    PDFParserUnavailableError,
    _reading_order,
    build_cv_pdf_loader,
    markdown_to_text,
)


class StubLoader:
    def __init__(self, result: str | Exception) -> None:
        self.result = result
        self.calls = 0

    def load(self, content: bytes, *, max_pages: int) -> str:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_docling_image_placeholder_counts_as_no_text() -> None:
    assert markdown_to_text("<!-- image -->") == ""
    assert markdown_to_text("# CV\n<!-- image -->\nPython") == "# CV\n\nPython"


def test_text_layer_is_used_without_ocr() -> None:
    ocr = StubLoader("ocr text")
    loader = OCRFallbackPDFTextLoader(StubLoader("native text"), ocr)

    assert loader.load(b"%PDF-", max_pages=5) == "native text"
    assert ocr.calls == 0


def test_empty_text_layer_falls_back_to_ocr() -> None:
    loader = OCRFallbackPDFTextLoader(
        StubLoader(EmptyPDFTextError("empty")), StubLoader("Kỹ năng: Python")
    )

    assert loader.load(b"%PDF-", max_pages=5) == "Kỹ năng: Python"


def test_invalid_pdf_never_reaches_ocr() -> None:
    ocr = StubLoader("ocr text")
    loader = OCRFallbackPDFTextLoader(StubLoader(CVPDFError("broken")), ocr)

    with pytest.raises(CVPDFError, match="broken"):
        loader.load(b"%PDF-", max_pages=5)
    assert ocr.calls == 0


def test_unavailable_ocr_keeps_the_no_text_error() -> None:
    loader = OCRFallbackPDFTextLoader(
        StubLoader(EmptyPDFTextError("empty")), StubLoader(PDFParserUnavailableError("no mac"))
    )

    with pytest.raises(EmptyPDFTextError, match="no extractable text"):
        loader.load(b"%PDF-", max_pages=5)


def test_disabled_ocr_keeps_the_no_text_error() -> None:
    loader = build_cv_pdf_loader(ocr_enabled=False)
    loader.primary = StubLoader(EmptyPDFTextError("PDF has no extractable text"))

    with pytest.raises(EmptyPDFTextError):
        loader.load(b"%PDF-", max_pages=5)


def test_ocr_lines_follow_reading_order() -> None:
    observations = [
        ("Kỹ năng: Python", 1.0, [0.05, 0.70, 0.5, 0.04]),
        ("NGUYỄN VĂN A", 1.0, [0.05, 0.86, 0.4, 0.05]),
        ("Docker", 1.0, [0.60, 0.70, 0.2, 0.04]),
        ("   ", 0.2, [0.05, 0.50, 0.1, 0.04]),
    ]

    assert _reading_order(observations) == ["NGUYỄN VĂN A", "Kỹ năng: Python", "Docker"]


def test_missing_ocr_extra_is_reported(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "ocrmac", None)
    monkeypatch.setitem(sys.modules, "ocrmac.ocrmac", None)

    with pytest.raises(PDFParserUnavailableError, match="--extra ocr"):
        MacOCRPDFTextLoader().load(b"%PDF-", max_pages=5)


def _scanned_pdf(lines: list[str]) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 34)
    image = Image.new("RGB", (1240, 600), "white")
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        draw.text((60, 80 + index * 70), line, fill="black", font=font)
    buffer = BytesIO()
    image.save(buffer, "PDF", resolution=150)
    return buffer.getvalue()


@pytest.mark.skipif(sys.platform != "darwin", reason="Apple Vision OCR is macOS only")
def test_scanned_vietnamese_cv_is_read_by_ocr() -> None:
    pytest.importorskip("ocrmac")
    content = _scanned_pdf(["NGUYỄN VĂN A - Kỹ sư học máy", "Kỹ năng: Python, Docker"])

    with pytest.raises(EmptyPDFTextError):
        DoclingPDFTextLoader().load(content, max_pages=5)
    text = build_cv_pdf_loader(ocr_enabled=True).load(content, max_pages=5)

    assert text.splitlines() == ["NGUYỄN VĂN A - Kỹ sư học máy", "Kỹ năng: Python, Docker"]


@pytest.mark.skipif(sys.platform != "darwin", reason="Apple Vision OCR is macOS only")
def test_ocr_enforces_the_page_limit() -> None:
    pytest.importorskip("ocrmac")

    with pytest.raises(CVPDFError, match="page limit"):
        MacOCRPDFTextLoader().load(_scanned_pdf(["Python"]), max_pages=0)
