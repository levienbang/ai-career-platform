import html
import re
from collections.abc import Sequence
from io import BytesIO
from typing import Protocol

# Docling renders a picture with no text layer (a scanned page) as this comment.
_MARKDOWN_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


class CVPDFError(ValueError):
    pass


class EmptyPDFTextError(CVPDFError):
    """The PDF parsed fine but has no text layer, e.g. a scanned CV."""


class PDFParserUnavailableError(RuntimeError):
    pass


class PDFTextLoader(Protocol):
    def load(self, content: bytes, *, max_pages: int) -> str: ...


def markdown_to_text(markdown: str) -> str:
    """Docling escapes HTML entities (``&`` becomes ``&amp;``); evidence checks need raw text."""
    return html.unescape(_MARKDOWN_COMMENT.sub("", markdown)).strip()


class DoclingPDFTextLoader:
    def load(self, content: bytes, *, max_pages: int) -> str:
        try:
            from docling.datamodel.base_models import DocumentStream, InputFormat
            from docling.document_converter import DocumentConverter, NativePdfFormatOption
        except ImportError as error:
            raise PDFParserUnavailableError("Docling PDF parser is unavailable") from error

        try:
            converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF],
                format_options={InputFormat.PDF: NativePdfFormatOption()},
            )
            result = converter.convert(
                DocumentStream(name="cv.pdf", stream=BytesIO(content)),
                max_num_pages=max_pages,
                max_file_size=len(content),
            )
            text = markdown_to_text(result.document.export_to_markdown())
        except Exception as error:
            raise CVPDFError("PDF is invalid, unreadable, or exceeds the page limit") from error
        if not text:
            raise EmptyPDFTextError("PDF has no extractable text")
        return text


def _reading_order(observations: Sequence[tuple[str, float, Sequence[float]]]) -> list[str]:
    """Vision boxes are normalized (x, y, w, h) with y from the bottom: top lines first."""
    ordered = sorted(observations, key=lambda item: (-round(item[2][1], 2), item[2][0]))
    return [text.strip() for text, _confidence, _box in ordered if text.strip()]


class MacOCRPDFTextLoader:
    """OCR each rendered page with Apple Vision (ocrmac); macOS only, no model download."""

    def __init__(
        self, *, languages: Sequence[str] = ("vi-VT", "en-US"), render_scale: float = 2.0
    ) -> None:
        self.languages = list(languages)
        self.render_scale = render_scale

    def load(self, content: bytes, *, max_pages: int) -> str:
        try:
            import pypdfium2
            from ocrmac.ocrmac import OCR
        except ImportError as error:
            raise PDFParserUnavailableError(
                "Scanned-PDF OCR needs macOS and the optional extra: uv sync --extra ocr"
            ) from error

        try:
            document = pypdfium2.PdfDocument(content)
        except Exception as error:
            raise CVPDFError("PDF is invalid or unreadable") from error
        try:
            if len(document) > max_pages:
                raise CVPDFError("PDF exceeds the page limit")
            pages = []
            for page in document:
                image = page.render(scale=self.render_scale).to_pil()
                observations = OCR(
                    image, recognition_level="accurate", language_preference=self.languages
                ).recognize()
                pages.append("\n".join(_reading_order(observations)))
        except CVPDFError:
            raise
        except Exception as error:
            raise CVPDFError("OCR failed on the PDF") from error
        finally:
            document.close()
        text = "\n\n".join(page for page in pages if page).strip()
        if not text:
            raise EmptyPDFTextError("PDF has no extractable text, even after OCR")
        return text


class OCRFallbackPDFTextLoader:
    """Read the text layer first; OCR only when it is empty, so normal CVs pay nothing."""

    def __init__(self, primary: PDFTextLoader, ocr: PDFTextLoader | None) -> None:
        self.primary = primary
        self.ocr = ocr

    def load(self, content: bytes, *, max_pages: int) -> str:
        try:
            return self.primary.load(content, max_pages=max_pages)
        except EmptyPDFTextError:
            if self.ocr is None:
                raise
            try:
                return self.ocr.load(content, max_pages=max_pages)
            except PDFParserUnavailableError as error:
                # Off macOS or without the extra: keep the original, clearer error.
                raise EmptyPDFTextError("PDF has no extractable text") from error


def build_cv_pdf_loader(*, ocr_enabled: bool) -> PDFTextLoader:
    return OCRFallbackPDFTextLoader(
        DoclingPDFTextLoader(), MacOCRPDFTextLoader() if ocr_enabled else None
    )


def validate_cv_pdf(content: bytes, *, filename: str, content_type: str, max_bytes: int) -> None:
    if not filename.lower().endswith(".pdf") or content_type != "application/pdf":
        raise CVPDFError("Only PDF uploads are accepted")
    if not content or not content.startswith(b"%PDF-"):
        raise CVPDFError("PDF file is empty or invalid")
    if len(content) > max_bytes:
        raise CVPDFError("PDF exceeds the configured size limit")
