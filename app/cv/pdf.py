import html
from io import BytesIO
from typing import Protocol


class CVPDFError(ValueError):
    pass


class PDFParserUnavailableError(RuntimeError):
    pass


class PDFTextLoader(Protocol):
    def load(self, content: bytes, *, max_pages: int) -> str: ...


def markdown_to_text(markdown: str) -> str:
    """Docling escapes HTML entities (``&`` becomes ``&amp;``); evidence checks need raw text."""
    return html.unescape(markdown).strip()


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
            raise CVPDFError("PDF has no extractable text")
        return text


def validate_cv_pdf(content: bytes, *, filename: str, content_type: str, max_bytes: int) -> None:
    if not filename.lower().endswith(".pdf") or content_type != "application/pdf":
        raise CVPDFError("Only PDF uploads are accepted")
    if not content or not content.startswith(b"%PDF-"):
        raise CVPDFError("PDF file is empty or invalid")
    if len(content) > max_bytes:
        raise CVPDFError("PDF exceeds the configured size limit")
