"""Stage 1: turn uploaded bytes into page text.

Machine-readable pages use the PDF text layer. Pages with (almost) no text
are treated as scanned images and sent through Tesseract OCR.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pymupdf

from src import config


class PdfReadError(Exception):
    """The file could not be read as a usable PDF (technical failure -> ERROR)."""


@dataclass
class PageText:
    page_number: int
    text: str
    method: str  # "text-layer" or "ocr"
    chars: int


@dataclass
class DocumentText:
    pages: list[PageText]
    page_count: int
    ocr_used: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(f"--- Page {p.page_number} ---\n{p.text}" for p in self.pages)

    @property
    def total_chars(self) -> int:
        return sum(p.chars for p in self.pages)


def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tessdata_dir() -> str | None:
    """Where the OCR language files are.

    TESSDATA_PREFIX (if set) wins; otherwise the eng.traineddata bundled in ./tessdata;
    otherwise None, letting PyMuPDF look for a system Tesseract installation.
    """
    env = config.get_setting("TESSDATA_PREFIX")
    if env:
        return env
    if (config.BUNDLED_TESSDATA_DIR / "eng.traineddata").exists():
        return str(config.BUNDLED_TESSDATA_DIR)
    return None


def _ocr_page(page: "pymupdf.Page") -> str:
    try:
        tp = page.get_textpage_ocr(language="eng", dpi=config.OCR_DPI, full=True, tessdata=tessdata_dir())
    except Exception as exc:  # Tesseract missing or failed
        raise PdfReadError(
            f"Page {page.number + 1} is a scanned image and OCR failed: {exc}. "
            "Check that tessdata/eng.traineddata exists in the project folder."
        ) from exc
    return page.get_text("text", textpage=tp, sort=True)


# A page whose images cover at least this share of its area, with little typed text, is a scan.
IMAGE_PAGE_COVERAGE = 0.5
MIXED_PAGE_MAX_TEXT = 1500


def _image_coverage(page: "pymupdf.Page") -> float:
    try:
        page_area = abs(page.rect)
        if not page_area:
            return 0.0
        covered = sum(abs(pymupdf.Rect(info["bbox"]) & page.rect) for info in page.get_image_info())
        return min(covered / page_area, 1.0)
    except Exception:
        return 0.0


def read_pdf(pdf_bytes: bytes) -> DocumentText:
    if not pdf_bytes:
        raise PdfReadError("The uploaded file is empty.")
    if len(pdf_bytes) > config.MAX_UPLOAD_BYTES:
        raise PdfReadError(f"File is larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if not pdf_bytes.lstrip()[:5].startswith(b"%PDF"):
        raise PdfReadError("File is not a PDF (missing %PDF header).")
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise PdfReadError(f"PDF is corrupted or unreadable: {exc}") from exc

    with document:
        if document.needs_pass:
            raise PdfReadError("PDF is password protected.")
        if document.page_count == 0:
            raise PdfReadError("PDF has no pages.")
        if document.page_count > config.MAX_PAGES:
            raise PdfReadError(f"PDF has {document.page_count} pages; the limit is {config.MAX_PAGES}.")

        pages: list[PageText] = []
        warnings: list[str] = []
        ocr_used = False
        for page in document:
            try:
                text = page.get_text("text", sort=True)
            except Exception as exc:
                raise PdfReadError(f"Could not read page {page.number + 1}: {exc}") from exc
            method = "text-layer"
            if len(text.strip()) < config.MIN_TEXT_CHARS_PER_PAGE:
                text = _ocr_page(page)
                method = "ocr"
                ocr_used = True
                warnings.append(f"Page {page.number + 1} had no text layer; read with OCR.")
            elif _image_coverage(page) >= IMAGE_PAGE_COVERAGE and len(text.strip()) < MIXED_PAGE_MAX_TEXT:
                # A scanned page with a small typed header/stamp: read the image too.
                text = text + "\n" + _ocr_page(page)
                method = "text+ocr"
                ocr_used = True
                warnings.append(f"Page {page.number + 1} is mostly a scanned image; read with OCR as well.")
            pages.append(PageText(page.number + 1, text, method, len(text.strip())))

    doc = DocumentText(pages=pages, page_count=len(pages), ocr_used=ocr_used, warnings=warnings)
    if doc.total_chars < config.MIN_TEXT_CHARS_PER_PAGE:
        raise PdfReadError("No readable text could be found in the PDF, even with OCR.")
    return doc
