"""
PDF text extraction with page-level metadata preservation. Uses PyMuPDF
(fitz) for speed and accurate page/text-block extraction.
"""
import logging
from typing import List, Dict, Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFParsingError(Exception):
    """Raised when a PDF cannot be opened or parsed."""


class PDFParser:
    def extract_text_with_metadata(self, pdf_path: str, doc_id: str) -> Dict[str, Any]:
        """
        Extracts text page-by-page from a PDF, preserving exact page-number
        associations, and performs light cleaning (whitespace normalization).

        Returns:
            {
                "total_pages": int,
                "pages": [{"doc_id", "page_number", "text"}, ...]
            }
        """
        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            logger.exception("Failed to open PDF: %s", pdf_path)
            raise PDFParsingError(f"Could not open PDF '{pdf_path}': {exc}") from exc

        pages: List[Dict[str, Any]] = []
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                raw_text = page.get_text("text")
                cleaned = self._clean_text(raw_text)
                if cleaned:
                    pages.append({
                        "doc_id": doc_id,
                        "page_number": page_num + 1,  # 1-indexed, human-readable
                        "text": cleaned,
                    })
            total_pages = len(doc)
        finally:
            doc.close()

        if not pages:
            logger.warning("No extractable text found in %s (may be a scanned/image PDF).", pdf_path)

        return {"total_pages": total_pages, "pages": pages}

    @staticmethod
    def _clean_text(text: str) -> str:
        """Basic data-cleaning: collapse repeated whitespace, strip control chars."""
        if not text:
            return ""
        lines = [ln.strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln]
        cleaned = "\n".join(lines)
        while "  " in cleaned:
            cleaned = cleaned.replace("  ", " ")
        return cleaned.strip()
