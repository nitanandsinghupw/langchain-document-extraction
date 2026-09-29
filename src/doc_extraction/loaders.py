"""Turns files into text. Plain text and Markdown are read directly; PDFs use pypdf's text layer."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

SUPPORTED = {".txt", ".md", ".pdf"}
MAX_CHARS = 100_000


class UnsupportedDocumentError(ValueError):
    pass


def text_from_bytes(data: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise UnsupportedDocumentError(f"'{suffix}' files are not supported (use {', '.join(sorted(SUPPORTED))}).")
    if suffix == ".pdf":
        reader = PdfReader(BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            raise UnsupportedDocumentError("The PDF has no text layer. Scanned PDFs need OCR first.")
    else:
        text = data.decode("utf-8", errors="replace")
    if len(text) > MAX_CHARS:
        raise UnsupportedDocumentError(f"The document is longer than {MAX_CHARS:,} characters.")
    return text


def load(path: str | Path) -> str:
    path = Path(path)
    return text_from_bytes(path.read_bytes(), path.name)
