"""Load supported documents into (text, metadata) records.

metadata carries {source, page} so answers can cite file + page number.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional


class LoadedText:
    __slots__ = ("text", "source", "page")

    def __init__(self, text: str, source: str, page: Optional[int]):
        self.text = text
        self.source = source
        self.page = page


def _load_pdf(path: Path) -> Iterator[LoadedText]:
    from pypdf import PdfReader  # lazy

    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            yield LoadedText(text=text, source=path.name, page=i)


def _load_plaintext(path: Path) -> Iterator[LoadedText]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if text:
        yield LoadedText(text=text, source=path.name, page=None)


def load_file(path: Path) -> list[LoadedText]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return list(_load_pdf(path))
    if suffix in (".txt", ".md"):
        return list(_load_plaintext(path))
    return []


def iter_documents(data_dir: Path, extensions: tuple[str, ...]) -> Iterator[Path]:
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path
