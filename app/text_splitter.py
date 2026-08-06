"""Simple, dependency-free character chunker with overlap.

Kept intentionally transparent (no framework) so the retrieval behaviour is easy
to reason about and benchmark. Swappable later for a token-aware splitter.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

from .document_loader import LoadedText


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: Optional[int]


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        return [text]
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def split_documents(
    docs: list[LoadedText], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        for piece in _chunk_text(doc.text, chunk_size, chunk_overlap):
            # Content hash as ID => re-ingesting identical content is a no-op
            # (this is the duplicate-detection mechanism).
            digest = hashlib.sha256(
                f"{doc.source}|{doc.page}|{piece}".encode("utf-8")
            ).hexdigest()
            chunks.append(
                Chunk(id=digest, text=piece, source=doc.source, page=doc.page)
            )
    return chunks
