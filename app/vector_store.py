"""Thin wrapper over a persistent ChromaDB collection.

We compute embeddings ourselves via the configured provider and hand the vectors
to Chroma directly (embedding_function=None), so the vector store stays decoupled
from which embedding model is in use.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .text_splitter import Chunk


class VectorStore:
    def __init__(self, persist_dir: Path, collection: str = "documents"):
        import chromadb  # lazy

        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def existing_ids(self, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        got = self._collection.get(ids=ids)
        return set(got.get("ids", []))

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {"source": c.source, "page": c.page if c.page is not None else -1}
                for c in chunks
            ],
        )

    def query(self, embedding: list[float], top_k: int) -> list[dict]:
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        out: list[dict] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            page: Optional[int] = meta.get("page")
            out.append(
                {
                    "snippet": doc,
                    "source": meta.get("source", "unknown"),
                    "page": None if page in (None, -1) else int(page),
                    "score": 1.0 - float(dist),  # cosine distance -> similarity
                }
            )
        return out

    def count(self) -> int:
        return self._collection.count()
