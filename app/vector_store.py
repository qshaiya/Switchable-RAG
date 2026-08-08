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

    def all_chunks(self) -> list[tuple[str, str]]:
        """Return every (id, text) pair in the collection.

        Used by evaluate.py to build a content-agnostic retrieval self-test:
        it works on whatever is indexed, so it stays meaningful after the user
        swaps in their own documents.
        """
        got = self._collection.get(include=["documents"])
        return list(zip(got.get("ids", []), got.get("documents", [])))


# --------------------------------------------------------------------------- #
# Qdrant (managed/hosted vector store) — switchable alternative to local Chroma.
# Same interface as VectorStore above, so the rest of the app is unchanged.
# --------------------------------------------------------------------------- #
def _to_point_id(chunk_id: str) -> str:
    """Qdrant point ids must be int or UUID; derive a stable UUID from our
    sha256 chunk id so re-ingesting identical content stays a no-op."""
    import uuid

    return str(uuid.UUID(hex=chunk_id[:32]))


class QdrantVectorStore:
    def __init__(self, url: str, api_key: Optional[str], collection: str = "documents"):
        from qdrant_client import QdrantClient  # lazy

        self._client = QdrantClient(url=url, api_key=api_key)
        self._name = collection

    def _exists(self) -> bool:
        return self._client.collection_exists(self._name)

    def _ensure_collection(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams  # lazy

        if not self._exists():
            self._client.create_collection(
                collection_name=self._name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def existing_ids(self, ids: list[str]) -> set[str]:
        if not ids or not self._exists():
            return set()
        pid_to_orig = {_to_point_id(i): i for i in ids}
        found = self._client.retrieve(
            collection_name=self._name, ids=list(pid_to_orig.keys())
        )
        return {pid_to_orig[str(p.id)] for p in found if str(p.id) in pid_to_orig}

    def add(self, chunks: list["Chunk"], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        from qdrant_client.models import PointStruct  # lazy

        self._ensure_collection(len(embeddings[0]))
        points = [
            PointStruct(
                id=_to_point_id(c.id),
                vector=emb,
                payload={
                    "source": c.source,
                    "page": c.page if c.page is not None else -1,
                    "text": c.text,
                },
            )
            for c, emb in zip(chunks, embeddings)
        ]
        self._client.upsert(collection_name=self._name, points=points)

    def query(self, embedding: list[float], top_k: int) -> list[dict]:
        if not self._exists():
            return []
        res = self._client.query_points(
            collection_name=self._name, query=embedding,
            limit=top_k, with_payload=True,
        ).points
        out: list[dict] = []
        for p in res:
            payload = p.payload or {}
            page = payload.get("page")
            out.append({
                "snippet": payload.get("text", ""),
                "source": payload.get("source", "unknown"),
                "page": None if page in (None, -1) else int(page),
                "score": float(p.score),  # cosine similarity, higher = closer
            })
        return out

    def count(self) -> int:
        if not self._exists():
            return 0
        return self._client.count(collection_name=self._name).count

    def all_chunks(self) -> list[tuple[str, str]]:
        if not self._exists():
            return []
        out: list[tuple[str, str]] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._name, limit=256,
                with_payload=True, offset=offset,
            )
            out.extend((str(p.id), (p.payload or {}).get("text", "")) for p in points)
            if offset is None:
                break
        return out


def get_vector_store(cfg):
    """Return the configured vector store: local Chroma (default) or Qdrant."""
    if getattr(cfg, "vector_store", "chroma") == "qdrant":
        if not cfg.qdrant_url:
            raise RuntimeError(
                "VECTOR_STORE=qdrant but QDRANT_URL is not set. "
                "Add QDRANT_URL and QDRANT_API_KEY to .env."
            )
        return QdrantVectorStore(cfg.qdrant_url, cfg.qdrant_api_key)
    return VectorStore(cfg.vector_store_dir)
