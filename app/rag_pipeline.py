"""Core RAG logic, framework-free and importable.

Both the FastAPI service (app/api.py) and the benchmark script call into these
two functions, so the retrieval/generation behaviour is defined in exactly one
place. Timing is returned with every answer to feed latency benchmarking.
"""
from __future__ import annotations

import time
from typing import Optional

from .config import Config
from .document_loader import iter_documents, load_file
from .providers import get_chat_provider, get_embedding_provider
from .text_splitter import split_documents
from .vector_store import get_vector_store

# A deliberately grounding prompt: the model must answer from context or admit
# it doesn't know, which curbs hallucination.
SYSTEM_PROMPT = (
    "You are a retrieval assistant. Answer the user's question using ONLY the "
    "context passages provided. If the answer is not in the context, say you "
    "don't have enough information. Cite the source filenames you used. Keep the "
    "answer concise and in the same language as the question."
)


def _store(cfg: Config):
    return get_vector_store(cfg)


def ingest(cfg: Config) -> dict:
    """Index every supported file under cfg.data_dir. Idempotent."""
    start = time.perf_counter()
    embedder = get_embedding_provider(cfg)
    store = _store(cfg)

    ingested: list[str] = []
    added = skipped = 0

    for path in iter_documents(cfg.data_dir, cfg.supported_extensions):
        docs = load_file(path)
        chunks = split_documents(docs, cfg.chunk_size, cfg.chunk_overlap)
        if not chunks:
            continue
        already = store.existing_ids([c.id for c in chunks])
        fresh = [c for c in chunks if c.id not in already]
        skipped += len(chunks) - len(fresh)
        if fresh:
            vectors = embedder.embed([c.text for c in fresh])
            store.add(fresh, vectors)
            added += len(fresh)
        ingested.append(path.name)

    return {
        "ingested_files": ingested,
        "chunks_added": added,
        "skipped_duplicate_chunks": skipped,
        "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
    }


def answer(cfg: Config, question: str, top_k: Optional[int] = None) -> dict:
    """Retrieve context and generate a grounded answer, with per-stage timing."""
    embedder = get_embedding_provider(cfg)
    chat = get_chat_provider(cfg)
    store = _store(cfg)
    k = top_k or cfg.top_k

    t0 = time.perf_counter()
    q_vec = embedder.embed([question])[0]
    hits = store.query(q_vec, k)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    # Keep only chunks similar enough to the question. This drops weakly-related
    # chunks the retriever always returns to fill top_k, so the answer and the
    # cited sources reflect what's actually relevant — not everything fetched.
    relevant = [h for h in hits if h["score"] >= cfg.score_threshold]

    # De-duplicate sources by file + page so the same passage isn't listed twice.
    seen = set()
    sources = []
    for h in relevant:
        key = (h["source"], h["page"])
        if key not in seen:
            seen.add(key)
            sources.append(h)

    context = "\n\n".join(
        f"[{h['source']}"
        + (f" p.{h['page']}" if h["page"] is not None else "")
        + f"]\n{h['snippet']}"
        for h in relevant
    )[: cfg.context_limit]
    user_msg = f"Context:\n{context}\n\nQuestion: {question}"

    t1 = time.perf_counter()
    text = chat.generate(SYSTEM_PROMPT, user_msg, cfg.temperature)
    generation_ms = (time.perf_counter() - t1) * 1000

    return {
        "answer": text,
        "sources": sources,
        "model": chat.model,
        "retrieval_ms": round(retrieval_ms, 1),
        "generation_ms": round(generation_ms, 1),
    }
