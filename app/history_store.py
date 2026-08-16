"""Chat-history logging in MongoDB (a document store).

Each query is saved as one document — question, answer, the sources used, model,
latency, and a timestamp — which is naturally document-shaped (nested arrays,
varying fields), so a document DB fits better than relational tables here.

The feature is optional and fault-tolerant: if MONGODB_URI is unset, or Mongo is
unreachable, logging silently no-ops and the app keeps answering. History is a
convenience, never a dependency of the core RAG flow.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from .config import Config

_client = None  # cached MongoClient across calls
_ttl_ready = False  # whether the TTL index has been ensured
HISTORY_TTL_DAYS = 3  # auto-delete chat history older than this many days


def _collection(cfg: Config):
    """Return the history collection, or None if history isn't configured/reachable."""
    global _client, _ttl_ready
    if not cfg.mongodb_uri:
        return None
    try:
        if _client is None:
            from pymongo import MongoClient  # lazy

            _client = MongoClient(cfg.mongodb_uri, serverSelectionTimeoutMS=3000)
        col = _client[cfg.mongodb_db]["chat_history"]
        if not _ttl_ready:
            # TTL index: MongoDB auto-deletes documents whose created_at is older
            # than HISTORY_TTL_DAYS. Created once; idempotent on repeat calls.
            col.create_index("created_at", expireAfterSeconds=HISTORY_TTL_DAYS * 86400)
            _ttl_ready = True
        return col
    except Exception:
        return None


def log_query(cfg: Config, question: str, result: dict) -> None:
    """Fire-and-forget: store one query. Never raises into the caller."""
    col = _collection(cfg)
    if col is None:
        return
    try:
        col.insert_one({
            "question": question,
            "answer": result.get("answer", ""),
            "sources": [
                {"source": s.get("source"), "page": s.get("page"),
                 "score": s.get("score")}
                for s in result.get("sources", [])
            ],
            "model": result.get("model"),
            "retrieval_ms": result.get("retrieval_ms"),
            "generation_ms": result.get("generation_ms"),
            "created_at": datetime.now(timezone.utc),
        })
    except Exception:
        pass  # logging must never break answering


def recent(cfg: Config, n: int = 20) -> list[dict]:
    """Most recent queries, newest first. Empty list if unavailable."""
    col = _collection(cfg)
    if col is None:
        return []
    try:
        docs = col.find({}, {"_id": 0}).sort("created_at", -1).limit(n)
        out = []
        for d in docs:
            ts = d.get("created_at")
            out.append({
                "question": d.get("question", ""),
                "answer": d.get("answer", ""),
                "model": d.get("model"),
                "created_at": ts.isoformat() if isinstance(ts, datetime) else None,
            })
        return out
    except Exception:
        return []
