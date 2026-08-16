"""FastAPI service layer.

Endpoints:
    GET  /health   liveness + which providers are active
    GET  /stats    indexed chunk count + provider info
    POST /ingest   (re)index everything under data/
    POST /query    ask a question, get a grounded answer + sources + timings

Run locally:   uvicorn app.api:app --reload
Interactive docs at /docs (Swagger) once running.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import load_config
from .providers import ProviderError
from . import rag_pipeline
from .schemas import (
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    StatsResponse,
)

app = FastAPI(
    title="Local RAG Assistant",
    description="Privacy-focused RAG over your own documents. Local (Ollama) or "
    "hosted (OpenAI/Anthropic) — switch with one config value.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _chunk_count() -> int:
    cfg = load_config()
    try:
        return rag_pipeline._store(cfg).count()
    except Exception:
        return 0


_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Simple end-user page. API docs remain at /docs."""
    return FileResponse(_WEB_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    cfg = load_config()
    return HealthResponse(
        chat_provider=cfg.chat_provider,
        embedding_provider=cfg.embedding_provider,
        chat_model=cfg.chat_model,
        document_chunks=_chunk_count(),
    )


@app.get("/history")
def history(limit: int = 20) -> list[dict]:
    """Recent questions and answers (empty if history logging isn't configured)."""
    from .history_store import recent
    cfg = load_config()
    return recent(cfg, min(max(limit, 1), 100))


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    cfg = load_config()
    return StatsResponse(
        document_chunks=_chunk_count(),
        chat_provider=cfg.chat_provider,
        embedding_provider=cfg.embedding_provider,
        chat_model=cfg.chat_model,
        embedding_model=cfg.embedding_model,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    cfg = load_config()
    try:
        result = rag_pipeline.ingest(cfg)
    except ProviderError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return IngestResponse(**result)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    cfg = load_config()
    try:
        result = rag_pipeline.answer(cfg, req.question, req.top_k)
    except ProviderError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return QueryResponse(
        answer=result["answer"],
        model=result["model"],
        retrieval_ms=result["retrieval_ms"],
        generation_ms=result["generation_ms"],
        sources=[SourceChunk(**s) for s in result["sources"]],
    )
