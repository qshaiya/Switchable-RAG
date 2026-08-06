"""Request / response schemas for the RAG API.

These Pydantic models are the public contract of the service. The JD asks for
"handling request/response schemas" explicitly, so every endpoint validates its
input and serialises its output through the models below.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question.")
    top_k: Optional[int] = Field(
        None, ge=1, le=20, description="Override config's top_k for this request."
    )


class SourceChunk(BaseModel):
    source: str = Field(..., description="File the chunk came from.")
    page: Optional[int] = Field(None, description="Page number (PDF only).")
    score: float = Field(..., description="Similarity score (higher = closer).")
    snippet: str = Field(..., description="The retrieved text chunk.")


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    model: str
    retrieval_ms: float
    generation_ms: float


class IngestResponse(BaseModel):
    ingested_files: list[str]
    chunks_added: int
    skipped_duplicate_chunks: int
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: str = "ok"
    chat_provider: str
    embedding_provider: str
    chat_model: str
    document_chunks: int


class StatsResponse(BaseModel):
    document_chunks: int
    chat_provider: str
    embedding_provider: str
    chat_model: str
    embedding_model: str


class ErrorResponse(BaseModel):
    detail: str
