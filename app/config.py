"""Configuration loading.

Precedence (low -> high):
    config.yaml defaults  ->  environment variables (.env)

Secrets (API keys) live ONLY in the environment / .env and never in config.yaml,
so config.yaml is safe to commit. Provider selection can be set in either place;
the environment wins so a user can flip local<->cloud without editing YAML.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()  # read .env if present; no-op otherwise

_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # llm
    chat_provider: str = "openai"          # openai | anthropic | ollama
    chat_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    context_limit: int = 6000
    # embedding
    embedding_provider: str = "openai"     # openai | ollama
    embedding_model: str = "text-embedding-3-small"
    # retrieval
    chunk_size: int = 700
    chunk_overlap: int = 100
    top_k: int = 4
    score_threshold: float = 0.5   # min cosine similarity to count a chunk as relevant
    # paths
    data_dir: Path = field(default_factory=lambda: _ROOT / "data")
    vector_store_dir: Path = field(default_factory=lambda: _ROOT / "storage" / "chroma_db")
    supported_extensions: tuple[str, ...] = (".pdf", ".txt", ".md")
    # secrets (never persisted to yaml)
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None   # e.g. https://openrouter.ai/api/v1
    anthropic_api_key: Optional[str] = None
    jina_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    # vector store: "chroma" (local, default) or "qdrant" (hosted)
    vector_store: str = "chroma"
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(yaml_path: Optional[Path] = None) -> Config:
    raw = _load_yaml(yaml_path or (_ROOT / "config.yaml"))
    llm = raw.get("llm", {})
    emb = raw.get("embedding", {})
    ret = raw.get("retrieval", {})
    paths = raw.get("paths", {})
    docs = raw.get("documents", {})

    cfg = Config(
        chat_provider=os.getenv("LLM_PROVIDER", llm.get("provider", "openai")),
        chat_model=llm.get("chat_model", "gpt-4o-mini"),
        temperature=float(llm.get("temperature", 0.0)),
        context_limit=int(llm.get("context_limit", 6000)),
        embedding_provider=os.getenv(
            "EMBEDDING_PROVIDER", emb.get("provider", "openai")
        ),
        embedding_model=emb.get("model", "text-embedding-3-small"),
        chunk_size=int(ret.get("chunk_size", 700)),
        chunk_overlap=int(ret.get("chunk_overlap", 100)),
        top_k=int(ret.get("top_k", 4)),
        score_threshold=float(ret.get("score_threshold", 0.5)),
        data_dir=_ROOT / paths.get("data", "data"),
        vector_store_dir=_ROOT / paths.get("vector_store", "storage/chroma_db"),
        supported_extensions=tuple(
            docs.get("supported_extensions", [".pdf", ".txt", ".md"])
        ),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        jina_api_key=os.getenv("JINA_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        vector_store=os.getenv("VECTOR_STORE", raw.get("vector_store", {}).get("provider", "chroma")),
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    )
    return cfg
