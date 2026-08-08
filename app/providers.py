"""Pluggable chat + embedding providers.

Switching between a hosted API (proprietary) and local Ollama (open-source) is a
single config value here, not two code paths. SDK imports are lazy so the module
loads even when a given provider's package or key is absent.
"""
from __future__ import annotations

from typing import Protocol

from .config import Config


class ProviderError(RuntimeError):
    """Raised when a provider is misconfigured (e.g. missing API key)."""


# --------------------------------------------------------------------------- #
# Interfaces
# --------------------------------------------------------------------------- #
class ChatProvider(Protocol):
    name: str
    model: str

    def generate(self, system: str, user: str, temperature: float) -> str: ...


class EmbeddingProvider(Protocol):
    name: str
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #
class OpenAIChat:
    name = "openai"

    def __init__(self, cfg: Config):
        if not cfg.openai_api_key:
            raise ProviderError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set. "
                "Add it to .env, or set the provider to 'ollama' for local mode."
            )
        from openai import OpenAI  # lazy

        # base_url lets us point at any OpenAI-compatible endpoint
        # (OpenRouter, Together, a local proxy, ...). None => OpenAI default.
        self._client = OpenAI(api_key=cfg.openai_api_key, base_url=cfg.openai_base_url)
        self.model = cfg.chat_model

    def generate(self, system: str, user: str, temperature: float) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class OpenAIEmbedding:
    name = "openai"

    def __init__(self, cfg: Config):
        if not cfg.openai_api_key:
            raise ProviderError(
                "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set."
            )
        from openai import OpenAI  # lazy

        self._client = OpenAI(api_key=cfg.openai_api_key, base_url=cfg.openai_base_url)
        self.model = cfg.embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


# --------------------------------------------------------------------------- #
# Anthropic (chat only; embeddings fall back to openai/ollama)
# --------------------------------------------------------------------------- #
class AnthropicChat:
    name = "anthropic"

    def __init__(self, cfg: Config):
        if not cfg.anthropic_api_key:
            raise ProviderError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set."
            )
        from anthropic import Anthropic  # lazy

        self._client = Anthropic(api_key=cfg.anthropic_api_key)
        self.model = cfg.chat_model

    def generate(self, system: str, user: str, temperature: float) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


# --------------------------------------------------------------------------- #
# Ollama (local, no key)
# --------------------------------------------------------------------------- #
class OllamaChat:
    name = "ollama"

    def __init__(self, cfg: Config):
        self._base = cfg.ollama_base_url.rstrip("/")
        self.model = cfg.chat_model

    def generate(self, system: str, user: str, temperature: float) -> str:
        import requests  # lazy

        r = requests.post(
            f"{self._base}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "options": {"temperature": temperature},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]


class OllamaEmbedding:
    name = "ollama"

    def __init__(self, cfg: Config):
        self._base = cfg.ollama_base_url.rstrip("/")
        self.model = cfg.embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        import requests  # lazy

        out: list[list[float]] = []
        for text in texts:
            r = requests.post(
                f"{self._base}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=120,
            )
            r.raise_for_status()
            out.append(r.json()["embedding"])
        return out


# --------------------------------------------------------------------------- #
# Jina (hosted embeddings, free tier — used for cloud deploy where there's no GPU)
# --------------------------------------------------------------------------- #
class JinaEmbedding:
    name = "jina"

    def __init__(self, cfg: Config):
        if not cfg.jina_api_key:
            raise ProviderError(
                "EMBEDDING_PROVIDER=jina but JINA_API_KEY is not set. "
                "Get a free key at https://jina.ai/embeddings and add it to .env."
            )
        self._key = cfg.jina_api_key
        self.model = cfg.embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        import requests  # lazy

        r = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": texts,
                # 'task' is required for jina-embeddings-v3+. retrieval.passage
                # encodes text for retrieval indexing; it also works acceptably
                # for the query side in this simple setup.
                "task": "retrieval.passage",
            },
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()["data"]
        # Preserve input order.
        return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]


# --------------------------------------------------------------------------- #
# Google Gemini (hosted embeddings via AI Studio — free tier, API key only)
# --------------------------------------------------------------------------- #
class GeminiEmbedding:
    name = "gemini"
    _URL = ("https://generativelanguage.googleapis.com/v1beta/"
            "models/{model}:batchEmbedContents")

    def __init__(self, cfg: Config):
        if not cfg.gemini_api_key:
            raise ProviderError(
                "EMBEDDING_PROVIDER=gemini but GEMINI_API_KEY is not set. "
                "Get a free key at https://aistudio.google.com/apikey and add it to .env."
            )
        self._key = cfg.gemini_api_key
        self.model = cfg.embedding_model  # e.g. gemini-embedding-001

    def embed(self, texts: list[str]) -> list[list[float]]:
        import requests  # lazy

        url = self._URL.format(model=self.model)
        out: list[list[float]] = []
        # The batch endpoint caps requests per call; page in chunks of 100.
        for start in range(0, len(texts), 100):
            batch = texts[start:start + 100]
            body = {
                "requests": [
                    {"model": f"models/{self.model}",
                     "content": {"parts": [{"text": t}]}}
                    for t in batch
                ]
            }
            r = requests.post(
                url,
                headers={"x-goog-api-key": self._key,
                         "Content-Type": "application/json"},
                json=body,
                timeout=60,
            )
            r.raise_for_status()
            out.extend(e["values"] for e in r.json()["embeddings"])
        return out


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
_CHAT = {"openai": OpenAIChat, "anthropic": AnthropicChat, "ollama": OllamaChat}
_EMBED = {
    "openai": OpenAIEmbedding,
    "ollama": OllamaEmbedding,
    "jina": JinaEmbedding,
    "gemini": GeminiEmbedding,
}


def get_chat_provider(cfg: Config) -> ChatProvider:
    try:
        return _CHAT[cfg.chat_provider](cfg)
    except KeyError:
        raise ProviderError(
            f"Unknown chat provider '{cfg.chat_provider}'. "
            f"Choose one of: {', '.join(_CHAT)}."
        )


def get_embedding_provider(cfg: Config) -> EmbeddingProvider:
    try:
        return _EMBED[cfg.embedding_provider](cfg)
    except KeyError:
        raise ProviderError(
            f"Unknown embedding provider '{cfg.embedding_provider}'. "
            f"Choose one of: {', '.join(_EMBED)}."
        )
