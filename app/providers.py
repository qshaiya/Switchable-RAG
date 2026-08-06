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
# Factories
# --------------------------------------------------------------------------- #
_CHAT = {"openai": OpenAIChat, "anthropic": AnthropicChat, "ollama": OllamaChat}
_EMBED = {"openai": OpenAIEmbedding, "ollama": OllamaEmbedding}


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
