"""Fast unit tests — no network, no models. Run with: pytest

These cover content-agnostic invariants of the pipeline (chunking, duplicate
detection, config precedence). The document-specific golden_qa.json in this
folder is a regression fixture for manual/integration runs, not a unit test,
because it needs a live model.
"""
from app.config import load_config
from app.document_loader import LoadedText
from app.text_splitter import split_documents


def test_chunking_respects_size_and_overlap():
    text = "x" * 2000
    docs = [LoadedText(text=text, source="a.txt", page=None)]
    chunks = split_documents(docs, chunk_size=700, chunk_overlap=100)
    assert len(chunks) > 1
    assert all(len(c.text) <= 700 for c in chunks)


def test_dedup_ids_are_stable_and_content_addressed():
    docs = [LoadedText(text="hello world " * 50, source="a.txt", page=1)]
    first = split_documents(docs, 200, 20)
    second = split_documents(docs, 200, 20)
    # Same content -> same ids, so re-ingestion is a no-op (dedup mechanism).
    assert [c.id for c in first] == [c.id for c in second]
    # Different source -> different ids.
    other = split_documents(
        [LoadedText(text="hello world " * 50, source="b.txt", page=1)], 200, 20
    )
    assert {c.id for c in first}.isdisjoint({c.id for c in other})


def test_empty_document_yields_no_chunks():
    docs = [LoadedText(text="   ", source="a.txt", page=None)]
    assert split_documents(docs, 700, 100) == []


def test_config_loads_with_defaults():
    cfg = load_config()
    assert cfg.chunk_size > 0
    assert cfg.top_k >= 1
    assert cfg.chat_provider in {"openai", "anthropic", "ollama"}
