# Local RAG Assistant

Ask questions over **your own documents** and get answers grounded in them, with
source citations. Runs two ways from the same codebase:

- **Hosted mode** — bring an OpenAI or Anthropic API key. No GPU needed; anyone
  can clone and run it.
- **Local mode** — run entirely on your machine with [Ollama](https://ollama.com).
  No API key, no data leaves your computer. Built for privacy-sensitive and
  enterprise-internal documents.

Switching between the two is **one config value**, not two code paths.

---

## Quick start (default: OpenRouter chat + local embedding)

Cheapest for development — generation via a hosted model on OpenRouter, embeddings
on your own GPU via Ollama (no embedding API cost, data for retrieval stays local).

```bash
git clone <repository-url>
cd local-rag-assistant

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

ollama pull bge-m3            # local embedding model

cp .env.example .env          # paste your OpenRouter key (sk-or-...) into .env
cp sample_data/sample_policy.txt data/

python ingest.py              # index your documents
uvicorn app.api:app --reload  # API + Swagger docs at http://localhost:8000/docs
```

To use OpenAI directly instead of OpenRouter: put your `sk-...` key in `.env`,
delete the `OPENAI_BASE_URL` line, and set `chat_model: gpt-4o-mini` in `config.yaml`.

Ask a question:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many remote days are allowed per week?"}'
```

## Quick start (local — no API key)

```bash
ollama pull bge-m3
ollama pull llama3.1
```

In `config.yaml` set both providers to `ollama` (chat model `llama3.1`,
embedding model `bge-m3`), then run the same `ingest` / `uvicorn` commands. No
key required; nothing leaves your machine.

---

## Two interfaces

Once the server is running:

- **`http://localhost:8000/`** — a simple ask-and-answer page for end users
  (question in, answer + sources out, errors shown plainly).
- **`http://localhost:8000/docs`** — the interactive API reference (Swagger) for
  developers.

## Architecture

```
                      ┌──────────── FastAPI service (app/api.py) ────────────┐
   data/*.pdf|txt|md  │  /ingest   /query   /stats   /health   (Swagger /docs) │
          │           └──────────────────────┬───────────────────────────────┘
          ▼                                   │
   document_loader ─► text_splitter ─► rag_pipeline (core, framework-free)
   (source + page)     (chunk+overlap)        │
                                              ├─► providers  ── openai | anthropic | ollama
                                              │   (chat + embedding, pluggable)
                                              └─► vector_store ── ChromaDB (persistent, cosine)
```

The core (`rag_pipeline`) is plain importable Python, so the API and the
benchmark script share one definition of retrieval + generation.

## API

| Method | Path      | Purpose                                             |
|--------|-----------|-----------------------------------------------------|
| GET    | `/health` | Liveness + active providers + indexed chunk count   |
| GET    | `/stats`  | Provider/model info + indexed chunk count           |
| POST   | `/ingest` | (Re)index everything under `data/` — idempotent      |
| POST   | `/query`  | Ask a question → answer, sources, per-stage timings |

Every request/response is validated by Pydantic models in `app/schemas.py`.
`/query` returns `retrieval_ms` and `generation_ms` so latency is measurable
per call.

## Configuration

Defaults live in `config.yaml` (safe to commit — **no secrets**). Keys live only
in `.env` (gitignored). What you can tune: providers, chat/embedding models,
`chunk_size`, `chunk_overlap`, `top_k`, `temperature`, `context_limit`, and the
data / vector-store paths.

## Deploy to the cloud (Cloud Run example)

```bash
gcloud run deploy local-rag-assistant \
  --source . \
  --set-env-vars LLM_PROVIDER=openai,EMBEDDING_PROVIDER=openai \
  --set-secrets OPENAI_API_KEY=openai-key:latest \
  --region asia-northeast1 --allow-unauthenticated
```

The included `Dockerfile` reads Cloud Run's `$PORT`, so the same image runs
locally (`docker build -t rag . && docker run -p 8080:8080 rag`) and in the cloud.

## What this project demonstrates

- **RAG data pipeline** — loading, chunking, embedding, persistent vector store
  with source attribution (`app/rag_pipeline.py`, `app/vector_store.py`).
- **Backend & API development** — FastAPI service with validated request/response
  schemas and graceful error handling (`app/api.py`, `app/schemas.py`).
- **Model integration** — pluggable proprietary (OpenAI/Anthropic) and
  open-source (Ollama) backends behind one interface (`app/providers.py`).
- **Vector + relational awareness** — ChromaDB for vectors; content-hash IDs for
  duplicate detection.
- **Performance measurement** — per-request retrieval/generation timings, ready
  for the benchmark harness (next milestone).
- **Cloud-ready** — containerised, `$PORT`-aware, one-command Cloud Run deploy.

## Roadmap

- [ ] Benchmark / eval harness: golden Q&A set, A/B across chunk sizes, `top_k`,
      embedding + chat models; report recall@k, latency, and cost.
- [ ] Reranker and hybrid (keyword + vector) retrieval.
- [ ] Streaming responses and a minimal web UI.
- [ ] OCR for scanned PDFs.

## License

MIT — see `LICENSE`.
