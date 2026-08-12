# Local RAG Assistant

Ask questions about **your own documents** and get answers grounded in them, with
the source file shown for each answer. The same codebase runs two ways:

- **Local mode** — everything runs on your own machine ([Ollama](https://ollama.com)
  for the model, Chroma for storage). No API keys, and no document ever leaves
  your computer. Good for private or work-internal files.
- **Hosted mode** — the model and vector store run as cloud services (Gemini or
  OpenRouter for the model, Qdrant Cloud for storage). No GPU needed, and it can
  be deployed to the cloud (e.g. Google Cloud Run).

Switching between the two is a **config change, not a code change** — you edit a
few provider settings and re-index; no application code is touched. See
[Switching modes](#switching-modes) below.

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

## Switching modes

Three settings in `.env` decide where things run. Change them, re-index, done —
no code changes.

| Setting | Local (private, on your GPU) | Hosted (cloud services) |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `openai` (OpenRouter) |
| `EMBEDDING_PROVIDER` | `ollama` | `gemini` |
| `VECTOR_STORE` | `chroma` | `qdrant` |
| API keys needed | none | OpenRouter + Gemini + Qdrant |

**To switch to local:** set the three values to `ollama` / `ollama` / `chroma`,
set `config.yaml` models to `llama3.1` and `bge-m3`, pull them
(`ollama pull llama3.1 && ollama pull bge-m3`), then rebuild the index:

```bash
rm -rf storage/          # clear the old index
python ingest.py
```

**To switch to hosted:** set the three values to `openai` / `gemini` / `qdrant`,
fill the matching keys in `.env`, set `config.yaml` models accordingly, then
re-index (`python ingest.py`).

**Why re-index?** Local and hosted use different embedding models (bge-m3 is
1024-dimensional, Gemini is 3072), and their vectors aren't interchangeable — so
switching the embedding provider means rebuilding the index. Check the current
mode any time:

```bash
python -c "from app.config import load_config; c=load_config(); print('chat:',c.chat_provider,'| embed:',c.embedding_provider,'| store:',c.vector_store)"
```

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

## Evaluation

`evaluate.py` measures properties of the *system* rather than answers to one
specific document, so the metrics stay meaningful after you swap in your own
corpus:

- **Retrieval self-test** — samples indexed chunks, queries with a fragment of
  each, and checks whether that chunk comes back (reports hit@k and MRR across
  several `top_k` values, plus retrieval latency). Runs on local embeddings
  only: no API cost, fully reproducible on any corpus.
- **Faithfulness / abstention** — asks questions whose answers are *not* in the
  corpus and checks that the system declines instead of fabricating, which tests
  the grounding prompt independently of document content.

```bash
python evaluate.py                    # retrieval self-test + faithfulness
python evaluate.py --no-faithfulness  # retrieval only (no chat-model calls)
```

### Sample results

Retrieval self-test on a 19-chunk corpus (bge-m3 local embeddings):

| top_k | hit@k | MRR   | latency p50 | latency p95 |
|-------|-------|-------|-------------|-------------|
| 3     | 1.00  | 0.965 | ~357 ms     | ~399 ms     |
| 5     | 1.00  | 0.965 | ~352 ms     | ~376 ms     |
| 10    | 1.00  | 0.965 | ~363 ms     | ~386 ms     |

The correct chunk is retrieved every time, and MRR ~0.97 means it is almost
always ranked first. Recall and MRR are **flat across `top_k`**, so raising
`top_k` only adds context length (and cost, on paid models) without improving
retrieval — hence `top_k=4` as the default. Retrieval latency is stable and
independent of `top_k`, so end-to-end latency is dominated by generation, not
search. These numbers are content-agnostic and reproduce on any indexed corpus.

`tests/` holds fast, no-network unit tests (`pytest`) for chunking, duplicate
detection, and config precedence. `tests/golden_qa.json` is a document-specific
regression fixture for manual runs — kept separate from the content-agnostic
evaluation above.

## Roadmap

- [x] Content-agnostic evaluation: retrieval self-test (hit@k, MRR) and
      faithfulness/abstention.
- [ ] Reranker and hybrid (keyword + vector) retrieval.
- [ ] Retrieval-latency-vs-corpus-size scaling curve.
- [ ] Streaming responses.
- [ ] OCR for scanned PDFs.

## License

MIT — see `LICENSE`.
