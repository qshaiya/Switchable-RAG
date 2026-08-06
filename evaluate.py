#!/usr/bin/env python3
"""Content-agnostic evaluation of the RAG system.

Unlike a fixed golden-Q&A set (which only measures one specific document), these
metrics describe properties of the *system* and stay meaningful after the user
swaps in their own documents:

  1. Retrieval self-test  - for a sample of indexed chunks, query with a fragment
     of each chunk and check whether that chunk is retrieved back. Measures
     hit@k and MRR. Runs on local embeddings only: no API cost, fully
     reproducible on any corpus.

  2. Faithfulness / abstention - ask questions whose answers are NOT in the
     corpus and check that the system declines instead of hallucinating.
     Measures the grounding prompt, independent of document content.

Usage:
    python evaluate.py                    # retrieval self-test + faithfulness
    python evaluate.py --sample 30        # more retrieval samples
    python evaluate.py --no-faithfulness  # retrieval only (no chat calls)
"""
from __future__ import annotations

import argparse
import random
import statistics
import time

from app.config import load_config
from app.rag_pipeline import _store, answer, ingest
from app.providers import get_embedding_provider

# Out-of-domain questions: almost certainly not answerable from a typical
# personal/enterprise document set. A grounded system should decline these.
OOD_QUESTIONS = [
    "What is the company's current stock price?",
    "What is the CEO's personal mobile phone number?",
    "Who won the FIFA World Cup in 2018?",
    "\u4f1a\u793e\u306e\u73fe\u5728\u306e\u682a\u4fa1\u306f\u3044\u304f\u3089\u3067\u3059\u304b\uff1f",  # JP: current stock price?
]

# Signals that the model declined rather than fabricated (EN + JP).
ABSTAIN_MARKERS = [
    "don't have", "do not have", "not enough information", "no information",
    "cannot find", "couldn't find", "not in the", "isn't in", "is not in",
    "not provided", "no relevant", "don't know", "unable to", "not mentioned",
    "\u60c5\u5831\u304c\u3042\u308a\u307e\u305b\u3093", "\u898b\u3064\u304b\u308a\u307e\u305b\u3093",
    "\u308f\u304b\u308a\u307e\u305b\u3093", "\u8a18\u8f09\u304c\u3042\u308a\u307e\u305b\u3093",
    "\u542b\u307e\u308c\u3066\u3044\u307e\u305b\u3093",
]


def fragment(text: str) -> str:
    """A middle slice of the chunk: a partial query, not the identical vector."""
    text = text.strip()
    if len(text) <= 60:
        return text
    a, b = int(len(text) * 0.25), int(len(text) * 0.75)
    return text[a:b]


def retrieval_self_test(cfg, sample_n: int, k_values: list[int]) -> None:
    embedder = get_embedding_provider(cfg)
    store = _store(cfg)
    chunks = store.all_chunks()

    if not chunks:
        print("No indexed chunks. Run `python ingest.py` first.")
        return

    sample = random.sample(chunks, min(sample_n, len(chunks)))
    print(f"Retrieval self-test on {len(sample)} of {len(chunks)} indexed chunks "
          f"(embeddings: {cfg.embedding_provider}/{cfg.embedding_model})\n")

    print(f"{'top_k':>6}  {'hit_rate':>9}  {'MRR':>6}  {'p50_ms':>7}  {'p95_ms':>7}")
    print("-" * 44)
    for k in k_values:
        hits = 0
        rr_sum = 0.0
        lat: list[float] = []
        for _id, text in sample:
            q = fragment(text)
            t0 = time.perf_counter()
            vec = embedder.embed([q])[0]
            results = store.query(vec, k)
            lat.append((time.perf_counter() - t0) * 1000)

            rank = next((i for i, r in enumerate(results) if r["snippet"] == text), None)
            if rank is not None:
                hits += 1
                rr_sum += 1.0 / (rank + 1)

        n = len(sample)
        p50 = statistics.median(lat)
        p95 = sorted(lat)[max(0, int(len(lat) * 0.95) - 1)]
        print(f"{k:>6}  {hits / n:>9.3f}  {rr_sum / n:>6.3f}  {p50:>7.1f}  {p95:>7.1f}")
    print()


def _answer_with_retry(cfg, q: str, retries: int = 2):
    delay = 4.0
    for attempt in range(retries + 1):
        try:
            return answer(cfg, q)
        except Exception as exc:
            if attempt == retries:
                print(f"    ! {type(exc).__name__} (giving up)")
                return None
            time.sleep(delay)
            delay *= 2
    return None


def faithfulness_test(cfg) -> None:
    print(f"Faithfulness test: {len(OOD_QUESTIONS)} out-of-domain questions "
          f"(chat: {cfg.chat_provider}/{cfg.chat_model})\n")
    abstained = 0
    answered = 0
    for q in OOD_QUESTIONS:
        res = _answer_with_retry(cfg, q)
        if res is None:
            continue
        answered += 1
        low = res["answer"].lower()
        did_abstain = any(m in low for m in ABSTAIN_MARKERS)
        abstained += did_abstain
        mark = "OK " if did_abstain else "!! "
        print(f"  {mark}{q[:50]:50s}  {'declined' if did_abstain else 'ANSWERED (hallucination risk)'}")

    if answered:
        print(f"\n  Abstention rate: {abstained}/{answered} = {abstained / answered:.0%}  "
              f"(higher is better)")
    else:
        print("\n  Faithfulness test skipped: chat provider unavailable.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=20,
                        help="Number of chunks to sample for the retrieval self-test.")
    parser.add_argument("--no-faithfulness", action="store_true",
                        help="Skip the faithfulness test (avoids any chat-model calls).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    cfg = load_config()

    stats = ingest(cfg)  # idempotent
    print(f"Index: {stats['skipped_duplicate_chunks'] + stats['chunks_added']} chunks "
          f"across the corpus.\n")

    print("=" * 60)
    retrieval_self_test(cfg, args.sample, k_values=[3, 5, 10])
    print("=" * 60)
    if not args.no_faithfulness:
        faithfulness_test(cfg)


if __name__ == "__main__":
    main()
