#!/usr/bin/env python3
"""Index everything under data/ from the command line.

Usage:  python ingest.py
"""
from app.config import load_config
from app.rag_pipeline import ingest

if __name__ == "__main__":
    cfg = load_config()
    result = ingest(cfg)
    print(
        f"Indexed {len(result['ingested_files'])} file(s): "
        f"+{result['chunks_added']} chunks, "
        f"{result['skipped_duplicate_chunks']} duplicates skipped, "
        f"in {result['elapsed_ms']} ms."
    )
