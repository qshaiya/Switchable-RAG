#!/usr/bin/env python3
"""Ask questions from the terminal (no server needed).

Usage:  python chat.py
"""
from app.config import load_config
from app.rag_pipeline import answer

if __name__ == "__main__":
    cfg = load_config()
    print(f"Provider: {cfg.chat_provider} ({cfg.chat_model}). Ctrl-C to quit.\n")
    try:
        while True:
            q = input("You: ").strip()
            if not q:
                continue
            res = answer(cfg, q)
            print(f"\nAssistant: {res['answer']}")
            if res["sources"]:
                cites = ", ".join(
                    s["source"] + (f" p.{s['page']}" if s["page"] else "")
                    for s in res["sources"]
                )
                print(f"Sources: {cites}")
            print(
                f"[retrieval {res['retrieval_ms']} ms | "
                f"generation {res['generation_ms']} ms]\n"
            )
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
