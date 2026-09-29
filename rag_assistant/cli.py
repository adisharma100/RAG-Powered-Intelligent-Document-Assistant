"""Command-line interface.

  python -m rag_assistant.cli ingest data/sample_docs
  python -m rag_assistant.cli ask "How many vacation days do employees get?"
  python -m rag_assistant.cli chat
"""
from __future__ import annotations

import argparse

from .pipeline import RAGAssistant


def _print_answer(ans) -> None:
    print(f"\nAnswer: {ans.answer}\n")
    for i, s in enumerate(ans.sources, start=1):
        page = f", page {s.page}" if s.page else ""
        print(f"  [{i}] {s.source}{page}  (relevance {s.score})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Document Assistant")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ing = sub.add_parser("ingest", help="Index a file or folder")
    p_ing.add_argument("path")
    p_ask = sub.add_parser("ask", help="Ask one question")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=None)
    sub.add_parser("chat", help="Interactive Q&A loop")
    sub.add_parser("reset", help="Delete the vector index")
    args = parser.parse_args()

    assistant = RAGAssistant()
    if args.cmd == "ingest":
        n = assistant.ingest(args.path)
        print(f"Indexed {n} chunks. Total in store: {assistant.count()}")
    elif args.cmd == "ask":
        _print_answer(assistant.ask(args.question, args.k))
    elif args.cmd == "chat":
        print(f"{assistant.count()} chunks indexed. Type 'exit' to quit.")
        while (q := input("\nYou: ").strip()).lower() not in {"exit", "quit"}:
            if q:
                _print_answer(assistant.ask(q))
    elif args.cmd == "reset":
        assistant.reset()
        print("Index cleared.")


if __name__ == "__main__":
    main()
