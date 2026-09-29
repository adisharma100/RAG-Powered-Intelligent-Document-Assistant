"""Lightweight evaluation harness.

Metrics over eval/questions.json:
  * retrieval_hit_rate - was the expected source among the retrieved chunks?
  * answer_keyword_rate - does the answer contain the expected fact?
  * refusal_rate       - for out-of-scope questions, did the assistant correctly refuse?

  python -m rag_assistant.evaluation eval/questions.json
"""
from __future__ import annotations

import json
import sys

from .pipeline import RAGAssistant


def evaluate(assistant: RAGAssistant, cases: list[dict]) -> dict:
    in_scope = [c for c in cases if c.get("expected_source")]
    out_scope = [c for c in cases if not c.get("expected_source")]
    hits = kw = refused = 0
    details = []

    for c in in_scope:
        retrieved = [d.metadata.get("source") for d, s in assistant.retrieve(c["question"])
                     if s >= assistant.settings.min_relevance]
        hit = c["expected_source"] in retrieved
        ans = assistant.ask(c["question"])
        has_kw = all(k.lower() in ans.answer.lower() for k in c.get("expected_keywords", []))
        hits += hit
        kw += has_kw
        details.append({"question": c["question"], "retrieval_hit": hit, "keyword_hit": has_kw})

    for c in out_scope:
        ans = assistant.ask(c["question"])
        ok = not ans.grounded
        refused += ok
        details.append({"question": c["question"], "correctly_refused": ok})

    return {
        "retrieval_hit_rate": hits / len(in_scope) if in_scope else None,
        "answer_keyword_rate": kw / len(in_scope) if in_scope else None,
        "refusal_rate": refused / len(out_scope) if out_scope else None,
        "details": details,
    }


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "eval/questions.json"
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    report = evaluate(RAGAssistant(), cases)
    for d in report.pop("details"):
        print(d)
    print("\nSUMMARY:", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
