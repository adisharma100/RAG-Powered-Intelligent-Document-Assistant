from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag_assistant import RAGAssistant, Settings
from rag_assistant.evaluation import evaluate
from rag_assistant.prompts import NO_ANSWER

import json

SAMPLE = Path(__file__).parent.parent / "data" / "sample_docs"


def make_assistant(tmp_path, embeddings, responses=None, min_relevance=0.15):
    settings = Settings(chroma_dir=str(tmp_path / "db"), chunk_size=400, chunk_overlap=60, min_relevance=min_relevance)
    llm = FakeListChatModel(responses=responses or ["Employees get 24 days of vacation [1]."])
    return RAGAssistant(settings=settings, embeddings=embeddings, llm=llm)


def test_ingest_is_idempotent(tmp_path, embeddings):
    a = make_assistant(tmp_path, embeddings)
    n1 = a.ingest(SAMPLE)
    total1 = a.count()
    a.ingest(SAMPLE)  # same files again
    assert n1 > 0 and a.count() == total1


def test_retrieval_finds_right_document(tmp_path, embeddings):
    a = make_assistant(tmp_path, embeddings)
    a.ingest(SAMPLE)
    top_doc, _ = a.retrieve("How many days of paid vacation do employees get?", k=1)[0]
    assert top_doc.metadata["source"] == "employee_handbook.md"


def test_grounded_answer_returns_sources(tmp_path, embeddings):
    a = make_assistant(tmp_path, embeddings)
    a.ingest(SAMPLE)
    ans = a.ask("How many vacation days do employees get?")
    assert ans.grounded and ans.sources
    assert ans.sources[0].source == "employee_handbook.md"


def test_relevance_gate_blocks_llm_on_out_of_scope(tmp_path, embeddings):
    # LLM would answer confidently - but the gate must stop it from being called at all.
    a = make_assistant(tmp_path, embeddings, responses=["Brazil won."])
    a.ingest(SAMPLE)
    ans = a.ask("Who won the football world cup in 2018?")
    assert not ans.grounded and ans.answer == NO_ANSWER and ans.sources == []


def test_llm_refusal_is_marked_ungrounded(tmp_path, embeddings):
    a = make_assistant(tmp_path, embeddings, responses=[NO_ANSWER])
    a.ingest(SAMPLE)
    ans = a.ask("How many vacation days do employees get?")
    assert not ans.grounded and ans.sources == []


def test_evaluation_harness_runs(tmp_path, embeddings):
    a = make_assistant(tmp_path, embeddings, responses=["24 days. 8 days. Tuesday Thursday. 9 USD. 30 days. 2 GB. AES-256. 6 hours. 1 hour."], min_relevance=0.3)
    a.ingest(SAMPLE)
    cases = json.loads((SAMPLE.parent.parent / "eval" / "questions.json").read_text())
    report = evaluate(a, cases)
    assert report["retrieval_hit_rate"] >= 0.8
    assert report["refusal_rate"] == 1.0
