"""The RAG orchestrator: ingest -> retrieve -> guard -> generate -> cite.

Hallucination mitigation is layered:
  1. Retrieval grounding  - the LLM only sees retrieved chunks.
  2. Relevance gate       - if nothing scores above MIN_RELEVANCE, the LLM is NOT called at all.
  3. Strict prompt        - "answer only from context, else say you don't know", temperature 0.
  4. Citations            - every answer is returned with the exact chunks it was based on.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from .chunking import chunk_documents
from .config import Settings
from .embeddings import get_embeddings
from .llm import get_llm
from .loaders import load_path
from .prompts import NO_ANSWER, RAG_PROMPT
from .vector_store import build_vector_store


@dataclass
class Source:
    source: str
    page: Optional[int]
    chunk_id: str
    score: float
    snippet: str


@dataclass
class Answer:
    question: str
    answer: str
    grounded: bool
    sources: List[Source] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class RAGAssistant:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        embeddings: Optional[Embeddings] = None,
        llm: Optional[BaseChatModel] = None,
    ):
        """`embeddings` and `llm` can be injected (used by the offline unit tests)."""
        self.settings = settings or Settings()
        self.embeddings = embeddings or get_embeddings(self.settings)
        self.store = build_vector_store(self.settings, self.embeddings)
        self._llm = llm
        self._chain = None

    # ---- lazy LLM so ingestion works without a running LLM -------------------------------
    @property
    def chain(self):
        if self._chain is None:
            llm = self._llm or get_llm(self.settings)
            self._chain = RAG_PROMPT | llm | StrOutputParser()
        return self._chain

    # ---- ingestion ------------------------------------------------------------------------
    def ingest(self, path: str | Path) -> int:
        """Load -> chunk -> embed -> store. Returns number of chunks written."""
        docs = load_path(path)
        if not docs:
            return 0
        chunks = chunk_documents(docs, self.settings.chunk_size, self.settings.chunk_overlap)
        self.store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
        return len(chunks)

    def count(self) -> int:
        return self.store._collection.count()

    def reset(self) -> None:
        self.store.delete_collection()
        self.store = build_vector_store(self.settings, self.embeddings)

    # ---- retrieval ------------------------------------------------------------------------
    def retrieve(self, question: str, k: Optional[int] = None):
        k = k or self.settings.top_k
        return self.store.similarity_search_with_relevance_scores(question, k=k)

    # ---- question answering ---------------------------------------------------------------
    def ask(self, question: str, k: Optional[int] = None) -> Answer:
        hits = [(d, s) for d, s in self.retrieve(question, k) if s >= self.settings.min_relevance]

        if not hits:  # relevance gate: never let the LLM improvise without evidence
            return Answer(question=question, answer=NO_ANSWER, grounded=False)

        context = "\n\n".join(f"[{i}] (source: {d.metadata.get('source')}) {d.page_content}"
                              for i, (d, _) in enumerate(hits, start=1))
        answer_text = self.chain.invoke({"context": context, "question": question}).strip()
        grounded = NO_ANSWER.lower() not in answer_text.lower()

        sources = [
            Source(
                source=d.metadata.get("source", "unknown"),
                page=None if d.metadata.get("page", -1) == -1 else d.metadata.get("page"),
                chunk_id=d.metadata.get("chunk_id", ""),
                score=round(float(s), 4),
                snippet=d.page_content[:300],
            )
            for d, s in hits
        ]
        return Answer(question=question, answer=answer_text, grounded=grounded,
                      sources=sources if grounded else [])
