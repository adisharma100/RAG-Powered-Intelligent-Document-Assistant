"""Step 2 - Text chunking.

Large documents cannot be embedded (or fed to an LLM) as one block, and huge chunks
dilute meaning. We split recursively on paragraph -> line -> sentence -> word boundaries
so chunks stay semantically coherent, with overlap so facts on a boundary are not lost.
"""
from __future__ import annotations

import hashlib
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 150) -> List[Document]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    per_source_counter: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        idx = per_source_counter.get(source, 0)
        per_source_counter[source] = idx + 1
        # Deterministic ID -> re-ingesting the same file overwrites instead of duplicating.
        digest = hashlib.sha1(f"{source}|{idx}|{chunk.page_content}".encode()).hexdigest()[:16]
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_id"] = digest
        # Chroma metadata cannot hold None
        if chunk.metadata.get("page") is None:
            chunk.metadata["page"] = -1
    return chunks
