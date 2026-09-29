import pytest
from langchain_core.documents import Document

from rag_assistant.chunking import chunk_documents


def test_chunks_respect_size_and_carry_metadata():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(10))
    chunks = chunk_documents([Document(page_content=text, metadata={"source": "a.txt", "page": None})], 300, 50)
    assert len(chunks) > 3
    assert all(len(c.page_content) <= 300 for c in chunks)
    assert all(c.metadata["source"] == "a.txt" and c.metadata["chunk_id"] for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_deterministic():
    doc = Document(page_content="hello world " * 200, metadata={"source": "x.txt", "page": None})
    a = chunk_documents([doc], 200, 20)
    b = chunk_documents([doc], 200, 20)
    assert [c.metadata["chunk_id"] for c in a] == [c.metadata["chunk_id"] for c in b]


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_documents([Document(page_content="x", metadata={})], 100, 100)
