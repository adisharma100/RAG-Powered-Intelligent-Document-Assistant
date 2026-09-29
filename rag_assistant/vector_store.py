"""Step 4 - Vector database (ChromaDB, persisted on disk)."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from .config import Settings


def build_vector_store(settings: Settings, embeddings: Embeddings) -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_dir,
        # cosine distance; relevance score = 1 - distance, so it lives in ~[0, 1]
        collection_metadata={"hnsw:space": "cosine"},
        relevance_score_fn=lambda distance: 1.0 - distance,
    )
