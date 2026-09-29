"""Step 3 - Vector embeddings. Maps text to dense vectors where similar meaning = nearby vectors."""
from __future__ import annotations

from langchain_core.embeddings import Embeddings

from .config import Settings


def get_embeddings(settings: Settings) -> Embeddings:
    provider = settings.embedding_provider.lower()
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=settings.hf_embedding_model,
            encode_kwargs={"normalize_embeddings": True},  # unit vectors -> cosine similarity
        )
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.openai_embedding_model)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER '{settings.embedding_provider}'")
