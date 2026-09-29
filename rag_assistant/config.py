"""Central configuration. Every value can be overridden through environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Settings:
    # LLM
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "ollama"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3.1"))
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))

    # Embeddings
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "huggingface"))
    hf_embedding_model: str = field(
        default_factory=lambda: _env("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    openai_embedding_model: str = field(
        default_factory=lambda: _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # Chunking / retrieval
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "150")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "4")))
    min_relevance: float = field(default_factory=lambda: float(_env("MIN_RELEVANCE", "0.30")))

    # Storage
    chroma_dir: str = field(default_factory=lambda: _env("CHROMA_DIR", "./chroma_db"))
    collection_name: str = field(default_factory=lambda: _env("COLLECTION_NAME", "documents"))
