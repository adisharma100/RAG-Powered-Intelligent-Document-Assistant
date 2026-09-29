"""Offline test doubles: a hashing bag-of-words embedder and a fake LLM. No network, no model downloads."""
import hashlib
import math
import re

import pytest
from langchain_core.embeddings import Embeddings


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-words embedding: same words -> similar vectors (good enough to test the pipeline)."""

    STOP = {"the", "a", "an", "of", "is", "are", "to", "in", "and", "for", "on", "how", "what", "does", "do", "per"}

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _embed(self, text: str):
        vec = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            if tok in self.STOP:
                continue
            vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture
def embeddings():
    return HashEmbeddings()
