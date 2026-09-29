"""Step 5 - LLM factory. LLaMA (local, via Ollama) or GPT (via OpenAI) behind one interface."""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from .config import Settings


def get_llm(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider.lower()
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url, temperature=0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.openai_model, temperature=0)
    raise ValueError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'")
