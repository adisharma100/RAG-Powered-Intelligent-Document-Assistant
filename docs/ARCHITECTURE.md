# Architecture & Design Decisions

## Pipeline

```
INGESTION (offline, once per document)
  PDF/TXT/MD ──► Loader ──► Chunker ──► Embedding model ──► ChromaDB (persisted)
                 loaders.py  chunking.py  embeddings.py      vector_store.py

QUERY (online, per question)
  Question ─► embed ─► top-k similarity search ─► relevance gate ─► prompt + context ─► LLM ─► answer + citations
                       (ChromaDB, cosine)         (MIN_RELEVANCE)    prompts.py         llm.py   pipeline.py
                                                        │
                                                        └─ nothing relevant? return "I don't know" WITHOUT calling the LLM
```

## Why each piece exists

| Component | Choice | Reason |
|---|---|---|
| Loader | `pypdf`, one Document per page | Keeps page numbers so answers can cite `file, page N`. |
| Chunker | `RecursiveCharacterTextSplitter`, 800 chars / 150 overlap | Splits on paragraph → line → sentence → word, so chunks stay coherent. Overlap prevents a fact on a boundary from being cut in half. |
| Chunk IDs | SHA-1 of source + index + text | Re-ingesting the same file is idempotent (no duplicates). |
| Embeddings | `all-MiniLM-L6-v2` (384-dim, local) or OpenAI `text-embedding-3-small` | Local model = free, private, no API key. Vectors are normalised so dot product = cosine similarity. |
| Vector DB | ChromaDB, cosine space, on-disk | Zero-setup, persistent, supports metadata. Swappable for FAISS/Qdrant/pgvector by editing `vector_store.py` only. |
| LLM | LLaMA 3.1 via Ollama (local) or GPT via OpenAI | One factory (`llm.py`); the rest of the code is provider-agnostic. `temperature=0` for deterministic, factual answers. |
| Orchestration | LangChain LCEL: `prompt | llm | StrOutputParser` | Declarative, easy to extend (add reranker, memory, streaming). |

## Reducing hallucinations (four layers)

1. **Retrieval grounding** - the LLM is only shown retrieved chunks, not asked to recall from memory.
2. **Relevance gate** - if the best chunk scores below `MIN_RELEVANCE`, the LLM is never called. This is the strongest guard: an LLM that is not called cannot hallucinate.
3. **Strict prompt** - "use only the context; otherwise reply with the exact refusal sentence; cite passages as [n]".
4. **Citations + snippets** - every answer returns the source file, page, relevance score and text of each chunk used, so a human can verify the claim in seconds.

> These layers *greatly reduce* hallucination but cannot guarantee zero. A model can still misread a correct passage. That is why the project includes an evaluation harness and returns evidence with every answer.

## Tuning guide

| Symptom | Try |
|---|---|
| Answers miss details that exist in the docs | Increase `TOP_K` (4 → 6), lower `MIN_RELEVANCE`, or reduce `CHUNK_SIZE` |
| Answers include irrelevant / mixed-up facts | Decrease `TOP_K`, raise `MIN_RELEVANCE` |
| Chunks cut sentences awkwardly | Increase `CHUNK_OVERLAP` |
| Says "I don't know" too often | Lower `MIN_RELEVANCE` (0.30 → 0.20) - values depend on the embedding model, so calibrate with `eval/questions.json` |

## Possible extensions

* Cross-encoder **re-ranking** of the top 20 hits before taking top 4
* **Hybrid search** (BM25 + vectors) for exact-term queries like error codes
* **Conversation memory** with question rewriting for follow-ups
* **Streaming** responses over Server-Sent Events
* Metadata filtering (per-user / per-document search)
