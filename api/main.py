"""FastAPI service exposing the assistant over REST (API integration).

  uvicorn api.main:app --reload      ->  http://localhost:8000/docs
"""
from __future__ import annotations

import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag_assistant import RAGAssistant
from rag_assistant.loaders import SUPPORTED_EXTENSIONS

app = FastAPI(title="RAG Document Assistant", version="1.0.0")


@lru_cache(maxsize=1)
def get_assistant() -> RAGAssistant:
    return RAGAssistant()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What is the refund policy?"])
    top_k: Optional[int] = Field(None, ge=1, le=20)


class SourceOut(BaseModel):
    source: str
    page: Optional[int]
    chunk_id: str
    score: float
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    grounded: bool
    sources: List[SourceOut]


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": get_assistant().count()}


@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    assistant = get_assistant()
    total, ingested = 0, []
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            name = Path(f.filename or "upload").name
            if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise HTTPException(400, f"Unsupported file type: {name}")
            dest = Path(tmp) / name
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            total += assistant.ingest(dest)
            ingested.append(name)
    return {"files": ingested, "chunks_added": total, "chunks_total": assistant.count()}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    assistant = get_assistant()
    if assistant.count() == 0:
        raise HTTPException(409, "No documents indexed yet. POST files to /ingest first.")
    return assistant.ask(req.question, req.top_k).to_dict()


@app.delete("/index")
def clear_index():
    get_assistant().reset()
    return {"status": "cleared"}
