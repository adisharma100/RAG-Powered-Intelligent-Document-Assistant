from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import api.main as api_main
from rag_assistant import RAGAssistant, Settings

SAMPLE = Path(__file__).parent.parent / "data" / "sample_docs"


def test_api_flow(tmp_path, embeddings, monkeypatch):
    settings = Settings(chroma_dir=str(tmp_path / "db"), chunk_size=400, chunk_overlap=60, min_relevance=0.15)
    assistant = RAGAssistant(settings, embeddings, FakeListChatModel(responses=["24 days [1]."]))
    monkeypatch.setattr(api_main, "get_assistant", lambda: assistant)
    client = TestClient(api_main.app)

    assert client.post("/query", json={"question": "vacation days?"}).status_code == 409  # empty index

    with open(SAMPLE / "employee_handbook.md", "rb") as f:
        r = client.post("/ingest", files=[("files", ("employee_handbook.md", f, "text/markdown"))])
    assert r.status_code == 200 and r.json()["chunks_added"] > 0

    r = client.post("/query", json={"question": "How many vacation days do employees get?"})
    body = r.json()
    assert r.status_code == 200 and body["grounded"] and body["sources"][0]["source"] == "employee_handbook.md"
    assert client.post("/ingest", files=[("files", ("x.exe", b"hi", "application/octet-stream"))]).status_code == 400
