"""Step 1 - Document loading. Turns files on disk into LangChain `Document` objects."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_pdf(path: Path) -> List[Document]:
    """One Document per page, so page numbers survive into the citations."""
    reader = PdfReader(str(path))
    docs = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": path.name, "page": page_no}))
    return docs


def load_text(path: Path) -> List[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [Document(page_content=text, metadata={"source": path.name, "page": None})] if text else []


def load_file(path: str | Path) -> List[Document]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    return load_pdf(path) if ext == ".pdf" else load_text(path)


def iter_supported_files(path: str | Path) -> Iterable[Path]:
    path = Path(path)
    if path.is_file():
        yield path
    else:
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield p


def load_path(path: str | Path) -> List[Document]:
    """Load a single file or every supported file inside a directory."""
    docs: List[Document] = []
    for file in iter_supported_files(path):
        docs.extend(load_file(file))
    return docs
