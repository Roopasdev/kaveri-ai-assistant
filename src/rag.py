"""Answers from the 12 policy documents, with citations (midstone 6, capstone P4).

ingest -> chunk -> HuggingFace embeddings -> FAISS -> grounded answer.

    python -m src.rag build          # (re)build indexes/faiss.index from data/docs
    python -m src.rag ask "How many days of sick leave do I get?"
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from src.llm_client import LLMClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(ROOT, "data", "docs")
INDEX_DIR = os.path.join(ROOT, "indexes")
PROMPT = os.path.join(ROOT, "prompts", "grounded_answer.md")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_CHARS = 600
OVERLAP = 100
TOP_K = 6
MIN_SCORE = 0.15          # below this nothing retrieved is about the question


@dataclass
class Chunk:
    source: str
    page: int
    text: str


# ── ingest and chunk ──

def read_documents(folder: str = DOCS_DIR) -> list[tuple[str, int, str]]:
    """(file name, page, text) for every page of every document."""
    from pypdf import PdfReader
    pages = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if name.lower().endswith(".pdf"):
            for i, page in enumerate(PdfReader(path).pages, start=1):
                pages.append((name, i, page.extract_text() or ""))
        elif name.lower().endswith((".md", ".txt")):
            with open(path, encoding="utf-8") as f:
                pages.append((name, 1, f.read()))
    return pages


def chunk(pages) -> list[Chunk]:
    """Split on paragraphs, then pack paragraphs into ~CHUNK_CHARS pieces with a
    little overlap, so a fact is never cut off from its heading."""
    chunks = []
    for source, page, text in pages:
        title = source.rsplit(".", 1)[0].replace("_", " ")
        paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=#+ |\d+\. [A-Z])", text) if p.strip()]
        buf = ""
        for p in paras:
            if buf and len(buf) + len(p) > CHUNK_CHARS:
                chunks.append(Chunk(source, page, f"[{title}] {buf.strip()}"))
                buf = buf[-OVERLAP:]
            buf += " " + p
        if buf.strip():
            chunks.append(Chunk(source, page, f"[{title}] {buf.strip()}"))
    return chunks


# ── embed and index ──

@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    return np.asarray(embedder().encode(texts, normalize_embeddings=True), dtype="float32")


def build(folder: str = DOCS_DIR) -> int:
    import faiss
    chunks = chunk(read_documents(folder))
    vectors = embed([c.text for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])     # inner product of unit vectors = cosine
    index.add(vectors)
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f)
    load.cache_clear()
    return len(chunks)


@lru_cache(maxsize=1)
def load():
    import faiss
    path = os.path.join(INDEX_DIR, "faiss.index")
    if not os.path.exists(path):
        build()
    with open(os.path.join(INDEX_DIR, "chunks.json"), encoding="utf-8") as f:
        chunks = [Chunk(**c) for c in json.load(f)]
    return faiss.read_index(path), chunks


def retrieve(question: str, k: int = TOP_K) -> list[tuple[float, Chunk]]:
    index, chunks = load()
    scores, ids = index.search(embed([question]), k)
    return [(float(s), chunks[i]) for s, i in zip(scores[0], ids[0]) if i >= 0]


def sources() -> set[str]:
    return {c.source for c in load()[1]}


# ── grounded answer ──

REFUSAL = {"answer": "That information is not in the Kaveri Exports documents.",
           "citations": [], "answer_days": None, "refused": True}


class DocAssistant:
    def __init__(self, client: LLMClient | None = None):
        self.client = client or LLMClient()
        with open(PROMPT, encoding="utf-8") as f:
            self.system = f.read()

    def query(self, question: str, k: int = TOP_K) -> dict:
        hits = retrieve(question, k)
        if not hits or hits[0][0] < MIN_SCORE:
            return dict(REFUSAL, retrieved=[])
        context = "\n\n".join(f"[{c.source}, page {c.page}]\n{c.text}" for _s, c in hits)
        out = self.client.complete_json([
            {"role": "system", "content": self.system},
            {"role": "user", "content": f"Document extracts:\n{context}\n\nQuestion: {question}"}],
            temperature=0)
        retrieved = {c.source for _s, c in hits}
        cited = [os.path.basename(str(c)) for c in out.get("citations") or []]
        cited = [c for c in dict.fromkeys(cited) if c in retrieved]     # never cite what we did not read
        refused = bool(out.get("refused")) or not cited
        if refused:
            return dict(REFUSAL, retrieved=sorted(retrieved))
        days = out.get("answer_days")
        try:
            days = int(days) if days is not None else None
        except (TypeError, ValueError):
            days = None
        return {"answer": str(out.get("answer", "")), "citations": cited, "answer_days": days,
                "refused": False, "retrieved": sorted(retrieved)}


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        print(f"indexed {build()} chunks")
    elif sys.argv[1:2] == ["ask"]:
        print(json.dumps(DocAssistant().query(" ".join(sys.argv[2:])), indent=2))
