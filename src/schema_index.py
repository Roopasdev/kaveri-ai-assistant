"""A Chroma index of the database schema (midstone 6, used by the SQL chain in 7).

One entry per table: its name, columns and what they mean, read from
data/raw/data_dictionary.md. `tables_for(question)` returns the few tables a
question needs, so the SQL prompt only carries those.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

from src.rag import embed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DICTIONARY = os.path.join(ROOT, "data", "raw", "data_dictionary.md")
CHROMA_DIR = os.path.join(ROOT, "indexes", "chroma")
TOP_K = 3
MAX_DISTANCE = 1.6      # cosine distance; further than this the table is not about the question


def table_descriptions(path: str = DICTIONARY) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    out = {}
    for m in re.finditer(r"^## (\w+)\s*$(.*?)(?=^## |\Z)", text, re.M | re.S):
        name, body = m.group(1), m.group(2)
        cols = re.findall(r"^\| `(\w+)` \| (.+?) \|$", body, re.M)
        if cols:
            out[name] = f"Table {name}. Columns: " + "; ".join(f"{c}: {d}" for c, d in cols)
    return out


class _Embedder:
    """Chroma calls this to embed; it is the same MiniLM model the RAG part uses."""

    def __call__(self, input):            # noqa: A002 - the name Chroma expects
        return embed(list(input)).tolist()

    def name(self) -> str:
        return "minilm"

    def embed_query(self, input):         # noqa: A002
        return self(input)

    def embed_documents(self, input):     # noqa: A002
        return self(input)

    def is_legacy(self):
        return False


@lru_cache(maxsize=1)
def collection():
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection("schema", embedding_function=_Embedder(),
                                          metadata={"hnsw:space": "cosine"})
    tables = table_descriptions()
    if col.count() != len(tables):
        col.upsert(ids=list(tables), documents=list(tables.values()))
    return col


def tables_for(question: str, k: int = TOP_K) -> list[str]:
    res = collection().query(query_texts=[question], n_results=k)
    names = res["ids"][0]
    dists = res["distances"][0]
    picked = [n for n, d in zip(names, dists) if d <= MAX_DISTANCE] or names[:1]
    # A join path needs its middle table: orders <-> regions goes through customers.
    if "orders" in picked and "regions" in picked and "customers" not in picked:
        picked.append("customers")
    if "order_items" in picked and "customers" in picked and "orders" not in picked:
        picked.append("orders")
    return picked[:4]
