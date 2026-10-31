"""Policy retriever and a small RAG chain over the midstone 6 FAISS index (midstone 7, P4)."""
from __future__ import annotations

from src import rag
from src.llm_client import LLMClient


def retrieve_docs(question: str, k: int = rag.TOP_K) -> list[dict]:
    return [{"text": c.text, "source": c.source, "page": c.page, "score": s}
            for s, c in rag.retrieve(question, k) if s >= rag.MIN_SCORE]


def rag_chain(client: LLMClient | None = None):
    """question -> retrieve -> grounded prompt -> model -> P4 JSON."""
    client = client or LLMClient()
    with open(rag.PROMPT, encoding="utf-8") as f:
        system = f.read()

    def run(question: str) -> dict:
        docs = retrieve_docs(question)
        if not docs:
            return dict(rag.REFUSAL)
        context = "\n\n".join(f'<extract source="{d["source"]}" page="{d["page"]}">\n{d["text"]}\n</extract>'
                              for d in docs)
        out = client.complete_json([{"role": "system", "content": system},
                                    {"role": "user", "content": f"Document extracts:\n{context}\n\nQuestion: {question}"}])
        read = {d["source"] for d in docs}
        cited = [c for c in dict.fromkeys(str(x) for x in out.get("citations") or []) if c in read]
        if out.get("refused") or not cited:
            return dict(rag.REFUSAL)
        days = out.get("answer_days")
        return {"answer": str(out.get("answer", "")), "citations": cited,
                "answer_days": int(days) if isinstance(days, (int, float)) else None, "refused": False}

    return run
