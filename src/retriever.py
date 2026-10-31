"""The midstone 6 FAISS index as a LangChain retriever, and a small RAG chain on it (midstone 7, P4)."""
from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from src import rag
from src.lc import chat_model
from src.llm_client import parse_json


class PolicyRetriever(BaseRetriever):
    k: int = rag.TOP_K
    min_score: float = rag.MIN_SCORE

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return [Document(page_content=c.text, metadata={"source": c.source, "page": c.page, "score": s})
                for s, c in rag.retrieve(query, self.k) if s >= self.min_score]


def _format(docs: list[Document]) -> str:
    return "\n\n".join(f"<extract source=\"{d.metadata['source']}\" page=\"{d.metadata['page']}\">\n"
                       f"{d.page_content}\n</extract>" for d in docs)


def rag_chain(llm=None):
    """question -> retriever -> grounded prompt -> model -> P4 JSON."""
    with open(rag.PROMPT, encoding="utf-8") as f:
        system = f.read()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system}"),
        ("human", "Document extracts:\n{context}\n\nQuestion: {question}")])
    model = prompt | (llm or chat_model()) | StrOutputParser()
    retriever = PolicyRetriever()

    def run(question: str) -> dict:
        docs = retriever.invoke(question)
        if not docs:
            return dict(rag.REFUSAL)
        out = parse_json(model.invoke({"system": system, "context": _format(docs), "question": question}))
        read = {d.metadata["source"] for d in docs}
        cited = [c for c in dict.fromkeys(str(x) for x in out.get("citations") or []) if c in read]
        if out.get("refused") or not cited:
            return dict(rag.REFUSAL)
        days = out.get("answer_days")
        return {"answer": str(out.get("answer", "")), "citations": cited,
                "answer_days": int(days) if isinstance(days, (int, float)) else None, "refused": False}

    return run
