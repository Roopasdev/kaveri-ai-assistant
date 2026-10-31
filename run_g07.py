"""Midstone 7: the SQL conversation, a retriever check, and the three research runs.

    python run_g07.py sql        # writes g07_sql_chat.json
    python run_g07.py rag        # one question through the LangChain retriever chain
    python run_g07.py research   # writes traces/r1.json .. r3.json (+ agent_history.db)
"""
import json
import os
import sys

REGISTER = os.getenv("REGISTER", "TEST-0001")

TURNS = {
    "t1": "What was the total revenue from customers in Germany in 2024?",
    "t2": "And in 2023?",
    "t3": "Which product earned the most from them that year?",
}
TASKS = {
    "r1": "Research vector databases and compare three approaches, citing sources.",
    "r2": "What is retrieval-augmented generation, and who introduced it? Cite sources.",
    "r3": "What are the main export markets for Indian spices? Cite sources.",
}


def sql():
    from src.sql_chain import SqlChain
    chain = SqlChain()
    turns = []
    for tid, q in TURNS.items():
        out = chain.ask(q)
        print(tid, out["answer_value"], "|", out["sql"])
        turns.append({"id": tid, **{k: out[k] for k in ("question", "sql", "answer_value", "answer", "tables")}})
    with open("g07_sql_chat.json", "w", encoding="utf-8") as f:
        json.dump({"register": REGISTER, "turns": turns}, f, indent=2, default=str)


def rag():
    from src.retriever import rag_chain
    print(json.dumps(rag_chain()("How many days of sick leave do I get?"), indent=2))


def research(use_mcp=False, names=None):
    from src.agent import ResearchAgent
    agent = ResearchAgent(use_mcp=use_mcp)
    os.makedirs("traces", exist_ok=True)
    for rid, task in TASKS.items():
        if names and rid not in names:
            continue
        run = agent.run(task, REGISTER)
        print(rid, [s["tool"] for s in run["trace"]], len(run["sources"]), "sources")
        with open(f"traces/{rid}.json", "w", encoding="utf-8") as f:
            json.dump(run, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "sql"
    {"sql": sql, "rag": rag, "research": research}[what]()
