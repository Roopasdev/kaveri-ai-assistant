"""Midstone 5: run the twelve NL->SQL questions and the three grounded-answer samples.

    python run_g05.py            # writes g05_results.json
"""
import json
import os

from src.llm_client import LLMClient
from src.sql_assistant import SqlAssistant, _read

QUESTIONS = {
    "q1": "How many products are in the Spices category?",
    "q2": "What was the total revenue from customers in India in 2024?",
    "q3": "Which customer (company name) brought in the most revenue in 2024?",
    "q4": "How many orders came through the Online channel in 2023?",
    "q5": "What is the average list price of Textiles products?",
    "q6": "Which sales head's region earned the most revenue in 2024?",
    "q7": "How many different customers placed at least one order in 2025?",
    "q8": "How many units of Tea & Coffee products were sold in 2024 (not counting cancelled orders)?",
    "q9": "What percentage of 2024 orders were cancelled?",
    "q10": "Which product category earned the most revenue in 2024?",
    "q11": "How many customers are in the Wholesale segment?",
    "q12": "What was the revenue for January to March 2025?",
}

# The three samples from the brief: the same extracts for everyone.
HANDBOOK = {"source": "Handbook_2025.pdf", "text": "Full-time staff receive 24 days of paid annual "
            "leave per year. Part-time staff receive annual leave pro rata to their hours."}
OFFICE = {"source": "Office_Guide.md", "text": "Meeting rooms are booked in the Outlook calendar using "
          "the room's name. Bookings longer than 4 hours need approval from Administration."}
SAMPLES = [
    ("c1", "How many days of annual leave do full-time staff get?", [HANDBOOK, OFFICE]),
    ("c2", "How do I book a meeting room?", [OFFICE, HANDBOOK]),
    ("c3", "What is the company's revenue target for 2026?", [HANDBOOK, OFFICE]),
]


def grounded(client: LLMClient, question: str, extracts: list[dict]) -> dict:
    context = "\n\n".join(f"[{e['source']}]\n{e['text']}" for e in extracts)
    out = client.complete_json([
        {"role": "system", "content": _read("grounded_answer.md")},
        {"role": "user", "content": f"Document extracts:\n{context}\n\nQuestion: {question}"}],
        temperature=0)
    return {"answer": str(out.get("answer", "")), "citations": list(out.get("citations") or []),
            "answer_days": out.get("answer_days"), "refused": bool(out.get("refused"))}


if __name__ == "__main__":
    client = LLMClient()
    sa = SqlAssistant(client)
    sql_answers = []
    for qid, q in QUESTIONS.items():
        try:
            row = sa.ask(q)
        except Exception as e:  # keep going: one bad question must not lose the rest
            row = {"question": q, "sql": None, "answer_value": None, "answer": f"error: {e}", "steps": []}
        print(f"{qid:4} {row['answer_value']!r:40} {row['sql']}")
        sql_answers.append({"id": qid, **row})
    grounded_answers = []
    for cid, q, extracts in SAMPLES:
        g = grounded(client, q, extracts)
        print(cid, g)
        grounded_answers.append({"id": cid, "question": q, **g})
    with open("g05_results.json", "w", encoding="utf-8") as f:
        json.dump({"register": os.getenv("REGISTER", "TEST-0001"), "model": client.model,
                   "sql_answers": sql_answers, "grounded_answers": grounded_answers}, f, indent=2, default=str)
    print("wrote g05_results.json")
