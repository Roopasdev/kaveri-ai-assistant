"""Midstone 6: answer the document questions and the schema questions.

    python run_g06.py            # builds the indexes if needed, writes g06_answers.json
"""
import json
import os

from src import rag, schema_index
from src.llm_client import LLMClient

QUESTIONS = {
    "s1": "How many days of sick leave does an employee get each year?",
    "s2": "What is the daily per diem for international travel?",
    "s3": "How many days a week can an employee in an office role work from home?",
    "s4": "How long is the probation period for new employees?",
    "s5": "Within how many days of delivery can a customer return damaged goods?",
    "s6": "How often must employees change their passwords?",
    "m1": "An employee spends 4 days abroad on a sales trip. What is their total international per diem, "
          "and within how many days must they submit the expense claim?",
    "m2": "How many days of paid leave (annual plus sick) does an employee get in a year, and how many "
          "days a week can they work from home?",
    "m3": "What is the notice period during probation, and how long does probation last?",
    "m4": "How many days of credit does a wholesale customer get, and how many days do they have to "
          "return damaged goods?",
    "u1": "What is the salary of the managing director?",
    "u2": "What is the share price of Kaveri Exports today?",
    "u3": "How many employees work at the Coimbatore head office?",
}

SCHEMA_QUESTIONS = {
    "t1": "Which region earned the most revenue in 2024?",
    "t2": "Which product sold the most units?",
    "t3": "How many customers are there in each segment?",
    "t4": "Which sales head looks after the most customers?",
    "t5": "What was the revenue for each product category in 2025?",
}

if __name__ == "__main__":
    print(f"indexed {rag.build()} chunks")
    client = LLMClient()
    docs = rag.DocAssistant(client)
    out = []
    for qid, q in QUESTIONS.items():
        a = docs.query(q)
        a.pop("retrieved", None)
        print(qid, json.dumps(a)[:200])
        out.append({"id": qid, "question": q, **a})
    tables = []
    for tid, q in SCHEMA_QUESTIONS.items():
        t = schema_index.tables_for(q)
        print(tid, t)
        tables.append({"id": tid, "question": q, "tables": t})
    with open("g06_answers.json", "w", encoding="utf-8") as f:
        json.dump({"register": os.getenv("REGISTER", "TEST-0001"), "model": client.model,
                   "answers": out, "schema_retrieval": tables}, f, indent=2)
    print("wrote g06_answers.json")
