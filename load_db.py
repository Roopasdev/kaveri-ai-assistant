"""Midstone 2: load the Kaveri Exports CSVs into a database and answer ten questions.

    python load_db.py                      # load data/raw/*.csv, write answers.json

The database address comes from DATABASE_URL (default sqlite:///kaveri.db), so the
same script loads PostgreSQL later with nothing changed but that URL.
"""
import json
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///kaveri.db")
RAW_DIR = os.getenv("RAW_DIR", "data/raw")
REGISTER = os.getenv("REGISTER", "TEST-0001")
TABLES = ["regions", "customers", "products", "orders", "order_items"]


def clean(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """The business brief's three rules."""
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()
    if name == "orders":
        # Two formats, each read with its own pattern: a single "dayfirst" parse
        # would also flip the ISO dates (2024-12-05 read as 12 May).
        iso = pd.to_datetime(df["order_date"], format="%Y-%m-%d", errors="coerce")
        dmy = pd.to_datetime(df["order_date"], format="%d/%m/%Y", errors="coerce")
        df["order_date"] = iso.fillna(dmy).dt.strftime("%Y-%m-%d")
    return df.drop_duplicates()


def load(engine) -> dict:
    counts = {}
    for name in TABLES:
        df = clean(name, pd.read_csv(os.path.join(RAW_DIR, f"{name}.csv"), dtype={"order_date": str}))
        df.to_sql(name, engine, if_exists="replace", index=False)
        counts[name] = len(df)
        print(f"{name:12} {len(df):6} rows")
    return counts


QUESTIONS = {
    "q1": "SELECT COUNT(*) FROM customers",
    "q2": "SELECT ROUND(SUM(total_amount), 2) FROM orders WHERE order_date LIKE '2024-%'",
    "q3": """SELECT c.country FROM orders o JOIN customers c ON c.customer_id = o.customer_id
             GROUP BY c.country ORDER BY COUNT(*) DESC LIMIT 1""",
    "q4": "SELECT ROUND(SUM(total_amount), 2) FROM orders WHERE order_date LIKE '2023-%'",
    "q5": "SELECT COUNT(*) FROM orders WHERE status = 'cancelled'",
    "q6": """SELECT p.product_name FROM order_items i
             JOIN orders o ON o.order_id = i.order_id
             JOIN products p ON p.product_id = i.product_id
             WHERE o.status <> 'cancelled' AND o.order_date LIKE '2024-%'
             GROUP BY p.product_name ORDER BY SUM(i.line_total) DESC LIMIT 1""",
    "q7": """SELECT ROUND(AVG(total_amount), 2) FROM orders
             WHERE status <> 'cancelled' AND order_date LIKE '2024-%'""",
    "q8": """SELECT r.region_name FROM orders o
             JOIN customers c ON c.customer_id = o.customer_id
             JOIN regions r ON r.region_id = c.region_id
             WHERE o.order_date LIKE '2024-%'
             GROUP BY r.region_name ORDER BY SUM(o.total_amount) DESC LIMIT 1""",
    "q9": """SELECT substr(order_date, 1, 7) FROM orders WHERE order_date LIKE '2024-%'
             GROUP BY 1 ORDER BY SUM(total_amount) DESC LIMIT 1""",
    "q10": """SELECT COUNT(*) FROM customers WHERE customer_id NOT IN
              (SELECT customer_id FROM orders WHERE order_date LIKE '2025-%')""",
}


def answer(engine) -> dict:
    out = {}
    with engine.connect() as con:
        for qid, sql in QUESTIONS.items():
            out[qid] = con.execute(text(sql)).scalar()
            print(f"{qid:4} {out[qid]}")
    return out


if __name__ == "__main__":
    engine = create_engine(DATABASE_URL)
    counts = load(engine)
    answers = answer(engine)
    with open("answers.json", "w", encoding="utf-8") as f:
        json.dump({"register": REGISTER, "row_counts": counts, "answers": answers}, f, indent=2)
    print("wrote answers.json")
