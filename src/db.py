"""The sales database, through SQLAlchemy only (midstone 2 built it).

DATABASE_URL picks the database: sqlite:///kaveri.db on a laptop, a PostgreSQL
URL in production. Nothing else changes.
"""
import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


@lru_cache(maxsize=1)
def engine():
    return create_engine(os.getenv("DATABASE_URL", "sqlite:///kaveri.db"))


def schema_text() -> str:
    """CREATE-style description of every table, for prompts."""
    from sqlalchemy import inspect
    insp = inspect(engine())
    lines = []
    for table in sorted(insp.get_table_names()):
        cols = ", ".join(f"{c['name']} {c['type']}" for c in insp.get_columns(table))
        lines.append(f"{table}({cols})")
    return "\n".join(lines)


def revenue(country: str, year: int) -> float:
    """Revenue from one country's customers in one year (the midstone 4 tool)."""
    sql = text("""SELECT ROUND(COALESCE(SUM(o.total_amount), 0), 2) FROM orders o
                  JOIN customers c ON c.customer_id = o.customer_id
                  WHERE c.country = :country AND o.order_date LIKE :year""")
    with engine().connect() as con:
        return float(con.execute(sql, {"country": country, "year": f"{int(year)}-%"}).scalar())
