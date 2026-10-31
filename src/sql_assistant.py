"""Question -> SQL -> answer, as a two-step prompt chain (midstone 5).

Step 1 (write_sql) asks the model for one SELECT, in JSON. The SQL is checked
and run read-only. Step 2 (answer) turns the rows into a sentence. When the
result is a single value, `answer_value` is taken from the database itself, so
the number can never be one the model made up.
"""
from __future__ import annotations

import json
import os
from decimal import Decimal
from functools import lru_cache

from sqlalchemy import text

from src import db
from src.llm_client import LLMClient, parse_json
from src.security import UnsafeSQL, check_sql

PROMPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")
MAX_ROWS = 50


def _read(name: str) -> str:
    with open(os.path.join(PROMPTS, name), encoding="utf-8") as f:
        return f.read()


@lru_cache(maxsize=1)
def system_prompt() -> str:
    return _read("nl2sql_system.md").replace("{schema}", db.schema_text())


def few_shot_messages() -> list[dict]:
    """Each example as a user question and the assistant's JSON reply."""
    msgs = []
    for ex in json.loads(_read("nl2sql_few_shot.json")):
        msgs.append({"role": "user", "content": ex["question"]})
        msgs.append({"role": "assistant", "content": json.dumps({"sql": ex["sql"]})})
    return msgs


def run_select(sql: str) -> tuple[list[str], list[list]]:
    """Run a checked SELECT and return (columns, rows)."""
    check_sql(sql)
    with db.engine().connect() as con:
        result = con.execute(text(sql))
        cols = list(result.keys())
        rows = [list(r) for r in result.fetchmany(MAX_ROWS)]
    return cols, rows


def _plain(value):
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, float):
        return round(value, 2)
    return value


class SqlAssistant:
    def __init__(self, client: LLMClient | None = None):
        self.client = client or LLMClient()

    def write_sql(self, question: str, history: list[dict] | None = None) -> str:
        messages = [{"role": "system", "content": system_prompt()}, *few_shot_messages(),
                    *(history or []), {"role": "user", "content": question}]
        out = self.client.complete_json(messages, temperature=0)
        return str(out.get("sql", "")).strip().rstrip(";")

    def answer(self, question: str, sql: str, cols: list[str], rows: list[list]) -> dict:
        messages = [{"role": "system", "content": _read("answer_system.md")},
                    {"role": "user", "content": json.dumps(
                        {"question": question, "sql": sql, "columns": cols, "rows": rows}, default=str)}]
        return self.client.complete_json(messages, temperature=0)

    def repair_sql(self, question: str, sql: str, error: str, history: list[dict] | None = None) -> str:
        """One more try, showing the model the database's own error message."""
        messages = [{"role": "system", "content": system_prompt()}, *few_shot_messages(),
                    *(history or []), {"role": "user", "content": question},
                    {"role": "assistant", "content": json.dumps({"sql": sql})},
                    {"role": "user", "content": f"That query failed with: {error}\n"
                     "Check the table and column names against the schema and reply with the "
                     'corrected query as JSON: {"sql": "..."}'}]
        out = self.client.complete_json(messages, temperature=0)
        return str(out.get("sql", "")).strip().rstrip(";")

    def ask(self, question: str, history: list[dict] | None = None) -> dict:
        steps = []
        sql = self.write_sql(question, history)
        steps.append({"step": "write_sql", "output": {"sql": sql}})
        try:
            cols, rows = run_select(sql)
        except UnsafeSQL:
            raise                       # never "repair" a blocked statement into running
        except Exception as e:
            error = str(getattr(e, "orig", e)).split("\n")[0]
            sql = self.repair_sql(question, sql, error, history)
            steps.append({"step": "repair_sql", "error": error, "output": {"sql": sql}})
            cols, rows = run_select(sql)
        rows = [[_plain(v) for v in r] for r in rows]
        steps.append({"step": "run_sql", "columns": cols, "rows": rows[:10]})
        said = self.answer(question, sql, cols, rows)
        steps.append({"step": "answer", "output": said})
        if len(rows) == 1 and len(rows[0]) == 1:
            value = rows[0][0]          # from the database, never from the model
        else:
            value = _plain(said.get("answer_value"))
        return {"question": question, "sql": sql, "answer_value": value,
                "answer": str(said.get("answer", "")), "steps": steps}
