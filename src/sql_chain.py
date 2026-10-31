"""NL -> SQL as a LangChain chain that remembers the conversation (midstone 7, P3).

prompt (schema of the tables Chroma picks + definitions + history) -> model ->
JSON {"sql"} -> safety check -> run read-only -> answer. The history holds each
earlier question and the SQL that answered it, so "And in 2023?" is read as
"the same question, for 2023".
"""
from __future__ import annotations

import json

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from sqlalchemy import inspect

from src import db, schema_index
from src.lc import chat_model
from src.llm_client import parse_json
from src.security import UnsafeSQL
from src.sql_assistant import _plain, _read, run_select

FOLLOW_UP = ("Earlier questions in this conversation are above, each with the SQL that answered "
             "it. A follow-up such as 'And in 2023?' or 'from them' refers to them: keep the same "
             "filters and change only what the new question changes.")


def schema_for(tables: list[str]) -> str:
    insp = inspect(db.engine())
    return "\n".join(f"{t}({', '.join(c['name'] + ' ' + str(c['type']) for c in insp.get_columns(t))})"
                     for t in tables)


def _few_shot() -> list:
    msgs = []
    for ex in json.loads(_read("nl2sql_few_shot.json")):
        msgs += [HumanMessage(ex["question"]), AIMessage(json.dumps({"sql": ex["sql"]}))]
    return msgs


class SqlChain:
    def __init__(self, llm=None):
        self.llm = llm or chat_model()
        self.history = InMemoryChatMessageHistory()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "{system}"),
            MessagesPlaceholder("examples"),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _tables(self, question: str) -> list[str]:
        tables = set(schema_index.tables_for(question))
        # A follow-up ("And in 2023?") carries no table words: keep the tables of the turns before.
        for m in self.history.messages:
            if isinstance(m, HumanMessage):
                tables |= set(schema_index.tables_for(m.content))
        return sorted(tables)

    def ask(self, question: str) -> dict:
        tables = self._tables(question)
        # The schema text is passed as a variable, never parsed as a template: its braces stay literal.
        system = _read("nl2sql_system.md").replace("{schema}", schema_for(tables)) + "\n\n" + FOLLOW_UP
        inputs = {"system": system, "examples": _few_shot(), "history": self.history.messages,
                  "question": question}
        sql = str(parse_json(self.chain.invoke(inputs)).get("sql", "")).strip().rstrip(";")
        try:
            cols, rows = run_select(sql)
        except UnsafeSQL:
            raise
        except Exception as e:          # one repair, with the database's own error
            error = str(getattr(e, "orig", e)).split("\n")[0]
            inputs["question"] = (f"{question}\n\nYour query was: {sql}\nIt failed with: {error}\n"
                                  "Join through the tables the columns belong to, and reply with the "
                                  'corrected query as JSON: {"sql": "..."}')
            sql = str(parse_json(self.chain.invoke(inputs)).get("sql", "")).strip().rstrip(";")
            cols, rows = run_select(sql)
        rows = [[_plain(v) for v in r] for r in rows]
        value = rows[0][0] if len(rows) == 1 and len(rows[0]) == 1 else rows
        self.history.add_messages([HumanMessage(question), AIMessage(json.dumps({"sql": sql}))])
        return {"question": question, "sql": sql, "tables": tables, "answer_value": value,
                "answer": f"{value}" if not isinstance(value, list) else f"{len(rows)} rows"}
