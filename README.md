# Kaveri Exports AI assistant

## About

An AI assistant for the sales heads of Kaveri Exports, a Coimbatore exporter. It answers three
kinds of question: **numbers from the sales database** (it writes SQL, runs it read-only and
shows the SQL), **answers from the 12 company policy documents** (with the document it used, or
a refusal when the answer is not there), and **research about the outside world** (a short
report from web search and Wikipedia, with a link for every claim). It never makes anything up,
and nobody can change or delete data through it.

## Setup

1. `python -m venv .venv` and activate it.
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in your own values. `.env` is never committed.
4. `python load_db.py` builds the database from `data/raw/`.

## Project layout

| Folder | What is in it |
|---|---|
| `src/` | The application code: the LLM client, the SQL assistant, RAG, the agent and the MCP server |
| `prompts/` | System prompts and few-shot examples, kept out of the code |
| `eval/` | The test set, the evaluation runner, results and the report |
| `docs/` | Architecture notes and diagrams |
| `tests/` | Automated tests |
| `data/` | `raw/`: the five sales CSVs and the data dictionary; later `docs/`: the policy documents |
