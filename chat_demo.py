"""Midstone 4: a three-turn conversation with streaming and one tool call.

    python chat_demo.py            # writes run_log.jsonl
"""
import os
import sys

from src import db
from src.llm_client import Conversation, LLMClient, RunLog, tool_spec

LOG = "run_log.jsonl"

TOOLS = [tool_spec(
    "get_revenue",
    "Revenue in USD from Kaveri Exports customers in one country in one year.",
    {"country": {"type": "string", "description": "Country name, e.g. Germany"},
     "year": {"type": "integer", "description": "Four-digit year, e.g. 2024"}},
    ["country", "year"])]

TURNS = [
    "Hi! In one sentence, what can you help me with?",
    "What was our revenue from Germany in 2024? Use the tool.",
    "Thanks. Remind me: what was the very first thing I asked you today?",
]

if __name__ == "__main__":
    if os.path.exists(LOG):
        os.remove(LOG)
    client = LLMClient(log=RunLog(LOG))
    chat = Conversation(client, system="You are the Kaveri Exports sales assistant. Be brief. "
                        "For revenue questions always call get_revenue; never guess numbers.",
                        tools=TOOLS, functions={"get_revenue": db.revenue}, conversation_id="c1")
    for question in TURNS:
        print(f"\nYOU: {question}\nAI:  ", end="", flush=True)
        chat.ask(question, stream=True, on_token=lambda t: print(t, end="", flush=True))
        print()
    print(f"\nwrote {LOG} with model {client.model}", file=sys.stderr)
