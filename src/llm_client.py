"""The one LLM client every part of the assistant uses (midstone 4).

Any OpenAI-compatible provider works: OpenRouter, Ollama, Groq, Gemini. The
provider is three settings in .env, never code:

    LLM_BASE_URL   e.g. https://openrouter.ai/api/v1  or  http://localhost:11434/v1
    LLM_API_KEY    the provider's key ("ollama" for Ollama)
    LLM_MODEL      e.g. meta-llama/llama-3.3-70b-instruct:free  or  qwen2.5:7b

What it adds on top of the `openai` package:
  * streaming, with tool-call pieces joined back together;
  * retry with exponential backoff on HTTP 429 (free tiers hit it often),
    honouring Retry-After;
  * a Conversation that keeps the history and runs tool calls;
  * an optional JSONL run log of every event.
"""
from __future__ import annotations

import json
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import openai
from dotenv import load_dotenv

load_dotenv()

MAX_RETRIES = 5
BASE_WAIT_S = 1.0
MAX_WAIT_S = 30.0


@dataclass
class Reply:
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)   # [{id, name, arguments(dict)}]
    model: str = ""
    streamed: bool = False
    chunks: int = 0
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


class RunLog:
    """Appends one JSON object per event to a .jsonl file (or does nothing)."""

    def __init__(self, path: str | None = None):
        self.path = path

    def write(self, **event) -> None:
        if not self.path:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


class LLMClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, log: RunLog | None = None,
                 max_retries: int = MAX_RETRIES, timeout: float | None = None):
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
        self.model = model or os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
        key = api_key or os.getenv("LLM_API_KEY") or "not-set"
        # A 7B model on a laptop CPU can take minutes for a long prompt.
        timeout = timeout or float(os.getenv("LLM_TIMEOUT", "300"))
        self._client = openai.OpenAI(base_url=self.base_url, api_key=key, timeout=timeout,
                                     max_retries=0)  # we retry ourselves, visibly
        self.create = self._client.chat.completions.create   # replaceable in tests
        self.log = log or RunLog()
        self.max_retries = max_retries

    # ── retry ──

    def _wait_for(self, err: openai.RateLimitError, attempt: int) -> float:
        retry_after = None
        try:
            retry_after = float(err.response.headers.get("retry-after"))
        except (AttributeError, TypeError, ValueError):
            pass
        if retry_after is not None:
            return min(retry_after, MAX_WAIT_S)
        return min(BASE_WAIT_S * 2 ** attempt + random.uniform(0, 0.5), MAX_WAIT_S)

    def _call(self, **kwargs):
        """One API call, retried on HTTP 429 with a growing wait. Any other error
        is raised at once: retrying a bad request or a wrong key only hides it."""
        for attempt in range(self.max_retries + 1):
            try:
                return self.create(model=self.model, **kwargs)
            except openai.RateLimitError as err:
                if attempt == self.max_retries:
                    self.log.write(event="retry_gave_up", status=429, attempts=attempt + 1)
                    raise
                wait = self._wait_for(err, attempt)
                self.log.write(event="retry", status=429, attempt=attempt + 1, wait_s=round(wait, 2))
                time.sleep(wait)

    # ── one chat call ──

    def chat(self, messages: list[dict], tools: list[dict] | None = None, stream: bool = False,
             on_token: Callable[[str], None] | None = None, **kwargs) -> Reply:
        started = time.perf_counter()
        extra = {"tools": tools} if tools else {}
        if not stream:
            resp = self._call(messages=messages, **extra, **kwargs)
            msg = resp.choices[0].message
            reply = Reply(content=msg.content or "", model=resp.model or self.model,
                          tool_calls=[_tool_call(tc.id, tc.function.name, tc.function.arguments)
                                      for tc in (msg.tool_calls or [])])
            if resp.usage:
                reply.tokens_in, reply.tokens_out = resp.usage.prompt_tokens, resp.usage.completion_tokens
        else:
            reply = self._stream(messages, extra, on_token, kwargs)
        reply.latency_ms = int((time.perf_counter() - started) * 1000)
        return reply

    def _stream(self, messages, extra, on_token, kwargs) -> Reply:
        reply = Reply(streamed=True, model=self.model)
        pieces: dict[int, dict] = {}   # tool calls arrive in pieces, keyed by index
        text = []
        resp = self._call(messages=messages, stream=True, stream_options={"include_usage": True},
                          **extra, **kwargs)
        for chunk in resp:
            if chunk.usage:
                reply.tokens_in, reply.tokens_out = chunk.usage.prompt_tokens, chunk.usage.completion_tokens
            if chunk.model:
                reply.model = chunk.model
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                reply.chunks += 1
                text.append(delta.content)
                if on_token:
                    on_token(delta.content)
            for tc in delta.tool_calls or []:
                slot = pieces.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
        reply.content = "".join(text)
        reply.tool_calls = [_tool_call(p["id"], p["name"], p["arguments"])
                            for _i, p in sorted(pieces.items())]
        return reply

    def complete_json(self, messages: list[dict], **kwargs) -> dict:
        """A non-streamed call whose reply must be one JSON object."""
        reply = self.chat(messages, response_format={"type": "json_object"}, **kwargs)
        return parse_json(reply.content)


def _tool_call(call_id: str, name: str, arguments: str | dict) -> dict:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            arguments = {"_raw": arguments}
    return {"id": call_id or f"call_{uuid.uuid4().hex[:8]}", "name": name, "arguments": arguments}


def parse_json(text: str) -> dict:
    """The first JSON object in a reply, also when a model wraps it in ```json fences."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON object in the reply: {text[:200]!r}")
    return json.loads(text[start:end + 1])


class Conversation:
    """Chat history plus tool calling. One instance per conversation."""

    def __init__(self, client: LLMClient, system: str | None = None,
                 tools: list[dict] | None = None,
                 functions: dict[str, Callable[..., Any]] | None = None,
                 conversation_id: str | None = None, max_tool_rounds: int = 3):
        self.client = client
        self.history: list[dict] = [{"role": "system", "content": system}] if system else []
        self.tools = tools
        self.functions = functions or {}
        self.id = conversation_id or uuid.uuid4().hex[:8]
        self.turn = 0
        self.max_tool_rounds = max_tool_rounds

    def ask(self, text: str, stream: bool = True, on_token: Callable[[str], None] | None = None) -> str:
        self.turn += 1
        log = self.client.log
        self.history.append({"role": "user", "content": text})
        log.write(conversation_id=self.id, turn=self.turn, role="user", content=text)

        for _round in range(self.max_tool_rounds + 1):
            reply = self.client.chat(self.history, tools=self.tools, stream=stream, on_token=on_token)
            event = dict(conversation_id=self.id, turn=self.turn, role="assistant", content=reply.content,
                         streamed=reply.streamed, chunks=reply.chunks, model=reply.model,
                         latency_ms=reply.latency_ms)
            if not reply.tool_calls:
                self.history.append({"role": "assistant", "content": reply.content})
                log.write(**event)
                return reply.content

            log.write(**event, tool_calls=reply.tool_calls)
            self.history.append({"role": "assistant", "content": reply.content or None, "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                for tc in reply.tool_calls]})
            for tc in reply.tool_calls:
                result = self._run_tool(tc)
                self.history.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                log.write(conversation_id=self.id, turn=self.turn, role="tool", tool_call_id=tc["id"],
                          name=tc["name"], content=result)
        raise RuntimeError("The model kept calling tools without answering.")

    def _run_tool(self, tc: dict) -> str:
        fn = self.functions.get(tc["name"])
        if fn is None:
            return json.dumps({"error": f"unknown tool {tc['name']}"})
        try:
            result = fn(**tc["arguments"])
        except Exception as e:  # the model sees the error and can correct itself
            return json.dumps({"error": f"{type(e).__name__}: {e}"})
        return result if isinstance(result, str) else json.dumps(result, default=str)


def tool_spec(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """An OpenAI-style function tool definition."""
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}
