"""The retry on HTTP 429, tested without a real provider."""
try:  # openai 3.x ships httpx2; 1.x and 2.x use httpx
    import httpx2 as httpx
except ImportError:
    import httpx
import openai
import pytest

from src import llm_client
from src.llm_client import LLMClient


def _rate_limited():
    req = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    resp = httpx.Response(429, request=req, headers={"retry-after": "0"})
    return openai.RateLimitError("rate limited", response=resp, body=None)


def test_retries_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise _rate_limited()
        return "ok"

    client = LLMClient(base_url="https://example.invalid/v1", api_key="x", model="m")
    client.create = fake_create
    assert client._call(messages=[]) == "ok"
    assert len(calls) == 3


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    client = LLMClient(base_url="https://example.invalid/v1", api_key="x", model="m", max_retries=2)
    client.create = lambda **kw: (_ for _ in ()).throw(_rate_limited())
    with pytest.raises(openai.RateLimitError):
        client._call(messages=[])


def test_other_errors_are_not_retried():
    client = LLMClient(base_url="https://example.invalid/v1", api_key="x", model="m")
    calls = []

    def fake_create(**kwargs):
        calls.append(1)
        raise ValueError("bad request")

    client.create = fake_create
    with pytest.raises(ValueError):
        client._call(messages=[])
    assert len(calls) == 1
