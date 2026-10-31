"""The LangChain chat model, pointed at the same free provider as src/llm_client.py."""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def chat_model(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("LLM_API_KEY") or "not-set",
        model=os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
        temperature=temperature,
        timeout=float(os.getenv("LLM_TIMEOUT", "300")),
        max_retries=4,          # LangChain's own backoff, including on HTTP 429
    )
