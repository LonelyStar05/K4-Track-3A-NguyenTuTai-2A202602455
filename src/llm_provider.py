from __future__ import annotations

"""Shared OpenAI/Gemini provider configuration for generation and RAGAS."""

import time

from langchain_core.embeddings import Embeddings

from config import (
    EVALUATOR_EMBEDDING_MODEL,
    EVALUATOR_MODEL,
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
)

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def has_api_key() -> bool:
    return bool(_provider_key())


def provider_name() -> str:
    return LLM_PROVIDER if has_api_key() else "offline"


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 500,
    temperature: float = 0,
) -> str:
    """Call the configured provider through the OpenAI-compatible chat API."""
    key = _provider_key()
    if not key:
        return ""
    from openai import OpenAI

    kwargs = {"api_key": key}
    if LLM_PROVIDER == "gemini":
        kwargs["base_url"] = GEMINI_OPENAI_BASE_URL
    client = OpenAI(max_retries=3, **kwargs)
    effective_max_tokens = max(max_tokens, 1024) if LLM_PROVIDER == "gemini" else max_tokens
    last_error = None
    for attempt, delay in enumerate((0, 5, 15, 30, 60)):
        if delay:
            time.sleep(delay)
        try:
            response = client.chat.completions.create(
                model=_chat_model(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=effective_max_tokens,
                temperature=temperature,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            last_error = exc
            if attempt == 4:
                raise
    raise last_error


def get_ragas_models():
    """Return LangChain LLM/embeddings configured for RAGAS."""
    key = _provider_key()
    if not key:
        return None, None
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    common = {"api_key": key}
    if LLM_PROVIDER == "gemini":
        common["base_url"] = GEMINI_OPENAI_BASE_URL
    llm = ChatOpenAI(model=_chat_model(), temperature=0, max_retries=12, **common)
    # Gemini endpoint không nhận token-id arrays của tiktoken → gửi text thô.
    embeddings = OpenAIEmbeddings(model=_embedding_model(), check_embedding_ctx_length=False,
                                  max_retries=12, **common)
    return llm, embeddings


class LocalHashEmbeddings(Embeddings):
    """LangChain-compatible local embeddings for Gemini-backed RAGAS."""

    def __init__(self, dimension: int = 1024):
        from src.m2_search import _HashingEncoder

        self.encoder = _HashingEncoder(dimension)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.encoder.encode(texts, show_progress_bar=False)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self.encoder.encode(text)
        return vector.tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def _provider_key() -> str:
    if LLM_PROVIDER == "gemini":
        return GEMINI_API_KEY
    if LLM_PROVIDER == "openai":
        return OPENAI_API_KEY
    return ""


def _chat_model() -> str:
    return GEMINI_MODEL if LLM_PROVIDER == "gemini" else EVALUATOR_MODEL


def _embedding_model() -> str:
    return GEMINI_EMBEDDING_MODEL if LLM_PROVIDER == "gemini" else EVALUATOR_EMBEDDING_MODEL


def ragas_model_ids() -> list[str]:
    """Model IDs actually configured for the RAGAS LLM and embeddings."""
    if not has_api_key():
        return []
    return [_chat_model(), _embedding_model()]
