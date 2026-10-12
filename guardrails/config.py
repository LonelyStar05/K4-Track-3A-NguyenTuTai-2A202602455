"""NeMo loads this module before initialising LLMs.

NeMo 0.10.0 only maps engine "openai" to ChatOpenAI when the model name contains
"gpt-3.5"/"gpt-4"; any other name goes to the legacy completions API. Registering an
explicit chat engine lets the guard use OpenAI-compatible chat endpoints (e.g. Gemini
via OPENAI_API_BASE) without changing the self-check flows.
"""
from langchain_openai import ChatOpenAI
from nemoguardrails.llm.providers import register_llm_provider

register_llm_provider("openai_chat", ChatOpenAI)
