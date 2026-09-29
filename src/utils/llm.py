"""Groq LLM client wrapper.

Interacts with the official Groq API for structured language generation.
Strictly configured to use Groq as the single LLM provider.
"""

import json
import logging
from typing import Any, Dict, List, Optional
import groq

from src.config import load_config

logger = logging.getLogger("arxiv_agent.llm")


class LLMError(RuntimeError):
    """Raised when LLM API call fails, times out, or produces unparseable output."""


def get_groq_client(client: Optional[groq.Groq] = None) -> groq.Groq:
    """Retrieve or create Groq client using configured API key."""
    if client is not None:
        return client

    config = load_config(require_llm_key=True)
    api_key = config.get_groq_api_key()
    return groq.Groq(api_key=api_key)


def call_groq_chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    client: Optional[groq.Groq] = None,
) -> str:
    """Send chat completion request to Groq API."""
    client = get_groq_client(client)
    if not model:
        config = load_config(require_llm_key=False)
        model = config.groq_model

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        return content.strip()
    except Exception as e:
        logger.error("Groq API chat completion failed: %s", str(e))
        raise LLMError(f"Groq API call failed: {e}") from e


def call_groq_json(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.1,
    client: Optional[groq.Groq] = None,
) -> Dict[str, Any]:
    """Send request to Groq API enforcing JSON output format."""
    client = get_groq_client(client)
    if not model:
        config = load_config(require_llm_key=False)
        model = config.groq_model

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        raw_text = response.choices[0].message.content or ""
        if not raw_text.strip():
            raise LLMError("Groq returned empty response.")

        parsed_json = json.loads(raw_text)
        if not isinstance(parsed_json, dict):
            raise LLMError(f"Expected JSON object from Groq, received: {type(parsed_json)}")

        return parsed_json
    except json.JSONDecodeError as e:
        logger.error("Failed to decode JSON from Groq response: %s", str(e))
        raise LLMError(f"Malformed JSON returned by LLM: {e}") from e
    except Exception as e:
        logger.error("Groq API JSON call failed: %s", str(e))
        raise LLMError(f"Groq API call failed: {e}") from e
