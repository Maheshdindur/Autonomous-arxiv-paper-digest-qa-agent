"""Unit tests for Groq LLM client wrapper (mocked)."""

import json
from unittest.mock import MagicMock
import pytest

from src.utils.llm import LLMError, call_groq_chat, call_groq_json


def test_call_groq_chat_success():
    """Verify chat completion returns stripped message content."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "  Hello from Groq!  "
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    result = call_groq_chat(
        messages=[{"role": "user", "content": "hi"}],
        model="llama-3.3-70b-versatile",
        client=mock_client,
    )
    assert result == "Hello from Groq!"


def test_call_groq_json_valid_response():
    """Verify JSON mode correctly decodes JSON response from LLM."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    payload = {"problem": "latency", "solution": "caching"}
    mock_choice.message.content = json.dumps(payload)
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    result = call_groq_json(
        messages=[{"role": "user", "content": "summarize"}],
        model="llama-3.3-70b-versatile",
        client=mock_client,
    )
    assert result == payload
    assert result["problem"] == "latency"


def test_call_groq_json_malformed_raises_error():
    """Verify malformed JSON raises LLMError."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Not a JSON: {unclosed bracket"
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    with pytest.raises(LLMError) as exc_info:
        call_groq_json(
            messages=[{"role": "user", "content": "summarize"}],
            client=mock_client,
        )
    assert "Malformed JSON" in str(exc_info.value)


def test_call_groq_json_empty_raises_error():
    """Verify empty response raises LLMError."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = ""
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    with pytest.raises(LLMError) as exc_info:
        call_groq_json(
            messages=[{"role": "user", "content": "summarize"}],
            client=mock_client,
        )
    assert "empty response" in str(exc_info.value)
