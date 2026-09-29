"""Unit tests for configuration management and validation."""

import pytest
from src.config import AppConfig, ConfigurationError, load_config


def test_missing_groq_key_raises_configuration_error(monkeypatch):
    """Verify that runtime execution fails fast when GROQ_API_KEY is missing."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ConfigurationError) as exc_info:
        load_config(require_llm_key=True)

    assert "Missing GROQ_API_KEY" in str(exc_info.value)
    assert "Groq is the required LLM provider" in str(exc_info.value)


def test_placeholder_groq_key_raises_configuration_error(monkeypatch):
    """Verify that obvious placeholder values are rejected."""
    monkeypatch.setenv("GROQ_API_KEY", "YOUR_API_KEY")

    with pytest.raises(ConfigurationError) as exc_info:
        load_config(require_llm_key=True)

    assert "Missing GROQ_API_KEY" in str(exc_info.value)


def test_valid_groq_key_loads_config(monkeypatch):
    """Verify that configuration loads cleanly when GROQ_API_KEY is present."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_testDummyKeyForTestingPurposes67890")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    config = load_config(require_llm_key=True)
    assert isinstance(config, AppConfig)
    assert config.groq_api_key == "gsk_testDummyKeyForTestingPurposes67890"
    assert config.groq_model == "llama-3.3-70b-versatile"
    assert config.get_groq_api_key() == "gsk_testDummyKeyForTestingPurposes67890"


def test_require_llm_key_false_allows_loading_without_key(monkeypatch):
    """Verify that load_config(require_llm_key=False) succeeds for non-LLM operations."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    config = load_config(require_llm_key=False)
    assert config.groq_api_key is None
