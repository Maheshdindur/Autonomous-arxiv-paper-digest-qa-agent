"""Test fixtures and shared configuration for pytest."""

import pytest


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Ensure tests run in a clean, predictable environment isolated from disk .env."""
    # Prevent tests from implicitly reading the disk .env file
    monkeypatch.setattr("src.config.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
