"""Live Groq LLM API integration tests.

Marked with 'live_llm' and 'integration'.
Runs ONLY when explicitly targeted with '-m live_llm'.
Strictly separated from the offline test suite.
"""

from pathlib import Path
import pytest
from dotenv import dotenv_values

from src.config import load_config
from src.utils.llm import call_groq_json


@pytest.mark.live_llm
@pytest.mark.integration
def test_live_groq_json_completion(monkeypatch):
    """Verify live Groq API connection and JSON generation when API key is configured."""
    # Check if a live key exists in environment or .env file
    env_file = Path(".env")
    key = None
    if env_file.exists():
        vals = dotenv_values(env_file)
        key = vals.get("GROQ_API_KEY")

    if not key or "YOUR_API_KEY" in key or len(key) < 20:
        pytest.skip("No valid live GROQ_API_KEY found in .env. Skipping live LLM test.")

    monkeypatch.setenv("GROQ_API_KEY", key)
    config = load_config(require_llm_key=True)

    messages = [
        {
            "role": "system",
            "content": "You are a test evaluator. Respond with a JSON object containing key 'status' with value 'ok'.",
        },
        {"role": "user", "content": "Respond with status ok in JSON format."},
    ]

    response = call_groq_json(messages=messages, model=config.groq_model)
    assert isinstance(response, dict)
    assert response.get("status") == "ok"
