"""Configuration management and logging setup.

Loads environment variables from `.env` in the project root.
Validates that required Groq credentials exist before runtime execution.
Ensures no silent fallback to mock LLM at runtime.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Resolve project root (/home/mahesh/Desktop/8byte)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass
class AppConfig:
    """Strongly typed application configuration."""
    groq_api_key: Optional[str]
    groq_model: str
    embedding_model: str
    chroma_persist_dir: Path
    pdf_download_dir: Path
    log_level: str

    def get_groq_api_key(self) -> str:
        """Returns the Groq API key or raises ConfigurationError."""
        if not self.groq_api_key:
            raise ConfigurationError(
                "GROQ_API_KEY is required. Please set it in your .env file or environment."
            )
        return self.groq_api_key


def _clean_key(key: Optional[str]) -> Optional[str]:
    """Return key only if non-empty and not a known placeholder."""
    if not key:
        return None
    val = key.strip()
    if val.upper() in {
        "YOUR_API_KEY",
        "YOUR_GROQ_API_KEY",
        "CHANGEME",
        "PLACEHOLDER",
        "NONE",
        "NULL",
    }:
        return None
    if len(val) < 8:
        return None
    return val


def load_config(require_llm_key: bool = True) -> AppConfig:
    """Load configuration from environment and .env file.

    Args:
        require_llm_key: If True, raises ConfigurationError if GROQ_API_KEY is not set.
                         Set to False for commands/nodes that do not require LLM calls.
    """
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()

    groq_key = _clean_key(os.getenv("GROQ_API_KEY"))

    if require_llm_key and not groq_key:
        raise ConfigurationError(
            "Missing GROQ_API_KEY. Groq is the required LLM provider.\n"
            "Please configure a valid GROQ_API_KEY in your .env file or environment.\n"
            "Placeholder values like 'YOUR_API_KEY' are not accepted.\n"
            "Refer to .env.example for guidance."
        )

    groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
    chroma_dir = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
    pdf_dir = PROJECT_ROOT / os.getenv("PDF_DOWNLOAD_DIR", "downloads")
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    return AppConfig(
        groq_api_key=groq_key,
        groq_model=groq_model,
        embedding_model=embedding_model,
        chroma_persist_dir=chroma_dir,
        pdf_download_dir=pdf_dir,
        log_level=log_level,
    )


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the root logger for the application."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("arxiv_agent")
