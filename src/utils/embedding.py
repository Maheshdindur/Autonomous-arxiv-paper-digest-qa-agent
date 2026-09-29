"""Local sentence-transformers embedding generation.

Generates normalized dense vector embeddings using local open-weight models
(default: all-MiniLM-L6-v2). Runs locally on CPU without requiring external API keys.
"""

import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import load_config

logger = logging.getLogger("arxiv_agent.embedding")

# Module-level cache for loaded SentenceTransformer model
_MODEL_CACHE: dict = {}


def get_embedding_model(model_name: Optional[str] = None) -> SentenceTransformer:
    """Retrieve or initialize the cached SentenceTransformer model instance."""
    if not model_name:
        config = load_config(require_llm_key=False)
        model_name = config.embedding_model

    if model_name not in _MODEL_CACHE:
        logger.info("Loading local embedding model: '%s'", model_name)
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        logger.info("Successfully loaded embedding model: '%s'", model_name)

    return _MODEL_CACHE[model_name]


def embed_texts(
    texts: List[str],
    model_name: Optional[str] = None,
    normalize: bool = True,
    batch_size: int = 32,
) -> List[List[float]]:
    """Generate dense vector embeddings for a list of text strings.

    Args:
        texts: List of text strings to embed.
        model_name: Optional model identifier.
        normalize: If True, embeddings are L2-normalized so dot-product equals cosine similarity.
        batch_size: Batch size for encoding.

    Returns:
        List of float vectors.
    """
    if not texts:
        return []

    model = get_embedding_model(model_name)
    # sentence-transformers encode supports normalize_embeddings
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )

    return embeddings.tolist()


def embed_query(
    query: str,
    model_name: Optional[str] = None,
    normalize: bool = True,
) -> List[float]:
    """Generate a single normalized dense vector embedding for a query string."""
    embeddings = embed_texts([query], model_name=model_name, normalize=normalize)
    if not embeddings:
        return []
    return embeddings[0]
