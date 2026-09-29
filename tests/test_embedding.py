"""Unit tests for local sentence-transformers embedding generation."""

import numpy as np
import pytest

from src.utils.embedding import embed_query, embed_texts, get_embedding_model


def test_embedding_model_loading():
    """Verify local embedding model loads cleanly and reports expected dimension."""
    model = get_embedding_model("all-MiniLM-L6-v2")
    assert model is not None
    # Support both new and legacy sentence-transformers methods
    if hasattr(model, "get_embedding_dimension"):
        dim = model.get_embedding_dimension()
    else:
        dim = model.get_sentence_embedding_dimension()
    assert dim == 384


def test_embed_texts_dimension_and_normalization():
    """Verify generated embeddings have 384 dimensions and unit L2 norm."""
    texts = [
        "KV-cache compression enables long-context LLM serving.",
        "Quantum entanglement in string junctions on tori.",
    ]
    embeddings = embed_texts(texts, normalize=True)

    assert len(embeddings) == 2
    for vec in embeddings:
        assert len(vec) == 384
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, atol=1e-4)


def test_embed_query_shape():
    """Verify embed_query returns a 1D vector of length 384."""
    query_vec = embed_query("What is the memory overhead of attention?")
    assert isinstance(query_vec, list)
    assert len(query_vec) == 384
    assert np.isclose(np.linalg.norm(query_vec), 1.0, atol=1e-4)


def test_embedding_semantic_similarity():
    """Verify semantically similar sentences have higher cosine similarity than unrelated ones."""
    query = "attention key-value cache memory compression"
    doc_similar = "Techniques for reducing KV-cache footprint in transformer decoding."
    doc_unrelated = "Optical spectroscopic observations of distant supernova explosions."

    q_vec = np.array(embed_query(query, normalize=True))
    sim_vec = np.array(embed_query(doc_similar, normalize=True))
    unrel_vec = np.array(embed_query(doc_unrelated, normalize=True))

    score_similar = float(np.dot(q_vec, sim_vec))
    score_unrelated = float(np.dot(q_vec, unrel_vec))

    # Similar document must have significantly higher similarity than unrelated document
    assert score_similar > score_unrelated
    assert score_similar - score_unrelated > 0.25
    assert score_similar > 0.35
