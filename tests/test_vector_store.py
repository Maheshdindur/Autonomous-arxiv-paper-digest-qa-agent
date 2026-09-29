"""Unit tests for Chroma local persistent vector database wrapper."""

from pathlib import Path
import pytest
import chromadb

from src.graph.state import DocumentChunk
from src.utils.vector_store import (
    get_chroma_client,
    get_or_create_paper_collection,
    index_paper_chunks,
    query_similar_chunks,
    sanitize_collection_name,
)


def test_sanitize_collection_name():
    """Verify collection names comply with Chroma naming rules."""
    assert sanitize_collection_name("2401.12345") == "paper_2401_12345"
    assert sanitize_collection_name("hep-th/9901001") == "paper_hep-th_9901001"
    assert sanitize_collection_name("cs.CL/0101001v2") == "paper_cs_CL_0101001v2"


def test_index_and_retrieve_chunks_with_metadata(tmp_path):
    """Verify indexing preserves metadata and query_similar_chunks retrieves top-K with scores."""
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    paper_id = "test_paper_001"

    chunks: list[DocumentChunk] = [
        {
            "text": "The transformer architecture relies on multi-head self-attention mechanisms.",
            "metadata": {
                "chunk_id": "test_c000",
                "paper_id": paper_id,
                "page_start": 1,
                "page_end": 1,
                "section": "Introduction",
                "word_count": 9,
            },
        },
        {
            "text": "KV-cache stores key and value representations during auto-regressive generation to avoid recomputation.",
            "metadata": {
                "chunk_id": "test_c001",
                "paper_id": paper_id,
                "page_start": 2,
                "page_end": 2,
                "section": "Background",
                "word_count": 13,
            },
        },
        {
            "text": "We evaluate on GSM8K and HumanEval benchmarks under greedy decoding.",
            "metadata": {
                "chunk_id": "test_c002",
                "paper_id": paper_id,
                "page_start": 5,
                "page_end": 6,
                "section": None,  # Test None section preservation
                "word_count": 10,
            },
        },
    ]

    indexed_count = index_paper_chunks(paper_id=paper_id, chunks=chunks, client=client)
    assert indexed_count == 3

    # Query for KV-cache information
    results = query_similar_chunks(
        paper_id=paper_id,
        query="Why is the key-value cache used during generation?",
        top_k=2,
        client=client,
    )

    assert len(results) == 2
    top_hit = results[0]
    assert top_hit["chunk_id"] == "test_c001"
    assert "recomputation" in top_hit["text"]
    assert top_hit["metadata"]["page_start"] == 2
    assert top_hit["metadata"]["section"] == "Background"
    assert top_hit["similarity_score"] > 0.4
    assert "distance" in top_hit

    # Query for benchmarks to check None section restoration
    bench_results = query_similar_chunks(
        paper_id=paper_id,
        query="What benchmarks were used for evaluation?",
        top_k=1,
        client=client,
    )
    assert len(bench_results) == 1
    assert bench_results[0]["chunk_id"] == "test_c002"
    assert bench_results[0]["metadata"]["section"] is None


def test_query_empty_collection(tmp_path):
    """Verify querying an un-indexed collection returns an empty list without error."""
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma_empty"))
    results = query_similar_chunks(paper_id="nonexistent_paper", query="test query", client=client)
    assert results == []
