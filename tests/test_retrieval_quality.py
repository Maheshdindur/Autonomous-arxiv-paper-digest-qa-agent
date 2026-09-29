"""Retrieval quality verification tests.

Verifies that semantic search returns the exact known ground-truth chunks
for targeted technical queries.
"""

from pathlib import Path
import chromadb
import pytest

from src.graph.state import DocumentChunk
from src.utils.pdf_parser import chunk_parsed_document, extract_text_from_pdf
from src.utils.vector_store import index_paper_chunks, query_similar_chunks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = PROJECT_ROOT / "downloads"


def test_retrieval_precision_on_known_ground_truth_chunks(tmp_path):
    """Verify top-1 retrieval matches the exact ground-truth chunk for distinct technical topics."""
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma_quality"))
    paper_id = "eval_paper_001"

    corpus: list[DocumentChunk] = [
        {
            "text": "Learning rate warmup linearly increases the learning rate from zero to the maximum value over the first 2000 steps to stabilize early transformer training.",
            "metadata": {"chunk_id": "chunk_lr_warmup", "paper_id": paper_id, "page_start": 2, "page_end": 2, "section": "Training", "word_count": 22},
        },
        {
            "text": "KV-cache eviction removes tokens with low attention scores from GPU memory while preserving important prompt context and recent tokens.",
            "metadata": {"chunk_id": "chunk_kv_eviction", "paper_id": paper_id, "page_start": 3, "page_end": 3, "section": "Method", "word_count": 20},
        },
        {
            "text": "FlashAttention uses GPU SRAM tiling to compute exact softmax without materializing the full N-by-N attention matrix in High Bandwidth Memory.",
            "metadata": {"chunk_id": "chunk_flash_attn", "paper_id": paper_id, "page_start": 4, "page_end": 4, "section": "Related Work", "word_count": 21},
        },
        {
            "text": "Byte-Pair Encoding merges the most frequent byte pairs iteratively to construct a subword vocabulary of 32000 tokens for text tokenization.",
            "metadata": {"chunk_id": "chunk_bpe_token", "paper_id": paper_id, "page_start": 1, "page_end": 1, "section": "Data", "word_count": 21},
        },
        {
            "text": "Quantization maps 16-bit floating point weights into 4-bit integers using symmetric per-channel scaling to decrease VRAM consumption.",
            "metadata": {"chunk_id": "chunk_quant", "paper_id": paper_id, "page_start": 5, "page_end": 5, "section": "Compression", "word_count": 18},
        },
    ]

    index_paper_chunks(paper_id=paper_id, chunks=corpus, client=client)

    test_queries = [
        ("How does the model evict tokens from key-value memory?", "chunk_kv_eviction"),
        ("What technique avoids materializing the attention matrix in HBM via SRAM tiling?", "chunk_flash_attn"),
        ("How is the learning rate scheduled in the initial training steps?", "chunk_lr_warmup"),
        ("What algorithm builds the 32000 subword vocabulary?", "chunk_bpe_token"),
        ("How are weights compressed from 16-bit floats to 4-bit integers?", "chunk_quant"),
    ]

    for question, expected_chunk_id in test_queries:
        results = query_similar_chunks(paper_id=paper_id, query=question, top_k=1, client=client)
        assert len(results) == 1, f"Expected 1 result for query: '{question}'"
        top_match = results[0]
        assert top_match["chunk_id"] == expected_chunk_id, (
            f"Query '{question}' expected chunk '{expected_chunk_id}' but got '{top_match['chunk_id']}' "
            f"(Score: {top_match['similarity_score']:.4f})"
        )
        assert top_match["similarity_score"] > 0.45


@pytest.mark.integration
def test_retrieval_quality_on_real_paper(tmp_path):
    """Verify semantic retrieval on real paper chunks (2512.14946: EVICPRESS)."""
    pdf_path = DOWNLOADS_DIR / "2512.14946v1.pdf"
    assert pdf_path.exists(), "Requires downloaded 2512.14946v1.pdf"

    doc = extract_text_from_pdf(pdf_path, paper_id="2512.14946")
    chunks = chunk_parsed_document(doc, chunk_size_words=500, chunk_overlap_words=100)

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma_real"))
    index_paper_chunks(paper_id="2512.14946", chunks=chunks, client=client)

    # Technical query about the paper's core theme
    query = "How does joint compression and eviction manage the key-value cache?"
    results = query_similar_chunks(paper_id="2512.14946", query=query, top_k=3, client=client)

    assert len(results) == 3
    top_chunk = results[0]
    # Check that retrieved text contains relevant terminology
    assert any(term in top_chunk["text"].lower() for term in ["eviction", "compression", "kv-cache", "serving"])
    assert top_chunk["similarity_score"] > 0.4
    assert top_chunk["metadata"]["page_start"] >= 1
