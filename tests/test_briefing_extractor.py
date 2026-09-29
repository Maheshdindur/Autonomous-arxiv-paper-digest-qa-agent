"""Unit tests for briefing evidence assembly and prompt formatting (mocked)."""

import json
from unittest.mock import MagicMock
import pytest

from src.graph.state import DocumentChunk, PaperMetadata
from src.utils.briefing_extractor import (
    assemble_briefing_evidence,
    format_evidence_block,
    generate_executive_briefing,
)


def _make_sample_paper() -> PaperMetadata:
    return {
        "arxiv_id": "2401.12345",
        "title": "Efficient KV-Cache Compression for LLMs",
        "authors": ["Alice Smith", "Bob Jones"],
        "abstract": "We propose a new compression method that reduces GPU VRAM footprint.",
        "published": "2024-01-15T00:00:00Z",
        "entry_id": "https://arxiv.org/abs/2401.12345",
    }


def _make_sample_chunks() -> list[DocumentChunk]:
    return [
        {
            "text": "Abstract and introduction discussing high memory costs in LLM serving.",
            "metadata": {"chunk_id": "c001", "paper_id": "2401.12345", "page_start": 1, "page_end": 1, "section": "Introduction", "word_count": 10},
        },
        {
            "text": "Detailed background on auto-regressive generation and attention caches.",
            "metadata": {"chunk_id": "c002", "paper_id": "2401.12345", "page_start": 2, "page_end": 2, "section": "Background", "word_count": 9},
        },
        {
            "text": "Our method dynamically evicts stale tokens using an attention score threshold.",
            "metadata": {"chunk_id": "c003", "paper_id": "2401.12345", "page_start": 4, "page_end": 5, "section": "Methodology", "word_count": 11},
        },
        {
            "text": "Evaluation results show 2.5x throughput gain on LLaMA-70B with negligible perplexity drop.",
            "metadata": {"chunk_id": "c004", "paper_id": "2401.12345", "page_start": 8, "page_end": 9, "section": "Results", "word_count": 12},
        },
        {
            "text": "A key limitation is that our method requires FP16 key representations.",
            "metadata": {"chunk_id": "c005", "paper_id": "2401.12345", "page_start": 12, "page_end": 12, "section": "Limitations", "word_count": 11},
        },
    ]


def test_assemble_briefing_evidence_selects_key_aspects():
    """Verify evidence assembly collects relevant sections without duplicating all chunks."""
    paper = _make_sample_paper()
    chunks = _make_sample_chunks()

    evidence = assemble_briefing_evidence(paper, chunks)
    assert len(evidence) >= 3
    assert len(evidence) <= len(chunks)

    # Check evidence formatting contains provenance metadata
    formatted = format_evidence_block(evidence)
    assert "--- EVIDENCE PIECE" in formatted
    assert "Chunk ID:" in formatted
    assert "Page" in formatted
    assert "Section:" in formatted


def test_generate_executive_briefing_preserves_authoritative_metadata():
    """Verify authoritative arXiv metadata is strictly preserved and not overwritten by LLM."""
    paper = _make_sample_paper()
    chunks = _make_sample_chunks()

    mock_client = MagicMock()
    mock_choice = MagicMock()
    llm_payload = {
        "why_it_matters": "Reduces inference costs for serving large language models.",
        "problem_statement": "KV-cache grows linearly with sequence length causing GPU memory bottlenecks.",
        "method_approach": [
            "Dynamic eviction of low-attention tokens",
            "Adaptive quantization of residual cache",
        ],
        "key_results_claims": [
            "2.5x higher throughput on LLaMA-70B",
            "Under 0.1 perplexity increase",
        ],
        "limitations": [
            "Requires FP16 key representations",
        ],
        "suggested_followup_questions": [
            "How does this perform on reasoning benchmarks like GSM8K?",
            "Can this be combined with speculative decoding?",
        ],
    }
    mock_choice.message.content = json.dumps(llm_payload)
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    briefing = generate_executive_briefing(
        paper_meta=paper,
        evidence_chunks=chunks,
        groq_client=mock_client,
    )

    # Authoritative metadata checks
    assert briefing["title"] == "Efficient KV-Cache Compression for LLMs"
    assert briefing["authors"] == ["Alice Smith", "Bob Jones"]
    assert briefing["arxiv_id"] == "2401.12345"
    assert briefing["publish_date"] == "2024-01-15T00:00:00Z"
    assert briefing["link"] == "https://arxiv.org/abs/2401.12345"

    # LLM-generated fields checks
    assert briefing["why_it_matters"] == llm_payload["why_it_matters"]
    assert briefing["problem_statement"] == llm_payload["problem_statement"]
    assert len(briefing["method_approach"]) == 2
    assert len(briefing["key_results_claims"]) == 2
    assert briefing["limitations"] == ["Requires FP16 key representations"]
    assert len(briefing["suggested_followup_questions"]) == 2


def test_generate_executive_briefing_unspecified_limitations_handling():
    """Verify that when the LLM reports no limitations, an informative statement is preserved."""
    paper = _make_sample_paper()
    chunks = _make_sample_chunks()

    mock_client = MagicMock()
    mock_choice = MagicMock()
    llm_payload = {
        "why_it_matters": "Provides high performance.",
        "problem_statement": "High compute cost.",
        "method_approach": ["Algorithmic optimization"],
        "key_results_claims": ["10% faster"],
        "limitations": [],  # Empty limitations list
        "suggested_followup_questions": ["What is next?"],
    }
    mock_choice.message.content = json.dumps(llm_payload)
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    briefing = generate_executive_briefing(
        paper_meta=paper,
        evidence_chunks=chunks,
        groq_client=mock_client,
    )

    # Must contain informative statement rather than empty list or hallucinated limitation
    assert len(briefing["limitations"]) == 1
    assert "does not explicitly state limitations" in briefing["limitations"][0]


def test_generate_executive_briefing_omits_incomplete_truncated_claims():
    """Regression test: verify incomplete or truncated numerical claims are omitted from output."""
    paper = _make_sample_paper()
    chunks = _make_sample_chunks()

    mock_client = MagicMock()
    mock_choice = MagicMock()
    llm_payload = {
        "why_it_matters": "Improves efficiency.",
        "problem_statement": "High memory consumption.",
        "method_approach": ["Eviction algorithm"],
        "key_results_claims": [
            "EVICPRESS reduces TTFT by 1.43× to (value truncated in evidence)",
            "Improves quality by 13.58% to 55.40% at the same TTFT",
            "Baseline latency is reduced by 2.5x to",
        ],
        "limitations": ["Requires FP16"],
        "suggested_followup_questions": ["What is next?"],
    }
    mock_choice.message.content = json.dumps(llm_payload)
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    briefing = generate_executive_briefing(
        paper_meta=paper,
        evidence_chunks=chunks,
        groq_client=mock_client,
    )

    # Incomplete claims must be omitted; complete claim must be preserved
    claims = briefing["key_results_claims"]
    assert len(claims) == 1
    assert claims[0] == "Improves quality by 13.58% to 55.40% at the same TTFT"
    assert not any("truncated" in c.lower() for c in claims)
    assert not any(c.strip().endswith("to") for c in claims)


def test_generate_executive_briefing_all_claims_truncated_fallback():
    """Regression test: when all claims are truncated, emit explicit statement that evidence lacks complete values."""
    paper = _make_sample_paper()
    chunks = _make_sample_chunks()

    mock_client = MagicMock()
    mock_choice = MagicMock()
    llm_payload = {
        "why_it_matters": "Improves efficiency.",
        "problem_statement": "High memory consumption.",
        "method_approach": ["Eviction algorithm"],
        "key_results_claims": [
            "EVICPRESS reduces TTFT by 1.43× to (value truncated in evidence)",
        ],
        "limitations": ["Requires FP16"],
        "suggested_followup_questions": ["What is next?"],
    }
    mock_choice.message.content = json.dumps(llm_payload)
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    briefing = generate_executive_briefing(
        paper_meta=paper,
        evidence_chunks=chunks,
        groq_client=mock_client,
    )

    claims = briefing["key_results_claims"]
    assert len(claims) == 1
    assert "does not provide complete" in claims[0].lower()
    assert "truncated" not in claims[0].lower()


def test_assemble_briefing_evidence_completes_trailing_sentence():
    """Verify trailing sentences ending mid-sentence at chunk boundaries are completed from the next chunk."""
    paper = _make_sample_paper()
    chunks: list[DocumentChunk] = [
        {
            "text": "Introduction section discussing LLM efficiency and KV caching.",
            "metadata": {"chunk_id": "c001", "paper_id": "2401.12345", "page_start": 1, "page_end": 1, "section": "Introduction", "word_count": 8},
        },
        {
            "text": "Evaluation results show EVICPRESS reduces TTFT by 1.43 to",
            "metadata": {"chunk_id": "c002", "paper_id": "2401.12345", "page_start": 2, "page_end": 2, "section": "Results", "word_count": 8},
        },
        {
            "text": "reduces TTFT by 1.43 to 3.77x on LongBench benchmarks.",
            "metadata": {"chunk_id": "c003", "paper_id": "2401.12345", "page_start": 2, "page_end": 3, "section": "Results", "word_count": 9},
        },
    ]

    evidence = assemble_briefing_evidence(paper, chunks)
    results_chunk = next(c for c in evidence if c["metadata"]["chunk_id"] == "c002")
    assert "3.77x on LongBench benchmarks." in results_chunk["text"]
    assert results_chunk["text"].endswith(".")
