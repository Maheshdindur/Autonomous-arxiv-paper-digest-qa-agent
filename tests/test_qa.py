"""Unit and integration tests for Grounded Question Answering (QA).

Covers:
- Clearly answerable question with provenance citations
- Multi-section synthesis across different chunks
- Unanswerable question (refusal handling)
- Question unrelated to paper (strict anti-hallucination refusal)
- Provenance/citation preservation (chunk_id, page range, section)
- Empty/poor retrieval handling
- Malformed/empty LLM response handling
- Conversation history tracking
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from src.graph.state import AgentState, DocumentChunk, PaperMetadata, create_initial_state
from src.graph.workflow import build_qa_graph
from src.nodes.qa import qa_answer_node, qa_retrieval_node
from src.utils.llm import LLMError
from src.utils.qa_engine import (
    REFUSAL_MESSAGE,
    format_conversation_history,
    format_qa_context,
    generate_grounded_answer,
    retrieve_qa_evidence,
)


def _make_sample_paper() -> PaperMetadata:
    return {
        "arxiv_id": "2401.12345",
        "title": "EVICPRESS: Joint KV-Cache Compression and Eviction",
        "authors": ["Alice Smith", "Bob Jones"],
        "abstract": "We present EVICPRESS for efficient KV-cache compression and eviction.",
        "published": "2024-01-15T00:00:00Z",
        "entry_id": "https://arxiv.org/abs/2401.12345",
    }


def _make_sample_evidence_chunks() -> list[DocumentChunk]:
    return [
        {
            "text": "EVICPRESS jointly decides which KV caches to compress and which to evict using a unified utility function.",
            "metadata": {
                "chunk_id": "2401.12345_c0001",
                "paper_id": "2401.12345",
                "page_start": 2,
                "page_end": 2,
                "section": "Method",
                "word_count": 18,
            },
        },
        {
            "text": "Across 12 datasets, EVICPRESS achieves up to 2.19x faster TTFT compared to baselines that only evict.",
            "metadata": {
                "chunk_id": "2401.12345_c0002",
                "paper_id": "2401.12345",
                "page_start": 7,
                "page_end": 8,
                "section": "Evaluation",
                "word_count": 16,
            },
        },
    ]


def test_qa_clearly_answerable_question():
    """Verify that answerable questions produce grounded responses with citations."""
    paper = _make_sample_paper()
    chunks = _make_sample_evidence_chunks()

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_payload = {
        "answer": "EVICPRESS uses a unified utility function to decide which caches to compress and evict [Chunk ID: 2401.12345_c0001, Page 2].",
        "cited_chunk_ids": ["2401.12345_c0001"],
        "is_sufficient_evidence": True,
    }
    mock_choice.message.content = json.dumps(mock_payload)
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    result = generate_grounded_answer(
        paper_meta=paper,
        question="How does EVICPRESS decide which caches to compress and evict?",
        evidence_chunks=chunks,
        groq_client=mock_client,
    )

    assert result["grounded"] is True
    assert "2401.12345_c0001" in result["answer"]
    assert "2401.12345_c0001" in result["cited_chunk_ids"]
    assert len(result["sources"]) == 2
    assert result["sources"][0]["chunk_id"] == "2401.12345_c0001"
    assert result["sources"][0]["page_start"] == 2
    assert result["sources"][0]["section"] == "Method"


def test_qa_multi_section_synthesis():
    """Verify synthesis across multiple chunks spanning distinct sections (Method & Evaluation)."""
    paper = _make_sample_paper()
    chunks = _make_sample_evidence_chunks()

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_payload = {
        "answer": "EVICPRESS uses a unified utility function [Chunk ID: 2401.12345_c0001, Page 2] and achieves up to 2.19x faster TTFT across 12 datasets [Chunk ID: 2401.12345_c0002, Page 7-8].",
        "cited_chunk_ids": ["2401.12345_c0001", "2401.12345_c0002"],
        "is_sufficient_evidence": True,
    }
    mock_choice.message.content = json.dumps(mock_payload)
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    result = generate_grounded_answer(
        paper_meta=paper,
        question="What method does EVICPRESS use and what speedup does it achieve in evaluations?",
        evidence_chunks=chunks,
        groq_client=mock_client,
    )

    assert result["grounded"] is True
    assert "2401.12345_c0001" in result["cited_chunk_ids"]
    assert "2401.12345_c0002" in result["cited_chunk_ids"]
    assert "2401.12345_c0001" in result["answer"]
    assert "2401.12345_c0002" in result["answer"]


def test_qa_unanswerable_question():
    """Verify that questions not supported by the evidence produce the standard refusal message."""
    paper = _make_sample_paper()
    chunks = _make_sample_evidence_chunks()

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_payload = {
        "answer": REFUSAL_MESSAGE,
        "cited_chunk_ids": [],
        "is_sufficient_evidence": False,
    }
    mock_choice.message.content = json.dumps(mock_payload)
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    result = generate_grounded_answer(
        paper_meta=paper,
        question="What was the learning rate and optimizer used during pretraining?",
        evidence_chunks=chunks,
        groq_client=mock_client,
    )

    assert result["grounded"] is False
    assert result["answer"] == REFUSAL_MESSAGE
    assert result["sources"] == []


def test_qa_unrelated_question():
    """Verify that unrelated off-topic questions produce refusal rather than hallucination."""
    paper = _make_sample_paper()
    chunks = _make_sample_evidence_chunks()

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_payload = {
        "answer": "The provided paper does not contain enough information to answer this question.",
        "cited_chunk_ids": [],
        "is_sufficient_evidence": False,
    }
    mock_choice.message.content = json.dumps(mock_payload)
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    result = generate_grounded_answer(
        paper_meta=paper,
        question="How do I bake classic sourdough bread?",
        evidence_chunks=chunks,
        groq_client=mock_client,
    )

    assert result["grounded"] is False
    assert result["answer"] == REFUSAL_MESSAGE


def test_qa_provenance_citation_preservation():
    """Verify that retrieved chunk metadata (chunk_id, page_start, page_end, section) is strictly preserved."""
    chunks = _make_sample_evidence_chunks()
    formatted = format_qa_context(chunks)

    assert "Chunk ID: 2401.12345_c0001" in formatted
    assert "Page 2-2" in formatted
    assert "Section: Method" in formatted

    assert "Chunk ID: 2401.12345_c0002" in formatted
    assert "Page 7-8" in formatted
    assert "Section: Evaluation" in formatted


def test_qa_empty_poor_retrieval_handling():
    """Verify that empty retrieval returns refusal immediately without calling LLM."""
    paper = _make_sample_paper()
    mock_client = MagicMock()

    result = generate_grounded_answer(
        paper_meta=paper,
        question="What is the speedup?",
        evidence_chunks=[],
        groq_client=mock_client,
    )

    # Must refuse immediately and not make any LLM API calls
    assert result["answer"] == REFUSAL_MESSAGE
    assert result["grounded"] is False
    assert result["sources"] == []
    mock_client.chat.completions.create.assert_not_called()


def test_qa_malformed_or_empty_llm_response_handling():
    """Verify that malformed or empty LLM output is handled gracefully without unhandled crashes."""
    state = create_initial_state("2401.12345")
    state["arxiv_id"] = "2401.12345"
    state["current_question"] = "What is the speedup?"
    state["retrieved_chunks"] = _make_sample_evidence_chunks()

    with patch("src.nodes.qa.generate_grounded_answer") as mock_gen:
        mock_gen.side_effect = LLMError("Malformed JSON returned by LLM: Unterminated string")
        result = qa_answer_node(state)

        assert result["status"] == "error"
        assert "QA answer generation failed" in result["error"]
        assert "Malformed JSON" in result["error"]


def test_qa_conversation_history_tracking():
    """Verify conversation history maintains past turns without contaminating primary evidence."""
    paper = _make_sample_paper()
    chunks = _make_sample_evidence_chunks()

    history = [
        {
            "question": "What is EVICPRESS?",
            "answer": "EVICPRESS is a joint compression and eviction framework [Chunk ID: 2401.12345_c0001, Page 2].",
            "sources": [{"chunk_id": "2401.12345_c0001"}],
        }
    ]

    history_str = format_conversation_history(history)
    assert "Turn 1 User Question: What is EVICPRESS?" in history_str
    assert "Turn 1 Assistant Answer: EVICPRESS is a joint" in history_str
    assert "NOT PRIMARY EVIDENCE" in history_str

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_payload = {
        "answer": "It achieves up to 2.19x faster TTFT [Chunk ID: 2401.12345_c0002, Page 7-8].",
        "cited_chunk_ids": ["2401.12345_c0002"],
        "is_sufficient_evidence": True,
    }
    mock_choice.message.content = json.dumps(mock_payload)
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    state: AgentState = {
        "user_input": "2401.12345",
        "arxiv_id": "2401.12345",
        "selected_paper": paper,
        "current_question": "What speedup does it achieve?",
        "retrieved_chunks": chunks,
        "conversation_history": history,
        "status": "qa_answering",
    }

    with patch("src.nodes.qa.generate_grounded_answer") as mock_gen:
        mock_gen.return_value = {
            "answer": mock_payload["answer"],
            "sources": [{"chunk_id": "2401.12345_c0002", "page_start": 7, "page_end": 8, "section": "Evaluation"}],
            "grounded": True,
            "cited_chunk_ids": ["2401.12345_c0002"],
        }
        res = qa_answer_node(state)

        assert res["status"] == "completed"
        assert len(res["conversation_history"]) == 2
        assert res["conversation_history"][0]["question"] == "What is EVICPRESS?"
        assert res["conversation_history"][1]["question"] == "What speedup does it achieve?"


def test_build_qa_graph_execution():
    """Verify that the compiled QA graph runs end-to-end through LangGraph."""
    graph = build_qa_graph().compile()

    state = create_initial_state("2401.12345")
    state["arxiv_id"] = "2401.12345"
    state["current_question"] = "What is the speedup?"

    with patch("src.nodes.qa.retrieve_qa_evidence") as mock_retrieve, \
         patch("src.nodes.qa.generate_grounded_answer") as mock_gen:
        mock_retrieve.return_value = _make_sample_evidence_chunks()
        mock_gen.return_value = {
            "answer": "EVICPRESS achieves 2.19x faster TTFT [Chunk ID: 2401.12345_c0002, Page 7-8].",
            "sources": [{"chunk_id": "2401.12345_c0002", "page_start": 7, "page_end": 8, "section": "Evaluation"}],
            "grounded": True,
            "cited_chunk_ids": ["2401.12345_c0002"],
        }

        output = graph.invoke(state)

        assert output["status"] == "completed"
        assert "2.19x faster" in output["qa_answer"]
        assert len(output["conversation_history"]) == 1
