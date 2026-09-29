"""Unit tests for summarize_node in LangGraph."""

from unittest.mock import patch
from src.graph.state import create_initial_state
from src.nodes.summarize import format_briefing_markdown, summarize_node


def test_summarize_node_missing_paper():
    """Verify missing selected_paper in state transitions to error status."""
    state = create_initial_state("2401.12345")
    state["selected_paper"] = None

    update = summarize_node(state)
    assert update["status"] == "error"
    assert "No paper selected" in update["error"]


@patch("src.nodes.summarize.generate_executive_briefing")
def test_summarize_node_success(mock_generate):
    """Verify successful briefing generation updates briefing and status."""
    mock_briefing = {
        "title": "Test Title",
        "authors": ["Author One"],
        "arxiv_id": "2401.12345",
        "publish_date": "2024-01-01",
        "link": "https://arxiv.org/abs/2401.12345",
        "why_it_matters": "Significant breakthrough in LLM serving.",
        "problem_statement": "Memory bottlenecks.",
        "method_approach": ["Technique A"],
        "key_results_claims": ["2x speedup"],
        "limitations": ["Requires GPU with tensor cores"],
        "suggested_followup_questions": ["How does it scale?"],
    }
    mock_generate.return_value = mock_briefing

    state = create_initial_state("2401.12345")
    state["selected_paper"] = {
        "arxiv_id": "2401.12345",
        "title": "Test Title",
        "authors": ["Author One"],
    }
    state["chunks"] = [{"text": "sample text", "metadata": {"chunk_id": "c1", "paper_id": "2401.12345"}}]

    update = summarize_node(state)
    assert update["status"] == "completed"
    assert update["briefing"] == mock_briefing
    assert update["error"] is None

    # Test Markdown formatting contains all required headers
    md = format_briefing_markdown(mock_briefing)
    assert "# Executive Briefing: Test Title" in md
    assert "Why This Paper Matters" in md
    assert "Problem Statement" in md
    assert "Method & Approach" in md
    assert "Key Results & Claims" in md
    assert "Limitations" in md
    assert "Suggested Follow-Up Questions" in md
    assert "Requires GPU with tensor cores" in md
