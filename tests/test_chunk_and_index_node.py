"""Unit tests for chunk_and_index_node in LangGraph."""

from unittest.mock import patch
from src.graph.state import create_initial_state
from src.nodes.chunk_and_index import chunk_and_index_node


def test_chunk_and_index_node_empty_chunks():
    """Verify empty chunks in state transitions to error status."""
    state = create_initial_state("2401.12345")
    state["chunks"] = []

    update = chunk_and_index_node(state)
    assert update["status"] == "error"
    assert "No chunks found" in update["error"]


@patch("src.nodes.chunk_and_index.index_paper_chunks")
def test_chunk_and_index_node_success(mock_index):
    """Verify successful indexing updates vector_collection_name and status."""
    mock_index.return_value = 5

    state = create_initial_state("2401.12345")
    state["arxiv_id"] = "2401.12345"
    state["chunks"] = [
        {"text": "Sample text", "metadata": {"chunk_id": "c1", "paper_id": "2401.12345"}}
    ]

    update = chunk_and_index_node(state)
    assert update["status"] == "indexing"
    assert update["vector_collection_name"] == "paper_2401_12345"
    assert update["error"] is None
