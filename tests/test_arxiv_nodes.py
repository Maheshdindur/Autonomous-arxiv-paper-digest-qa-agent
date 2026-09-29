"""Unit tests for Phase 2 LangGraph nodes (mocked)."""

from unittest.mock import MagicMock, patch
from src.graph.state import create_initial_state
from src.nodes.arxiv_retrieval import arxiv_retrieval_node
from src.nodes.pdf_fetch import pdf_fetch_node
from src.nodes.query_understanding import query_understanding_node


def test_query_understanding_node_empty_input():
    """Verify empty input transitions to error status."""
    state = create_initial_state("   ")
    update = query_understanding_node(state)
    assert update["status"] == "error"
    assert "Query cannot be empty" in update["error"]


def test_query_understanding_node_paper_id():
    """Verify arXiv ID is classified as paper lookup."""
    state = create_initial_state("2401.12345")
    update = query_understanding_node(state)
    assert update["query_type"] == "paper"
    assert update["arxiv_id"] == "2401.12345"
    assert update["error"] is None


def test_query_understanding_node_topic():
    """Verify natural language string is classified as topic search."""
    state = create_initial_state("KV-cache compression")
    update = query_understanding_node(state)
    assert update["query_type"] == "topic"
    assert update["topic"] == "KV-cache compression"
    assert update["arxiv_id"] is None


@patch("src.nodes.arxiv_retrieval.fetch_paper_by_id")
def test_arxiv_retrieval_node_paper_found(mock_fetch):
    """Verify successful paper metadata lookup."""
    mock_paper = {
        "arxiv_id": "2401.12345",
        "title": "Test Title",
        "authors": ["Test Author"],
        "abstract": "Test Abstract",
        "published": "2024-01-01",
        "pdf_url": "https://arxiv.org/pdf/2401.12345.pdf",
    }
    mock_fetch.return_value = mock_paper

    state = create_initial_state("2401.12345")
    state["query_type"] = "paper"
    state["arxiv_id"] = "2401.12345"

    update = arxiv_retrieval_node(state)
    assert update["status"] == "retrieving"
    assert update["selected_paper"] == mock_paper
    assert update["candidate_papers"] == [mock_paper]


@patch("src.nodes.arxiv_retrieval.fetch_paper_by_id")
def test_arxiv_retrieval_node_paper_not_found(mock_fetch):
    """Verify missing paper sets error status."""
    mock_fetch.return_value = None

    state = create_initial_state("9999.99999")
    state["query_type"] = "paper"
    state["arxiv_id"] = "9999.99999"

    update = arxiv_retrieval_node(state)
    assert update["status"] == "error"
    assert "was not found on arXiv" in update["error"]


@patch("src.nodes.arxiv_retrieval.search_papers_by_topic")
def test_arxiv_retrieval_node_zero_topic_results(mock_search):
    """Verify zero search results sets error status."""
    mock_search.return_value = []

    state = create_initial_state("obscure query")
    state["query_type"] = "topic"
    state["topic"] = "obscure query"

    update = arxiv_retrieval_node(state)
    assert update["status"] == "error"
    assert "No papers found" in update["error"]


@patch("src.nodes.pdf_fetch.download_and_verify_pdf")
def test_pdf_fetch_node_success(mock_download, tmp_path):
    """Verify successful PDF download sets pdf_path in state."""
    fake_path = tmp_path / "paper.pdf"
    mock_download.return_value = fake_path

    state = create_initial_state("2401.12345")
    state["selected_paper"] = {
        "arxiv_id": "2401.12345",
        "pdf_url": "https://arxiv.org/pdf/2401.12345.pdf",
    }

    update = pdf_fetch_node(state)
    assert update["status"] == "fetching_pdf"
    assert update["pdf_path"] == str(fake_path)
    assert update["error"] is None
