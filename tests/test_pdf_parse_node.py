"""Unit tests for pdf_parse_node in LangGraph."""

from pathlib import Path
import pymupdf
import pytest

from src.graph.state import create_initial_state
from src.nodes.pdf_parse import pdf_parse_node


def test_pdf_parse_node_missing_pdf_path():
    """Verify missing pdf_path in state transitions to error status."""
    state = create_initial_state("2401.12345")
    state["pdf_path"] = None

    update = pdf_parse_node(state)
    assert update["status"] == "error"
    assert "No pdf_path found" in update["error"]


def test_pdf_parse_node_nonexistent_file():
    """Verify non-existent file path in state transitions to error status."""
    state = create_initial_state("2401.12345")
    state["pdf_path"] = "/nonexistent/path/paper.pdf"

    update = pdf_parse_node(state)
    assert update["status"] == "error"
    assert "File does not exist" in update["error"]


def test_pdf_parse_node_success(tmp_path):
    """Verify successful parsing updates raw_text, chunks, and status."""
    pdf_path = tmp_path / "valid_node_test.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), "1. Introduction\nThis is a test paper for testing the PDF parse node in LangGraph.")
    for i in range(15):
        page.insert_text((50, 100 + i * 20), f"Line {i} content text demonstrating parser functionality " * 3)
    doc.save(pdf_path)
    doc.close()

    state = create_initial_state("2401.12345")
    state["pdf_path"] = str(pdf_path)
    state["arxiv_id"] = "2401.12345"
    state["selected_paper"] = {
        "arxiv_id": "2401.12345",
        "title": "Test Title",
        "authors": ["Author"],
    }

    update = pdf_parse_node(state)
    assert update["status"] == "parsing_pdf"
    assert update["error"] is None
    assert len(update["raw_text"]) > 100
    assert len(update["chunks"]) >= 1

    first_chunk = update["chunks"][0]
    assert first_chunk["metadata"]["paper_id"] == "2401.12345"
    assert first_chunk["metadata"]["page_start"] == 1
