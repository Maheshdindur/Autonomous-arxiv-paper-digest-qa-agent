"""Real arXiv integration tests against the live official arXiv API.

Distinguished from unit tests: requires internet connectivity.
Marked with pytest marker 'integration'.
"""

from pathlib import Path
import pytest
import pymupdf

from src.utils.arxiv_api import fetch_paper_by_id, search_papers_by_topic
from src.utils.pdf_fetcher import download_and_verify_pdf, verify_pdf_file


@pytest.mark.integration
def test_real_arxiv_fetch_modern_id():
    """Verify live fetch of a well-known modern paper (Attention Is All You Need - 1706.03762)."""
    paper = fetch_paper_by_id("1706.03762")
    assert paper is not None
    assert "1706.03762" in paper["arxiv_id"]
    assert "Attention Is All You Need" in paper["title"]
    assert len(paper["authors"]) > 0
    assert "transformer" in paper["abstract"].lower()
    assert paper["pdf_url"].startswith("https://arxiv.org/pdf/")


@pytest.mark.integration
def test_real_arxiv_fetch_legacy_id():
    """Verify live fetch of a legacy format arXiv ID (hep-th/9901001)."""
    paper = fetch_paper_by_id("hep-th/9901001")
    assert paper is not None
    assert "hep-th/9901001" in paper["arxiv_id"]
    assert len(paper["title"]) > 0
    assert len(paper["authors"]) > 0
    assert paper["pdf_url"].startswith("https://arxiv.org/pdf/")


@pytest.mark.integration
def test_real_arxiv_topic_search():
    """Verify live topic search returns candidate papers with metadata."""
    topic = "KV-cache compression for large language models"
    candidates = search_papers_by_topic(topic, max_results=3)

    assert len(candidates) >= 1
    assert candidates[0]["title"]
    assert candidates[0]["abstract"]
    assert candidates[0]["pdf_url"].startswith("https://")


@pytest.mark.integration
def test_real_pdf_download_and_verify(tmp_path):
    """Verify live PDF download and PyMuPDF verification for a real paper."""
    # Use a small legacy paper: hep-th/9901001 (~3-4 pages, fast download)
    paper = fetch_paper_by_id("hep-th/9901001")
    assert paper is not None

    download_dir = tmp_path / "downloads"
    verified_path = download_and_verify_pdf(
        pdf_url=paper["pdf_url"],
        arxiv_id=paper["arxiv_id"],
        target_dir=download_dir,
    )

    assert verified_path.exists()
    assert verified_path.stat().st_size > 1024
    assert verify_pdf_file(verified_path) is True

    # Confirm PyMuPDF can inspect pages
    doc = pymupdf.open(verified_path)
    assert doc.page_count > 0
    first_page_text = doc[0].get_text()
    assert len(first_page_text) > 50
    doc.close()
