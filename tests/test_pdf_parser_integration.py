"""Real arXiv paper PDF parsing integration tests on downloaded papers.

Tests extraction, section detection, and chunking against real research papers.
Marked with pytest marker 'integration'.
"""

from pathlib import Path
import pytest

from src.utils.pdf_parser import chunk_parsed_document, extract_text_from_pdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = PROJECT_ROOT / "downloads"


@pytest.mark.integration
def test_real_paper_parsing_hep_th():
    """Verify parsing and chunking on real legacy paper (hep-th/9901001v3)."""
    pdf_path = DOWNLOADS_DIR / "hep-th_9901001v3.pdf"
    assert pdf_path.exists(), "Test requires downloaded hep-th_9901001v3.pdf"

    doc = extract_text_from_pdf(pdf_path, paper_id="hep-th/9901001")

    assert doc.paper_id == "hep-th/9901001"
    assert doc.total_pages >= 3
    assert len(doc.full_text) > 2000
    assert len(doc.pages) == doc.total_pages

    # Check page numbers are correctly 1-indexed
    for idx, page in enumerate(doc.pages):
        assert page.page_number == idx + 1
        assert len(page.text) > 0

    # Chunk the real paper
    chunks = chunk_parsed_document(doc, chunk_size_words=400, chunk_overlap_words=80)
    assert len(chunks) >= 3

    for chunk in chunks:
        assert len(chunk["text"]) > 50
        meta = chunk["metadata"]
        assert meta["paper_id"] == "hep-th/9901001"
        assert meta["chunk_id"].startswith("hep-th/9901001_c")
        assert 1 <= meta["page_start"] <= meta["page_end"] <= doc.total_pages
        assert meta["token_count"] <= 400


@pytest.mark.integration
def test_real_paper_parsing_kv_cache():
    """Verify section detection and chunking on modern research paper (2512.14946v1)."""
    pdf_path = DOWNLOADS_DIR / "2512.14946v1.pdf"
    assert pdf_path.exists(), "Test requires downloaded 2512.14946v1.pdf"

    doc = extract_text_from_pdf(pdf_path, paper_id="2512.14946")

    assert doc.paper_id == "2512.14946"
    assert doc.total_pages > 5
    assert len(doc.full_text) > 10000

    # Verify detected sections include standard academic sections
    detected_sections = {p.detected_section for p in doc.pages if p.detected_section}
    assert len(detected_sections) > 0
    # Common sections like Introduction or References should be present
    common_matches = detected_sections.intersection({"Introduction", "Background", "Methodology", "Experiments", "References", "Conclusion"})
    assert len(common_matches) > 0

    chunks = chunk_parsed_document(doc, chunk_size_words=500, chunk_overlap_words=100)
    assert len(chunks) > 10

    # Ensure metadata integrity across all chunks
    for chunk in chunks:
        meta = chunk["metadata"]
        assert meta["paper_id"] == "2512.14946"
        assert 1 <= meta["page_start"] <= meta["page_end"] <= doc.total_pages
        # section must be either a valid non-empty string or None
        assert meta["section"] is None or isinstance(meta["section"], str)
