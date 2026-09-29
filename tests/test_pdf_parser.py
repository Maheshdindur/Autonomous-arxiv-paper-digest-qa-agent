"""Unit tests for PDF text extraction, section detection, and chunking (synthetic PDFs)."""

from pathlib import Path
import pymupdf
import pytest

from src.utils.pdf_parser import (
    PDFParsingError,
    _detect_heading,
    chunk_parsed_document,
    extract_text_from_pdf,
)


def test_detect_heading_patterns():
    """Verify heading detection matches standard academic headers and rejects arbitrary text."""
    assert _detect_heading("1. Introduction") == "Introduction"
    assert _detect_heading("2 Related Work") == "Related Work"
    assert _detect_heading("Methodology") == "Methodology"
    assert _detect_heading("3.1 Approach") == "Approach"
    assert _detect_heading("4. Experiments and Results") == "Experiments"
    assert _detect_heading("5. Discussion") == "Discussion"
    assert _detect_heading("6. Limitations") == "Limitations"
    assert _detect_heading("7. Conclusion") == "Conclusion"
    assert _detect_heading("References") == "References"
    assert _detect_heading("Appendix A") == "Appendix A"

    # Arbitrary sentences or long text must NOT be detected as sections
    assert _detect_heading("In this section we discuss the background of our proposed system.") is None
    assert _detect_heading("This is a regular sentence describing our methodology in detail.") is None
    assert _detect_heading("Table 1: Comparison of baseline models.") is None
    assert _detect_heading("") is None


def test_extract_text_with_sections(tmp_path):
    """Verify text extraction and section detection on a multi-page synthetic document."""
    pdf_path = tmp_path / "structured_paper.pdf"
    doc = pymupdf.open()

    # Page 1: Abstract & Introduction
    page1 = doc.new_page()
    page1.insert_text((50, 72), "Abstract\nThis paper proposes a new method for test parsing.")
    page1.insert_text((50, 150), "1. Introduction\nLarge language models require efficient caching mechanisms.")

    # Page 2: Methodology & Experiments
    page2 = doc.new_page()
    page2.insert_text((50, 72), "2. Methodology\nWe describe our algorithm and formal proofs here.")
    page2.insert_text((50, 200), "3. Experiments\nWe evaluate our approach on benchmark datasets.")

    doc.save(pdf_path)
    doc.close()

    parsed = extract_text_from_pdf(pdf_path, paper_id="2401.00001")

    assert parsed.paper_id == "2401.00001"
    assert parsed.total_pages == 2
    assert len(parsed.pages) == 2
    assert "Abstract" in parsed.full_text
    assert "Methodology" in parsed.full_text
    assert parsed.pages[0].page_number == 1
    assert parsed.pages[1].page_number == 2


def test_no_invented_sections_when_ambiguous(tmp_path):
    """Verify section is None when document lacks standard headings (no hallucinations)."""
    pdf_path = tmp_path / "plain_text_paper.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    # Content without recognized headings
    for i in range(15):
        page.insert_text((50, 50 + i * 25), f"Paragraph {i}: Some regular academic discourse without any headings " * 3)
    doc.save(pdf_path)
    doc.close()

    parsed = extract_text_from_pdf(pdf_path, paper_id="2401.00002")

    assert parsed.total_pages == 1
    assert parsed.pages[0].detected_section is None

    chunks = chunk_parsed_document(parsed, chunk_size_words=100, chunk_overlap_words=20)
    assert len(chunks) > 0
    # Every chunk must have section=None rather than an invented heading
    for chunk in chunks:
        assert chunk["metadata"]["section"] is None


def test_empty_or_scanned_pdf_raises_error(tmp_path):
    """Verify empty pages or scanned images without text raise PDFParsingError."""
    pdf_path = tmp_path / "scanned_or_empty.pdf"
    doc = pymupdf.open()
    # Create blank pages without text layer
    doc.new_page()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()

    with pytest.raises(PDFParsingError) as exc_info:
        extract_text_from_pdf(pdf_path, paper_id="2401.00003")

    assert "contains no extractable text" in str(exc_info.value)


def test_chunking_metadata_and_window_span(tmp_path):
    """Verify chunking preserves metadata, page span, and token count."""
    pdf_path = tmp_path / "chunk_test.pdf"
    doc = pymupdf.open()

    # Page 1: 300 words
    page1 = doc.new_page()
    page1.insert_textbox(pymupdf.Rect(50, 50, 500, 700), "1. Introduction\n" + ("word " * 300))

    # Page 2: 300 words
    page2 = doc.new_page()
    page2.insert_textbox(pymupdf.Rect(50, 50, 500, 700), "2. Methodology\n" + ("anotherword " * 300))

    doc.save(pdf_path)
    doc.close()

    parsed = extract_text_from_pdf(pdf_path, paper_id="2401.00004")
    chunks = chunk_parsed_document(parsed, chunk_size_words=200, chunk_overlap_words=50)

    assert len(chunks) > 1

    # Check structure of each chunk
    for idx, chunk in enumerate(chunks):
        assert "text" in chunk
        meta = chunk["metadata"]
        assert meta["paper_id"] == "2401.00004"
        assert meta["chunk_id"] == f"2401.00004_c{idx:04d}"
        assert 1 <= meta["page_start"] <= meta["page_end"] <= 2
        assert meta["token_count"] <= 200
        assert meta["section"] in ("Introduction", "Methodology", None)


def test_chunking_invalid_overlap_raises():
    """Verify chunk_overlap_words >= chunk_size_words raises ValueError."""
    pdf_path = Path("dummy.pdf")
    from src.utils.pdf_parser import ParsedDocument
    doc = ParsedDocument(paper_id="test", total_pages=1, pages=[], full_text="")

    with pytest.raises(ValueError):
        chunk_parsed_document(doc, chunk_size_words=100, chunk_overlap_words=100)
