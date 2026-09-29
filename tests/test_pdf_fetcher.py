"""Unit tests for PDF downloading and integrity verification."""

from pathlib import Path
import pymupdf
import pytest

from src.utils.pdf_fetcher import (
    PDFDownloadError,
    download_and_verify_pdf,
    sanitize_paper_id_for_filename,
    verify_pdf_file,
)


def test_sanitize_paper_id():
    """Verify safe filename sanitization."""
    assert sanitize_paper_id_for_filename("2401.12345") == "2401.12345"
    assert sanitize_paper_id_for_filename("hep-th/9901001") == "hep-th_9901001"
    assert sanitize_paper_id_for_filename("cs/0101001") == "cs_0101001"


def test_verify_pdf_file_non_existent(tmp_path):
    """Verify non-existent file returns False."""
    assert not verify_pdf_file(tmp_path / "does_not_exist.pdf")


def test_verify_pdf_file_too_small(tmp_path):
    """Verify file under 1KB returns False."""
    tiny_file = tmp_path / "tiny.pdf"
    tiny_file.write_bytes(b"%PDF-1.4 tiny content")
    assert not verify_pdf_file(tiny_file)


def test_verify_pdf_file_invalid_magic_bytes(tmp_path):
    """Verify file without %PDF- returns False (e.g. HTML 404 page)."""
    fake_pdf = tmp_path / "fake.pdf"
    # Create file > 1024 bytes containing HTML error
    fake_pdf.write_text("<html><body>404 Not Found</body></html>" * 50)
    assert not verify_pdf_file(fake_pdf)


def test_verify_pdf_file_valid_document(tmp_path):
    """Verify genuine readable PDF with pages returns True."""
    valid_pdf_path = tmp_path / "valid.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Hello World PDF for testing")
    # Pad to ensure size > 1024 bytes
    for i in range(10):
        page.insert_text((50, 100 + i * 20), f"Additional test text line {i} " * 5)
    doc.save(valid_pdf_path)
    doc.close()

    assert valid_pdf_path.stat().st_size > 1024
    assert verify_pdf_file(valid_pdf_path) is True


def test_download_and_verify_reuses_existing_valid_pdf(tmp_path):
    """Verify existing valid PDF is reused without re-downloading."""
    target_dir = tmp_path / "downloads"
    target_dir.mkdir()
    pdf_path = target_dir / "2401.12345.pdf"

    # Create valid PDF
    doc = pymupdf.open()
    page = doc.new_page()
    for i in range(20):
        page.insert_text((50, 50 + i * 20), f"Line {i} content text " * 10)
    doc.save(pdf_path)
    doc.close()

    # Call with a dummy URL; it should reuse the existing file without network call
    result_path = download_and_verify_pdf(
        pdf_url="http://dummy.url/not_called",
        arxiv_id="2401.12345",
        target_dir=target_dir,
    )

    assert result_path == pdf_path
