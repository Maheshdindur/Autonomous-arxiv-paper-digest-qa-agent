"""PDF download and integrity verification.

Downloads paper PDFs from arXiv, verifying:
1. HTTP request success (200 OK)
2. Response content type or magic bytes (%PDF-)
3. Non-empty file on disk (> 1 KB)
4. Document integrity check with PyMuPDF (can be opened and page_count > 0)
"""

import logging
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional
import pymupdf

logger = logging.getLogger("arxiv_agent.pdf_fetcher")


class PDFDownloadError(RuntimeError):
    """Raised when PDF download or verification fails."""


def sanitize_paper_id_for_filename(arxiv_id: str) -> str:
    """Replace slashes and unsafe characters in arXiv ID for local filename."""
    return arxiv_id.replace("/", "_").replace(":", "_").strip()


def verify_pdf_file(file_path: Path) -> bool:
    """Verify that a downloaded file is a valid, readable PDF.

    Checks:
    - File exists
    - File size > 1024 bytes
    - Starts with PDF magic bytes (%PDF-)
    - Can be opened by PyMuPDF with page_count > 0
    """
    if not file_path.exists():
        return False

    size = file_path.stat().st_size
    if size < 1024:
        logger.error("Downloaded file %s is suspiciously small (%d bytes).", file_path, size)
        return False

    # Check magic bytes
    with open(file_path, "rb") as f:
        magic_bytes = f.read(5)
        if magic_bytes != b"%PDF-":
            logger.error("File %s does not begin with %%PDF- magic bytes (got %s).", file_path, magic_bytes)
            return False

    # Verify readable structure with PyMuPDF
    try:
        doc = pymupdf.open(file_path)
        page_count = doc.page_count
        doc.close()
        if page_count < 1:
            logger.error("PDF %s contains zero pages.", file_path)
            return False
        return True
    except Exception as e:
        logger.error("PyMuPDF failed to parse %s: %s", file_path, str(e))
        return False


def download_and_verify_pdf(
    pdf_url: str,
    arxiv_id: str,
    target_dir: Path,
    timeout_seconds: int = 30,
) -> Path:
    """Download PDF from URL and verify integrity.

    Args:
        pdf_url: Direct link to PDF.
        arxiv_id: arXiv identifier.
        target_dir: Directory where the PDF should be stored.
        timeout_seconds: Request timeout in seconds.

    Returns:
        Path to the verified PDF file on disk.

    Raises:
        PDFDownloadError: If download fails or the file fails integrity checks.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_filename = f"{sanitize_paper_id_for_filename(arxiv_id)}.pdf"
    target_file = target_dir / safe_filename

    # If already downloaded and valid, reuse
    if target_file.exists() and verify_pdf_file(target_file):
        logger.info("Reusing existing verified PDF for %s at %s", arxiv_id, target_file)
        return target_file

    logger.info("Downloading PDF for arXiv ID '%s' from %s", arxiv_id, pdf_url)

    req = urllib.request.Request(
        pdf_url,
        headers={
            "User-Agent": "Autonomous-ArXiv-Digest-QA-Agent/1.0 (academic research evaluation)"
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            status_code = response.getcode()
            if status_code != 200:
                raise PDFDownloadError(
                    f"HTTP request failed with status code {status_code} for URL {pdf_url}"
                )

            content_type = response.headers.get("Content-Type", "")
            # Read first chunk to check magic bytes
            initial_chunk = response.read(1024)
            if not initial_chunk.startswith(b"%PDF-"):
                raise PDFDownloadError(
                    f"Server returned non-PDF content (Content-Type: '{content_type}', magic bytes: {initial_chunk[:10]}). "
                    "This usually indicates an arXiv rate-limit, captcha, or non-existent paper."
                )

            # Write file
            with open(target_file, "wb") as f_out:
                f_out.write(initial_chunk)
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    f_out.write(chunk)

    except urllib.error.HTTPError as e:
        if target_file.exists():
            target_file.unlink(missing_ok=True)
        raise PDFDownloadError(f"HTTP error {e.code} while downloading {pdf_url}: {e.reason}") from e
    except urllib.error.URLError as e:
        if target_file.exists():
            target_file.unlink(missing_ok=True)
        raise PDFDownloadError(f"Network error while connecting to {pdf_url}: {e.reason}") from e
    except Exception as e:
        if target_file.exists():
            target_file.unlink(missing_ok=True)
        raise PDFDownloadError(f"Unexpected error downloading PDF from {pdf_url}: {e}") from e

    # Perform full structural verification
    if not verify_pdf_file(target_file):
        if target_file.exists():
            target_file.unlink(missing_ok=True)
        raise PDFDownloadError(
            f"Downloaded file for arXiv ID '{arxiv_id}' failed PDF integrity verification."
        )

    logger.info("Successfully downloaded and verified PDF at %s", target_file)
    return target_file
