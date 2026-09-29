"""PDF text extraction, section detection, and chunking using PyMuPDF.

Provides:
- Page-aware text extraction
- Detection of empty/scanned PDFs
- Best-effort section header detection (strictly no hallucinated sections; section=None if ambiguous)
- Overlapping token/word chunking preserving page numbers and section metadata
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import pymupdf

from src.graph.state import ChunkMetadata, DocumentChunk

logger = logging.getLogger("arxiv_agent.pdf_parser")


class PDFParsingError(RuntimeError):
    """Raised when PDF text extraction fails or PDF contains no extractable text."""


@dataclass
class ParsedPage:
    """Represents text and metadata extracted from a single PDF page."""
    page_number: int  # 1-indexed
    text: str
    detected_section: Optional[str] = None


@dataclass
class ParsedDocument:
    """Full parsed representation of a PDF document."""
    paper_id: str
    total_pages: int
    pages: List[ParsedPage]
    full_text: str


# Regex pattern for common academic section headers (numbers optional: e.g. "1. Introduction", "3.1 Approach")
SECTION_PATTERN = re.compile(
    r"^(?:(?:[0-9]{1,2}(?:\.[0-9]{1,2})*|[IVXLCDM]+)[\.\:\)]?\s+)?(Abstract|Introduction|Related\s+Work|Background|Methodology|Method|Methods|Approach|Architecture|System\s+Model|Experimental\s+Setup|Experiments|Results|Evaluation|Discussion|Limitations|Conclusion|Conclusions|Broader\s+Impacts?|Acknowledgements?|References|Appendix(?:\s+[A-Z0-9]+)?)\b",
    re.IGNORECASE,
)


def _detect_heading(line: str) -> Optional[str]:
    """Check if a line matches a recognized academic section heading.

    Returns the normalized heading title or None.
    Never invents names.
    """
    clean_line = line.strip()
    # Heading lines are typically short
    if not clean_line or len(clean_line) > 80:
        return None

    # Headings generally do not end with a sentence-ending period or comma
    if clean_line.endswith((".", ",", ";", ":")):
        # Only allow colon if it's right after section name (e.g., "1. Introduction:")
        clean_line = clean_line.rstrip(".:,;")

    match = SECTION_PATTERN.match(clean_line)
    if match:
        heading_name = match.group(1).title()
        return heading_name

    return None


def extract_text_from_pdf(pdf_path: Path, paper_id: str) -> ParsedDocument:
    """Extract text from PDF pages and perform best-effort section detection.

    Args:
        pdf_path: Local filesystem path to the PDF.
        paper_id: arXiv identifier for the paper.

    Returns:
        ParsedDocument containing per-page text and metadata.

    Raises:
        PDFParsingError: If file cannot be read or contains no extractable text.
    """
    if not pdf_path.exists():
        raise PDFParsingError(f"PDF file does not exist: {pdf_path}")

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        raise PDFParsingError(f"Failed to open PDF {pdf_path}: {e}") from e

    total_pages = doc.page_count
    if total_pages == 0:
        doc.close()
        raise PDFParsingError(f"PDF {pdf_path} contains 0 pages.")

    pages: List[ParsedPage] = []
    total_extracted_chars = 0
    current_section: Optional[str] = None

    for page_idx in range(total_pages):
        page = doc[page_idx]
        page_num = page_idx + 1  # 1-indexed

        try:
            page_text = page.get_text("text")
        except Exception as e:
            logger.warning("Failed to extract text from page %d of %s: %s", page_num, paper_id, e)
            page_text = ""

        # Check for section headings within the page
        lines = page_text.splitlines()
        page_section = current_section
        for line in lines:
            detected = _detect_heading(line)
            if detected:
                current_section = detected
                page_section = detected
                logger.debug("Page %d: detected section '%s'", page_num, detected)

        total_extracted_chars += len(page_text.strip())
        pages.append(
            ParsedPage(
                page_number=page_num,
                text=page_text,
                detected_section=page_section,
            )
        )

    doc.close()

    # Detect empty/scanned PDFs
    # If the average characters per page is suspiciously low (< 50 chars), it's likely scanned images
    if total_extracted_chars < 100 or (total_extracted_chars / total_pages < 50):
        raise PDFParsingError(
            f"PDF '{pdf_path.name}' contains no extractable text ({total_extracted_chars} characters found across {total_pages} pages). "
            "This document appears to be a scanned image or rasterized PDF without an embedded text layer."
        )

    full_text = "\n\n".join(p.text for p in pages if p.text.strip())
    logger.info(
        "Successfully extracted %d characters from %d pages for paper '%s'",
        len(full_text),
        total_pages,
        paper_id,
    )

    return ParsedDocument(
        paper_id=paper_id,
        total_pages=total_pages,
        pages=pages,
        full_text=full_text,
    )


def chunk_parsed_document(
    doc: ParsedDocument,
    chunk_size_words: int = 500,
    chunk_overlap_words: int = 100,
) -> List[DocumentChunk]:
    """Chunk a parsed document into overlapping windows while preserving page and section metadata.

    Args:
        doc: ParsedDocument instance.
        chunk_size_words: Target word count per chunk (~500 words is ~650 tokens).
        chunk_overlap_words: Overlap word count between consecutive chunks.

    Returns:
        List of DocumentChunk dicts conforming to state.py schema.
    """
    if chunk_overlap_words >= chunk_size_words:
        raise ValueError("chunk_overlap_words must be smaller than chunk_size_words.")

    # Flatten pages into word-level tokens tagged with (word, page_num, section)
    tagged_words = []
    for page in doc.pages:
        words = page.text.split()
        for word in words:
            tagged_words.append((word, page.page_number, page.detected_section))

    if not tagged_words:
        return []

    chunks: List[DocumentChunk] = []
    step = chunk_size_words - chunk_overlap_words
    chunk_idx = 0

    for start_idx in range(0, len(tagged_words), step):
        end_idx = min(start_idx + chunk_size_words, len(tagged_words))
        window = tagged_words[start_idx:end_idx]

        if not window:
            break

        chunk_text = " ".join(item[0] for item in window)
        page_start = window[0][1]
        page_end = window[-1][1]

        # Determine section: if all words in window share section or take majority section
        sections = [item[2] for item in window if item[2] is not None]
        if sections:
            # Most frequent detected section in this window
            chunk_section = max(set(sections), key=sections.count)
        else:
            chunk_section = None

        chunk_id = f"{doc.paper_id}_c{chunk_idx:04d}"
        word_count = len(window)

        metadata: ChunkMetadata = {
            "chunk_id": chunk_id,
            "paper_id": doc.paper_id,
            "page_start": page_start,
            "page_end": page_end,
            "section": chunk_section,
            "word_count": word_count,
            "token_count": word_count,
        }

        chunks.append({
            "text": chunk_text,
            "metadata": metadata,
        })

        chunk_idx += 1

        # Stop if we reached the end of document
        if end_idx >= len(tagged_words):
            break

    logger.info("Generated %d chunks for paper '%s'", len(chunks), doc.paper_id)
    return chunks
