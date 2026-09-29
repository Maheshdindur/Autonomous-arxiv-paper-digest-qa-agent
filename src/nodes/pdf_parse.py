"""Node: PDF Parsing and Text Chunking.

Extracts text from the downloaded paper PDF using PyMuPDF, conducts
best-effort section detection, generates overlapping chunks with metadata,
and updates the shared AgentState.
"""

import logging
from pathlib import Path
from typing import Any, Dict

from src.graph.state import AgentState
from src.utils.pdf_parser import PDFParsingError, chunk_parsed_document, extract_text_from_pdf

logger = logging.getLogger("arxiv_agent.nodes.pdf_parse")


def pdf_parse_node(state: AgentState) -> Dict[str, Any]:
    """Parse PDF file, extract text, and generate metadata-rich chunks.

    Updates:
    - raw_text: Full extracted string across all pages
    - chunks: List of DocumentChunk dicts with page and section metadata
    - status: 'parsing_pdf' (or 'error' if parsing fails)
    - error: Error message if PDF is empty, scanned, or invalid
    """
    if state.get("status") == "error":
        return {}

    pdf_path_str = state.get("pdf_path")
    if not pdf_path_str:
        return {
            "error": "Cannot parse PDF: No pdf_path found in state.",
            "status": "error",
        }

    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        return {
            "error": f"Cannot parse PDF: File does not exist at {pdf_path}",
            "status": "error",
        }

    selected_paper = state.get("selected_paper") or {}
    paper_id = selected_paper.get("arxiv_id") or state.get("arxiv_id") or "unknown_paper"

    logger.info("Parsing PDF for paper '%s' from %s", paper_id, pdf_path)

    try:
        parsed_doc = extract_text_from_pdf(pdf_path, paper_id=paper_id)
        chunks = chunk_parsed_document(
            doc=parsed_doc,
            chunk_size_words=500,
            chunk_overlap_words=100,
        )

        return {
            "raw_text": parsed_doc.full_text,
            "chunks": chunks,
            "status": "parsing_pdf",
            "error": None,
        }

    except PDFParsingError as e:
        logger.error("PDF text extraction failed: %s", str(e))
        return {
            "error": f"PDF parsing failed: {e}",
            "status": "error",
        }
    except Exception as e:
        logger.error("Unexpected error during PDF parsing: %s", str(e))
        return {
            "error": f"Unexpected error parsing PDF: {e}",
            "status": "error",
        }
