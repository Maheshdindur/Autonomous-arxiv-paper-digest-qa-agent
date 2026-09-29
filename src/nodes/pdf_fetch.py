"""Node: PDF Fetch and Download Verification.

Downloads the target paper PDF from arXiv and conducts structural integrity
verification using PyMuPDF before passing to downstream nodes.
"""

import logging
from pathlib import Path
from typing import Any, Dict

from src.config import load_config
from src.graph.state import AgentState
from src.utils.pdf_fetcher import PDFDownloadError, download_and_verify_pdf

logger = logging.getLogger("arxiv_agent.nodes.pdf_fetch")


def pdf_fetch_node(state: AgentState) -> Dict[str, Any]:
    """Download and verify PDF for the selected paper.

    Updates:
    - pdf_path: Local filesystem path to the verified PDF file
    - status: 'fetching_pdf' (or 'error' if download/verification fails)
    - error: Detailed error message if download or verification fails
    """
    if state.get("status") == "error":
        return {}

    selected_paper = state.get("selected_paper")
    if not selected_paper:
        return {
            "error": "Cannot download PDF: No paper selected.",
            "status": "error",
        }

    arxiv_id = selected_paper.get("arxiv_id")
    pdf_url = selected_paper.get("pdf_url")

    if not pdf_url:
        return {
            "error": f"Selected paper '{arxiv_id}' has no PDF URL.",
            "status": "error",
        }

    # Load download directory configuration
    config = load_config(require_llm_key=False)
    target_dir = config.pdf_download_dir

    try:
        pdf_path = download_and_verify_pdf(
            pdf_url=pdf_url,
            arxiv_id=arxiv_id,
            target_dir=target_dir,
        )
        return {
            "pdf_path": str(pdf_path),
            "status": "fetching_pdf",
            "error": None,
        }
    except PDFDownloadError as e:
        logger.error("PDF download/verification failed: %s", str(e))
        return {
            "error": f"Failed to download or verify PDF: {e}",
            "status": "error",
        }
