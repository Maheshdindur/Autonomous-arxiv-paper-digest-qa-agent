"""Node: Chunk Embedding and Vector Store Indexing.

Generates dense vector embeddings for paper chunks and indexes them into
the local persistent Chroma vector database for downstream retrieval.
"""

import logging
from typing import Any, Dict

from src.graph.state import AgentState
from src.utils.vector_store import index_paper_chunks, sanitize_collection_name

logger = logging.getLogger("arxiv_agent.nodes.chunk_and_index")


def chunk_and_index_node(state: AgentState) -> Dict[str, Any]:
    """Index paper chunks into Chroma vector database.

    Updates:
    - vector_collection_name: Name of the Chroma collection holding paper chunks
    - status: 'indexing' (or 'error' if indexing fails)
    - error: Error message if indexing encounters an issue
    """
    if state.get("status") == "error":
        return {}

    chunks = state.get("chunks") or []
    if not chunks:
        return {
            "error": "Cannot index chunks: No chunks found in state.",
            "status": "error",
        }

    selected_paper = state.get("selected_paper") or {}
    paper_id = selected_paper.get("arxiv_id") or state.get("arxiv_id") or "unknown_paper"

    logger.info("Indexing %d chunks for paper '%s' into vector store...", len(chunks), paper_id)

    try:
        count = index_paper_chunks(paper_id=paper_id, chunks=chunks)
        collection_name = sanitize_collection_name(paper_id)

        return {
            "vector_collection_name": collection_name,
            "status": "indexing",
            "error": None,
        }
    except Exception as e:
        logger.error("Failed to index chunks into vector store: %s", str(e))
        return {
            "error": f"Vector indexing failed: {e}",
            "status": "error",
        }
