"""Node: arXiv Retrieval and Candidate Ranking.

Handles paper metadata lookup (for paper queries) or candidate search and
TF-IDF ranking (for topic queries). Populates candidate_papers and selected_paper.
"""

import logging
from typing import Any, Dict

from src.graph.state import AgentState
from src.utils.arxiv_api import ArxivAPIError, fetch_paper_by_id, search_papers_by_topic
from src.utils.ranking import rank_candidates

logger = logging.getLogger("arxiv_agent.nodes.arxiv_retrieval")


def arxiv_retrieval_node(state: AgentState) -> Dict[str, Any]:
    """Retrieve metadata from arXiv API and select the target paper.

    Updates:
    - candidate_papers: List of candidate metadata dicts
    - selected_paper: The chosen PaperMetadata dict
    - arxiv_id: Normalized arXiv ID of the selected paper
    - status: 'retrieving' (or 'error' if retrieval fails)
    - error: Detailed error message if paper not found or search returns 0 results
    """
    # Guard against prior errors
    if state.get("status") == "error":
        return {}

    query_type = state.get("query_type")

    # Path A: Specific paper lookup
    if query_type == "paper":
        arxiv_id = state.get("arxiv_id")
        if not arxiv_id:
            return {
                "error": "Missing arXiv identifier for paper lookup.",
                "status": "error",
            }

        logger.info("Looking up metadata for arXiv ID: %s", arxiv_id)
        try:
            paper_meta = fetch_paper_by_id(arxiv_id)
            if not paper_meta:
                return {
                    "error": f"Paper with arXiv ID '{arxiv_id}' was not found on arXiv.",
                    "status": "error",
                }

            return {
                "candidate_papers": [paper_meta],
                "selected_paper": paper_meta,
                "arxiv_id": paper_meta["arxiv_id"],
                "status": "retrieving",
                "error": None,
            }
        except ArxivAPIError as e:
            return {
                "error": f"arXiv API error: {e}",
                "status": "error",
            }

    # Path B: Natural language topic search
    elif query_type == "topic":
        topic = state.get("topic")
        if not topic:
            return {
                "error": "Missing research topic for arXiv search.",
                "status": "error",
            }

        logger.info("Searching arXiv for topic: '%s'", topic)
        try:
            candidates = search_papers_by_topic(topic, max_results=5)
            if not candidates:
                return {
                    "error": f"No papers found on arXiv matching topic: '{topic}'. Try different search terms.",
                    "status": "error",
                    "candidate_papers": [],
                    "selected_paper": None,
                }

            # Rank candidate papers deterministically using TF-IDF term overlap
            selected_paper, ranked = rank_candidates(topic, candidates)
            logger.info("Selected top paper: '%s' (%s)", selected_paper["title"], selected_paper["arxiv_id"])

            return {
                "candidate_papers": candidates,
                "selected_paper": selected_paper,
                "arxiv_id": selected_paper["arxiv_id"],
                "status": "retrieving",
                "error": None,
            }
        except ArxivAPIError as e:
            return {
                "error": f"arXiv API error: {e}",
                "status": "error",
            }

    else:
        return {
            "error": f"Unknown query_type: '{query_type}'. Expected 'paper' or 'topic'.",
            "status": "error",
        }
