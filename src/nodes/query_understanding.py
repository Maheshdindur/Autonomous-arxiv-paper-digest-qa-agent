"""Node: Query Understanding.

Parses raw user input to determine whether it is an arXiv ID, an arXiv URL,
or a research topic. Normalizes identifiers and sets the initial state fields.
"""

import logging
from typing import Any, Dict

from src.graph.state import AgentState
from src.utils.query_parser import parse_user_query

logger = logging.getLogger("arxiv_agent.nodes.query_understanding")


def query_understanding_node(state: AgentState) -> Dict[str, Any]:
    """Parse user query and classify intent.

    Updates:
    - query_type: 'paper' or 'topic'
    - arxiv_id: canonical ID if 'paper'
    - arxiv_url: canonical URL if 'paper'
    - topic: topic text if 'topic'
    - status: 'understanding_query' (or 'error' if empty)
    """
    user_input = state.get("user_input", "").strip()
    if not user_input:
        logger.error("Empty user query received.")
        return {
            "error": "Query cannot be empty. Please provide an arXiv ID, URL, or research topic.",
            "status": "error",
        }

    parsed = parse_user_query(user_input)
    logger.info("Classified query '%s' as type: %s", user_input, parsed.query_type)

    if parsed.query_type == "paper":
        return {
            "query_type": "paper",
            "arxiv_id": parsed.arxiv_id,
            "arxiv_url": parsed.canonical_url,
            "topic": None,
            "status": "understanding_query",
            "error": None,
        }
    else:
        return {
            "query_type": "topic",
            "arxiv_id": None,
            "arxiv_url": None,
            "topic": parsed.topic,
            "status": "understanding_query",
            "error": None,
        }
