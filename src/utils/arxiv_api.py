"""Official arXiv API client wrapper.

Interacts with the official arXiv Atom feed API via the `arxiv` Python package.
Retrieves metadata and formats it into structured PaperMetadata dictionaries.
No web scraping is used.
"""

import logging
from typing import Any, List, Optional
import arxiv

from src.graph.state import PaperMetadata

logger = logging.getLogger("arxiv_agent.arxiv_api")


class ArxivAPIError(RuntimeError):
    """Raised when arXiv API queries fail or return unexpected responses."""


def _format_arxiv_result(result: arxiv.Result) -> PaperMetadata:
    """Transform an arxiv.Result object into a structured PaperMetadata dict."""
    # Ensure PDF URL uses https
    pdf_url = result.pdf_url or f"https://arxiv.org/pdf/{result.get_short_id()}.pdf"
    if pdf_url.startswith("http://"):
        pdf_url = "https://" + pdf_url[7:]

    # Clean whitespace and newlines from title and abstract
    title = " ".join(result.title.strip().split())
    abstract = " ".join(result.summary.strip().split())
    authors = [author.name for author in result.authors]

    return {
        "arxiv_id": result.get_short_id(),
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "published": result.published.isoformat() if result.published else "",
        "updated": result.updated.isoformat() if result.updated else None,
        "pdf_url": pdf_url,
        "entry_id": result.entry_id,
        "primary_category": result.primary_category,
        "categories": list(result.categories),
        "comment": result.comment,
        "journal_ref": result.journal_ref,
        "doi": result.doi,
    }


def fetch_paper_by_id(arxiv_id: str, client: Optional[arxiv.Client] = None) -> Optional[PaperMetadata]:
    """Fetch metadata for a specific arXiv ID using official arXiv API.

    Args:
        arxiv_id: Valid arXiv identifier (modern or legacy).
        client: Optional arxiv.Client instance (for dependency injection/testing).

    Returns:
        PaperMetadata dictionary if found, or None if no matching paper exists.

    Raises:
        ArxivAPIError: If the API call encounters a network or HTTP error.
    """
    if client is None:
        client = arxiv.Client(page_size=1, delay_seconds=3.0, num_retries=3)

    search = arxiv.Search(id_list=[arxiv_id])

    try:
        results = list(client.results(search))
        if not results:
            logger.warning("No paper found for arXiv ID: %s", arxiv_id)
            return None
        return _format_arxiv_result(results[0])
    except Exception as e:
        logger.error("Failed to fetch paper by ID '%s': %s", arxiv_id, str(e))
        raise ArxivAPIError(f"arXiv API error while fetching ID '{arxiv_id}': {e}") from e


def search_papers_by_topic(
    topic: str,
    max_results: int = 5,
    client: Optional[arxiv.Client] = None,
) -> List[PaperMetadata]:
    """Search for candidate papers matching a natural-language research topic.

    Args:
        topic: Natural-language research topic.
        max_results: Maximum candidate papers to retrieve (default 5).
        client: Optional arxiv.Client instance.

    Returns:
        List of PaperMetadata dictionaries. May be empty if zero results found.

    Raises:
        ArxivAPIError: If the API call encounters an error.
    """
    if client is None:
        client = arxiv.Client(page_size=max_results, delay_seconds=3.0, num_retries=3)

    # Use relevance sorting for topic queries
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    try:
        results = list(client.results(search))
        candidates = [_format_arxiv_result(r) for r in results]
        logger.info("Found %d candidate papers for topic '%s'", len(candidates), topic)
        return candidates
    except Exception as e:
        logger.error("Failed to search arXiv for topic '%s': %s", topic, str(e))
        raise ArxivAPIError(f"arXiv API search error for topic '{topic}': {e}") from e
