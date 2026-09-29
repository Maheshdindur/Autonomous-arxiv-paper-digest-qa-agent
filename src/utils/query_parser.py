"""Query parsing and normalization for arXiv identifiers, URLs, and topics.

Supports:
- Modern arXiv IDs: e.g. 2401.12345, 2401.12345v1, 1706.03762
- Legacy arXiv IDs: e.g. hep-th/9901001, cs/0101001, math.PR/0101001
- arXiv URLs: https://arxiv.org/abs/2401.12345, https://arxiv.org/pdf/hep-th/9901001.pdf
- Natural language research topics
"""

import re
from dataclasses import dataclass
from typing import Literal, Optional

# Modern identifier pattern: YYMM.NNNNN or YYMM.NNNN with optional version vN
MODERN_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$")

# Legacy identifier pattern: category/YYMMNNN (e.g. hep-th/9901001, cs/0101001, math.PR/0101001)
LEGACY_ID_PATTERN = re.compile(r"^[a-zA-Z\-]+(?:\.[a-zA-Z\-]+)?/\d{7}(?:v\d+)?$")

# URL pattern matching arxiv.org abs or pdf links
ARXIV_URL_PATTERN = re.compile(
    r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/([a-zA-Z0-9\.\-\/]+?)(?:\.pdf)?/?(?:[?#].*)?$",
    re.IGNORECASE,
)


@dataclass
class ParsedQuery:
    """Result of classifying and normalizing a user query."""
    query_type: Literal["paper", "topic"]
    raw_input: str
    arxiv_id: Optional[str] = None
    canonical_url: Optional[str] = None
    topic: Optional[str] = None


def normalize_arxiv_id(raw_id: str) -> Optional[str]:
    """Validate and normalize an arXiv ID string.

    Returns the normalized ID (preserving version if provided) or None if invalid.
    """
    clean = raw_id.strip()
    # Strip optional 'arxiv:' or 'arXiv:' prefix
    if clean.lower().startswith("arxiv:"):
        clean = clean[6:].strip()

    if MODERN_ID_PATTERN.match(clean) or LEGACY_ID_PATTERN.match(clean):
        return clean
    return None


def parse_user_query(user_input: str) -> ParsedQuery:
    """Classify user input as either a specific paper lookup or a research topic search.

    Args:
        user_input: Raw input string from CLI.

    Returns:
        ParsedQuery with query_type 'paper' (with arxiv_id and canonical_url)
        or 'topic' (with topic text).
    """
    cleaned = user_input.strip()

    # 1. Check for arXiv URL
    url_match = ARXIV_URL_PATTERN.match(cleaned)
    if url_match:
        extracted_id = url_match.group(1).rstrip("/")
        normalized = normalize_arxiv_id(extracted_id)
        if normalized:
            return ParsedQuery(
                query_type="paper",
                raw_input=cleaned,
                arxiv_id=normalized,
                canonical_url=f"https://arxiv.org/abs/{normalized}",
            )

    # 2. Check for direct arXiv ID (modern or legacy)
    direct_id = normalize_arxiv_id(cleaned)
    if direct_id:
        return ParsedQuery(
            query_type="paper",
            raw_input=cleaned,
            arxiv_id=direct_id,
            canonical_url=f"https://arxiv.org/abs/{direct_id}",
        )

    # 3. Otherwise treat as natural-language research topic
    return ParsedQuery(
        query_type="topic",
        raw_input=cleaned,
        topic=cleaned,
    )
