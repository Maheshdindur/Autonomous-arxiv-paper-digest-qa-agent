"""Unit tests for query parsing and arXiv identifier normalization."""

from src.utils.query_parser import normalize_arxiv_id, parse_user_query


def test_normalize_modern_arxiv_ids():
    """Verify modern arXiv ID patterns (YYMM.NNNNN, YYMM.NNNN, and versioning)."""
    assert normalize_arxiv_id("2401.12345") == "2401.12345"
    assert normalize_arxiv_id("2401.12345v1") == "2401.12345v1"
    assert normalize_arxiv_id("1706.03762") == "1706.03762"
    assert normalize_arxiv_id("arXiv:2401.12345") == "2401.12345"
    assert normalize_arxiv_id("  arxiv:1706.03762v2  ") == "1706.03762v2"


def test_normalize_legacy_arxiv_ids():
    """Verify legacy arXiv ID patterns (category/YYMMNNN)."""
    assert normalize_arxiv_id("hep-th/9901001") == "hep-th/9901001"
    assert normalize_arxiv_id("cs/0101001") == "cs/0101001"
    assert normalize_arxiv_id("math.PR/0101001") == "math.PR/0101001"
    assert normalize_arxiv_id("arXiv:quant-ph/0201001v1") == "quant-ph/0201001v1"


def test_parse_arxiv_urls():
    """Verify URL extraction and canonicalization for abs and pdf links."""
    parsed1 = parse_user_query("https://arxiv.org/abs/2401.12345")
    assert parsed1.query_type == "paper"
    assert parsed1.arxiv_id == "2401.12345"
    assert parsed1.canonical_url == "https://arxiv.org/abs/2401.12345"

    parsed2 = parse_user_query("https://arxiv.org/pdf/2401.12345.pdf")
    assert parsed2.query_type == "paper"
    assert parsed2.arxiv_id == "2401.12345"

    parsed3 = parse_user_query("http://arxiv.org/abs/hep-th/9901001")
    assert parsed3.query_type == "paper"
    assert parsed3.arxiv_id == "hep-th/9901001"


def test_parse_direct_paper_ids():
    """Verify direct modern and legacy IDs are classified as paper queries."""
    parsed_modern = parse_user_query("2401.12345")
    assert parsed_modern.query_type == "paper"
    assert parsed_modern.arxiv_id == "2401.12345"

    parsed_legacy = parse_user_query("cs/0101001")
    assert parsed_legacy.query_type == "paper"
    assert parsed_legacy.arxiv_id == "cs/0101001"


def test_parse_research_topics():
    """Verify natural-language topics are classified as topic queries."""
    topic1 = "KV-cache compression for LLMs"
    parsed1 = parse_user_query(topic1)
    assert parsed1.query_type == "topic"
    assert parsed1.topic == topic1
    assert parsed1.arxiv_id is None

    topic2 = "deep reinforcement learning with memory"
    parsed2 = parse_user_query(topic2)
    assert parsed2.query_type == "topic"
    assert parsed2.topic == topic2
