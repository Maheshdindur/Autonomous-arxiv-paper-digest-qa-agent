"""Unit tests for TF-IDF candidate paper ranking."""

from src.graph.state import PaperMetadata
from src.utils.ranking import rank_candidates


def _make_candidate(arxiv_id: str, title: str, abstract: str) -> PaperMetadata:
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": ["Author One"],
        "abstract": abstract,
        "published": "2024-01-01T00:00:00Z",
        "updated": None,
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        "entry_id": f"http://arxiv.org/abs/{arxiv_id}",
        "primary_category": "cs.CL",
        "categories": ["cs.CL"],
        "comment": None,
        "journal_ref": None,
        "doi": None,
    }


def test_rank_empty_candidates():
    """Verify ranking empty candidate list returns None and empty scores."""
    selected, scored = rank_candidates("KV-cache compression", [])
    assert selected is None
    assert scored == []


def test_rank_single_candidate():
    """Verify single candidate is selected directly."""
    c = _make_candidate("2401.00001", "A Paper", "Abstract content")
    selected, scored = rank_candidates("A Paper", [c])
    assert selected["arxiv_id"] == "2401.00001"
    assert len(scored) == 1


def test_rank_candidates_selects_most_relevant():
    """Verify TF-IDF scoring prioritizes relevant title and abstract matches."""
    topic = "KV-cache compression for large language models"
    c_irrelevant = _make_candidate(
        "2401.00001",
        "Observational Studies of Supernovae",
        "We analyze optical light curves of Type Ia supernovae.",
    )
    c_relevant = _make_candidate(
        "2401.00002",
        "Efficient KV-Cache Compression for Large Language Models",
        "This paper proposes novel KV-cache compression techniques for transformer LLMs.",
    )
    c_partial = _make_candidate(
        "2401.00003",
        "Memory Footprint Reduction in Neural Networks",
        "We discuss quantization approaches for general deep learning architectures.",
    )

    selected, scored = rank_candidates(topic, [c_irrelevant, c_relevant, c_partial])

    assert selected["arxiv_id"] == "2401.00002"
    assert scored[0][0]["arxiv_id"] == "2401.00002"
    # Ensure relevant paper score is strictly higher than irrelevant
    assert scored[0][1] > scored[1][1]
    assert scored[0][1] > scored[2][1]
