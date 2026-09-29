"""Simple, deterministic candidate paper ranking using TF-IDF term similarity.

Computes cosine similarity between the research topic query and candidate
paper titles and abstracts. Avoids speculative LLM ranking.
"""

import logging
from typing import List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.graph.state import PaperMetadata

logger = logging.getLogger("arxiv_agent.ranking")


def rank_candidates(
    topic: str,
    candidates: List[PaperMetadata],
) -> Tuple[Optional[PaperMetadata], List[Tuple[PaperMetadata, float]]]:
    """Rank candidate papers by TF-IDF cosine similarity against the topic.

    Args:
        topic: The user's research topic query string.
        candidates: Candidate papers retrieved from the arXiv API.

    Returns:
        Tuple of (selected_paper, ranked_candidates_with_scores).
        If candidates list is empty, returns (None, []).
    """
    if not candidates:
        logger.warning("Cannot rank empty candidates list.")
        return None, []

    if len(candidates) == 1:
        logger.info("Only one candidate retrieved; selecting it directly.")
        return candidates[0], [(candidates[0], 1.0)]

    # Prepare document texts: title is weighted by repeating it to give it higher relevance
    candidate_docs = [
        f"{c.get('title', '')} {c.get('title', '')} {c.get('abstract', '')}"
        for c in candidates
    ]

    all_texts = [topic] + candidate_docs

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 2),
        )
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        # Query is at index 0, candidates are at indices 1..N
        query_vec = tfidf_matrix[0:1]
        doc_vecs = tfidf_matrix[1:]

        similarities = cosine_similarity(query_vec, doc_vecs)[0]

        scored_candidates = [
            (candidates[i], float(similarities[i]))
            for i in range(len(candidates))
        ]

        # Sort descending by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        top_candidate, top_score = scored_candidates[0]
        logger.info(
            "Selected top paper '%s' (arXiv ID: %s) with relevance score: %.4f",
            top_candidate.get("title"),
            top_candidate.get("arxiv_id"),
            top_score,
        )

        return top_candidate, scored_candidates

    except Exception as e:
        logger.error("TF-IDF ranking encountered error: %s. Falling back to first candidate.", str(e))
        # Deterministic fallback: preserve original arXiv relevance order
        fallback_scored = [(c, 0.0) for c in candidates]
        return candidates[0], fallback_scored
