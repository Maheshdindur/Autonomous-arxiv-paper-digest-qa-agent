"""Node: Executive Briefing Summarization.

Generates the structured executive briefing grounded in retrieved paper evidence.
Adheres strictly to the assignment rubric and anti-hallucination rules.
"""

import logging
from typing import Any, Dict

from src.graph.state import AgentState, ExecutiveBriefing
from src.utils.briefing_extractor import assemble_briefing_evidence, generate_executive_briefing

logger = logging.getLogger("arxiv_agent.nodes.summarize")


def format_briefing_markdown(briefing: ExecutiveBriefing) -> str:
    """Format an ExecutiveBriefing dictionary into readable Markdown."""
    authors_str = ", ".join(briefing.get("authors", []))
    lines = [
        f"# Executive Briefing: {briefing.get('title')}",
        "",
        f"**Authors**: {authors_str}  ",
        f"**arXiv ID**: `{briefing.get('arxiv_id')}`  ",
        f"**Published**: {briefing.get('publish_date')}  ",
        f"**Link**: {briefing.get('link')}",
        "",
        "## Why This Paper Matters",
        briefing.get("why_it_matters", "Not specified."),
        "",
        "## Problem Statement",
        briefing.get("problem_statement", "Not specified."),
        "",
        "## Method & Approach",
    ]

    for bullet in briefing.get("method_approach", []):
        lines.append(f"- {bullet}")

    lines.append("")
    lines.append("## Key Results & Claims")
    for bullet in briefing.get("key_results_claims", []):
        lines.append(f"- {bullet}")

    lines.append("")
    lines.append("## Limitations")
    for bullet in briefing.get("limitations", []):
        lines.append(f"- {bullet}")

    lines.append("")
    lines.append("## Suggested Follow-Up Questions")
    for idx, q in enumerate(briefing.get("suggested_followup_questions", []), 1):
        lines.append(f"{idx}. {q}")

    return "\n".join(lines)


def summarize_node(state: AgentState) -> Dict[str, Any]:
    """Execute executive briefing summarization node.

    Updates:
    - briefing: Structured ExecutiveBriefing dict
    - status: 'completed' (or 'error' if generation fails)
    - error: Detailed error message if LLM fails
    """
    if state.get("status") == "error":
        return {}

    selected_paper = state.get("selected_paper")
    if not selected_paper:
        return {
            "error": "Cannot generate briefing: No paper selected.",
            "status": "error",
        }

    chunks = state.get("chunks") or []
    paper_id = selected_paper.get("arxiv_id", "")
    logger.info("Generating grounded executive briefing for paper: %s", paper_id)

    try:
        # Assemble targeted evidence chunks (abstract, intro, method, results, limitations)
        evidence_chunks = assemble_briefing_evidence(selected_paper, chunks)

        briefing = generate_executive_briefing(
            paper_meta=selected_paper,
            evidence_chunks=evidence_chunks,
        )

        return {
            "briefing": briefing,
            "status": "completed",
            "error": None,
        }
    except Exception as e:
        logger.error("Failed to generate executive briefing: %s", str(e))
        return {
            "error": f"Briefing generation failed: {e}",
            "status": "error",
        }
