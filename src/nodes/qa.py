"""LangGraph nodes for Grounded Question Answering (QA).

Separated cleanly from CLI:
- qa_retrieval_node: retrieves top-K evidence chunks from Chroma
- qa_answer_node: generates grounded response citing evidence and updating conversation history
"""

import logging
from typing import Any, Dict

from src.graph.state import AgentState, QAExchange
from src.utils.qa_engine import generate_grounded_answer, retrieve_qa_evidence

logger = logging.getLogger("arxiv_agent.nodes.qa")


def qa_retrieval_node(state: AgentState) -> Dict[str, Any]:
    """Retrieve top-K semantic chunks from Chroma for the current question."""
    if state.get("status") == "error":
        return {}

    question = state.get("current_question")
    if not question or not question.strip():
        return {
            "error": "Cannot retrieve evidence: No question provided in state.",
            "status": "error",
        }

    # Resolve target paper ID
    paper_meta = state.get("selected_paper")
    paper_id = paper_meta.get("arxiv_id") if paper_meta else state.get("arxiv_id")
    if not paper_id:
        return {
            "error": "Cannot retrieve evidence: No paper ID identified in state.",
            "status": "error",
        }

    logger.info("Executing QA retrieval for paper '%s', question: '%s'", paper_id, question)
    try:
        chunks = retrieve_qa_evidence(paper_id=paper_id, question=question, top_k=4)
        return {
            "retrieved_chunks": chunks,
            "status": "qa_answering",
            "error": None,
        }
    except Exception as e:
        logger.error("QA retrieval node failed: %s", str(e))
        return {
            "error": f"QA retrieval failed: {e}",
            "status": "error",
        }


def qa_answer_node(state: AgentState) -> Dict[str, Any]:
    """Generate grounded answer from retrieved chunks and update conversation history."""
    if state.get("status") == "error":
        return {}

    question = state.get("current_question")
    if not question:
        return {
            "error": "Cannot generate answer: No question provided in state.",
            "status": "error",
        }

    retrieved_chunks = state.get("retrieved_chunks") or []
    paper_meta = state.get("selected_paper")
    history = state.get("conversation_history") or []

    logger.info("Generating grounded answer for question: '%s' with %d chunks", question, len(retrieved_chunks))
    try:
        result = generate_grounded_answer(
            paper_meta=paper_meta,
            question=question,
            evidence_chunks=retrieved_chunks,
            conversation_history=history,
        )

        answer = result["answer"]
        sources = result["sources"]

        # Record QA exchange into stateful conversation history
        exchange: QAExchange = {
            "question": question,
            "answer": answer,
            "sources": sources,
        }
        updated_history = list(history) + [exchange]

        return {
            "qa_answer": answer,
            "conversation_history": updated_history,
            "status": "completed",
            "error": None,
        }
    except Exception as e:
        logger.error("QA answer generation failed: %s", str(e))
        return {
            "error": f"QA answer generation failed: {e}",
            "status": "error",
        }
