"""Grounded Question Answering (QA) engine.

Retrieves top-K evidence chunks from Chroma vector store, formats provenance-preserving
context, and generates strictly grounded answers using Groq LLM with anti-hallucination
constraints and explicit refusal when evidence is insufficient.
"""

import logging
from typing import Any, Dict, List, Optional
import groq

from src.graph.state import DocumentChunk, PaperMetadata, QAExchange
from src.utils.llm import LLMError, call_groq_json
from src.utils.vector_store import query_similar_chunks

logger = logging.getLogger("arxiv_agent.qa_engine")

REFUSAL_MESSAGE = "The provided paper does not contain enough information to answer this question."


def retrieve_qa_evidence(
    paper_id: str,
    question: str,
    top_k: int = 4,
    client: Optional[Any] = None,
    embedding_model: Optional[str] = None,
) -> List[DocumentChunk]:
    """Retrieve top-K most relevant evidence chunks for a user question from Chroma.

    Preserves chunk_id, page_start, page_end, and section metadata.
    """
    if not paper_id or not question.strip():
        return []

    try:
        hits = query_similar_chunks(
            paper_id=paper_id,
            query=question,
            top_k=top_k,
            client=client,
            embedding_model=embedding_model,
        )
    except Exception as e:
        logger.error("Failed to query similar chunks for paper '%s': %s", paper_id, e)
        return []

    chunks: List[DocumentChunk] = []
    for hit in hits:
        chunks.append({
            "text": hit["text"],
            "metadata": hit["metadata"],
        })

    logger.info("Retrieved %d evidence chunks for QA on paper '%s'", len(chunks), paper_id)
    return chunks


def format_qa_context(evidence_chunks: List[DocumentChunk]) -> str:
    """Format evidence chunks with explicit provenance headers."""
    if not evidence_chunks:
        return "No relevant paper evidence found."

    blocks = []
    for idx, c in enumerate(evidence_chunks, 1):
        meta = c["metadata"]
        sec_label = meta.get("section") or "Unlabeled Section"
        pages = f"Page {meta.get('page_start', '?')}-{meta.get('page_end', '?')}"
        cid = meta.get("chunk_id", f"chunk_{idx}")
        blocks.append(
            f"--- EVIDENCE CHUNK {idx} [Chunk ID: {cid} | {pages} | Section: {sec_label}] ---\n"
            f"{c['text']}\n"
        )
    return "\n".join(blocks)


def format_conversation_history(history: Optional[List[QAExchange]], max_turns: int = 3) -> str:
    """Format previous conversation turns for dialogue flow context."""
    if not history:
        return ""

    recent_turns = history[-max_turns:]
    lines = ["PRIOR CONVERSATION HISTORY (FOR CONVERSATIONAL FLOW ONLY; NOT PRIMARY EVIDENCE):"]
    for idx, turn in enumerate(recent_turns, 1):
        lines.append(f"Turn {idx} User Question: {turn.get('question', '')}")
        lines.append(f"Turn {idx} Assistant Answer: {turn.get('answer', '')}")
    lines.append("")
    return "\n".join(lines)


def generate_grounded_answer(
    paper_meta: Optional[PaperMetadata],
    question: str,
    evidence_chunks: List[DocumentChunk],
    conversation_history: Optional[List[QAExchange]] = None,
    groq_client: Optional[groq.Groq] = None,
) -> Dict[str, Any]:
    """Generate a strictly grounded answer to a user question using retrieved evidence.

    Returns:
        Dict with keys:
        - answer: Grounded text answer or REFUSAL_MESSAGE
        - sources: List of source chunk metadata dicts
        - grounded: Boolean indicating if answer is grounded in evidence
        - cited_chunk_ids: List of chunk IDs referenced in the answer
    """
    if not question or not question.strip():
        return {
            "answer": "Please ask a question about the paper.",
            "sources": [],
            "grounded": False,
            "cited_chunk_ids": [],
        }

    # Empty retrieval check: immediately refuse without wasting LLM tokens
    if not evidence_chunks:
        logger.info("No evidence chunks available for question '%s'; returning refusal.", question)
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "grounded": False,
            "cited_chunk_ids": [],
        }

    evidence_text = format_qa_context(evidence_chunks)
    history_text = format_conversation_history(conversation_history)

    paper_title = paper_meta.get("title", "Unknown Title") if paper_meta else "Unknown Title"
    paper_id = paper_meta.get("arxiv_id", "") if paper_meta else ""

    system_prompt = (
        "You are an expert academic research assistant answering questions about a scientific paper.\n\n"
        "STRICT GROUNDING AND ANTI-HALLUCINATION RULES:\n"
        "1. You must answer the user's question EXCLUSIVELY using the retrieved paper evidence chunks provided below.\n"
        "2. Do NOT use general background knowledge, training memory, or extrapolation to fill gaps or answer unrelated questions.\n"
        "3. If the provided evidence chunks do NOT contain enough information to answer the question, or if the question is "
        "unrelated to the paper, you MUST output exactly:\n"
        f"\"{REFUSAL_MESSAGE}\"\n"
        "4. Do NOT fabricate, infer, or guess citations, metrics, results, page numbers, or chunk IDs.\n"
        "5. In your answer, cite every factual claim using the specific Chunk ID and Page number from the evidence "
        "(e.g. '[Chunk ID: 2512.14946_c0012, Page 7]').\n"
        "6. Prior conversation history is provided solely for conversational flow. It must NEVER be treated as a source "
        "of factual evidence. All factual claims must be grounded in the retrieved paper evidence chunks.\n"
        "7. Output your response as a valid JSON object matching the requested schema."
    )

    user_prompt = f"""
PAPER METADATA:
- Title: {paper_title}
- arXiv ID: {paper_id}

{history_text}
RETRIEVED PAPER EVIDENCE CHUNKS:
{evidence_text}

USER QUESTION:
{question}

INSTRUCTIONS:
Generate a grounded JSON response with the following schema:
{{
  "answer": "Detailed answer citing [Chunk ID: ...] and [Page ...], or exactly '{REFUSAL_MESSAGE}'",
  "cited_chunk_ids": ["list", "of", "cited", "chunk_ids"],
  "is_sufficient_evidence": true or false
}}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response_json = call_groq_json(messages=messages, client=groq_client)

    raw_answer = str(response_json.get("answer", "")).strip()
    is_sufficient = bool(response_json.get("is_sufficient_evidence", True))
    cited_ids = [str(cid).strip() for cid in response_json.get("cited_chunk_ids", []) if str(cid).strip()]

    # Normalize refusal if LLM indicated insufficient evidence or emitted refusal substring
    refusal_lower = REFUSAL_MESSAGE.lower()
    answer_lower = raw_answer.lower()
    if (
        not is_sufficient
        or refusal_lower in answer_lower
        or "does not contain enough information" in answer_lower
        or "not enough information" in answer_lower
        or "not mentioned in the provided" in answer_lower
        or "not provided in the paper" in answer_lower
    ):
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "grounded": False,
            "cited_chunk_ids": [],
        }

    # Build provenance sources list from retrieved evidence chunks
    sources = []
    for c in evidence_chunks:
        meta = c["metadata"]
        sources.append({
            "chunk_id": meta.get("chunk_id", ""),
            "page_start": meta.get("page_start"),
            "page_end": meta.get("page_end"),
            "section": meta.get("section"),
            "word_count": meta.get("word_count"),
        })

    return {
        "answer": raw_answer,
        "sources": sources,
        "grounded": True,
        "cited_chunk_ids": cited_ids,
    }
