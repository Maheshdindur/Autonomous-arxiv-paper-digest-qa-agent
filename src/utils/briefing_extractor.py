"""Evidence assembly and executive briefing prompt formulation.

Collects targeted evidence from parsed chunks across key sections (problem, method,
results, limitations) while preserving provenance metadata. Formulates grounded
prompts for the Groq LLM with strict anti-hallucination constraints.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from src.graph.state import DocumentChunk, ExecutiveBriefing, PaperMetadata
from src.utils.llm import LLMError, call_groq_json
from src.utils.vector_store import query_similar_chunks

logger = logging.getLogger("arxiv_agent.briefing_extractor")

TRUNCATED_PATTERNS = [
    re.compile(r"[\(\[][^\)\]]*\btruncated\b[^\)\]]*[\)\]]", re.IGNORECASE),
    re.compile(r"\btruncated\s+in\s+evidence\b", re.IGNORECASE),
    re.compile(r"\b(?:to|by|from|than)\s*[\.\,\;\:\s]*$", re.IGNORECASE),
    re.compile(r"\b(?:to|by|from|than)\s*[\(\[][^\)\]]*[\)\]]\s*$", re.IGNORECASE),
]


def is_incomplete_or_truncated(text: str) -> bool:
    """Check if a claim or text snippet contains incomplete or truncated numerical claims."""
    if not text:
        return False
    t = text.strip()
    for pat in TRUNCATED_PATTERNS:
        if pat.search(t):
            return True
    return False


def complete_trailing_sentence(
    chunk_text: str,
    next_chunk_text: Optional[str] = None,
    max_extra_words: int = 50,
) -> str:
    """If chunk_text ends mid-sentence, append words from next_chunk_text up to sentence end.

    Prevents numerical claims and statements from being artificially cut off at chunk boundaries.
    """
    if not chunk_text or not next_chunk_text:
        return chunk_text

    text = chunk_text.strip()
    if text.endswith((".", "!", "?", ".\"", "!\"", "?\"", ".'", "!'", "?'")):
        return chunk_text

    words_c = text.split()
    words_next = next_chunk_text.strip().split()

    overlap_idx = -1
    for match_len in range(min(15, len(words_c)), 2, -1):
        suffix = words_c[-match_len:]
        for j in range(min(len(words_next) - match_len + 1, 150)):
            if words_next[j : j + match_len] == suffix:
                overlap_idx = j + match_len
                break
        if overlap_idx != -1:
            break

    if overlap_idx == -1:
        return chunk_text

    continuation_words = []
    for w in words_next[overlap_idx : overlap_idx + max_extra_words]:
        continuation_words.append(w)
        if any(w.endswith(term) for term in (".", "!", "?", ".\"", "!\"", "?\"")):
            break

    if continuation_words:
        return chunk_text + " " + " ".join(continuation_words)
    return chunk_text


def sanitize_briefing_claims(briefing: ExecutiveBriefing) -> ExecutiveBriefing:
    """Sanitize executive briefing fields to ensure no incomplete/truncated claims are emitted.

    If an incomplete or truncated claim is found:
    - Omit the incomplete claim.
    - If omitting leaves key_results_claims empty, provide an explicit statement that
      the available evidence does not provide complete numerical values.
    """
    for list_key in ("key_results_claims", "method_approach", "limitations", "suggested_followup_questions"):
        items = briefing.get(list_key, [])
        cleaned = [item for item in items if not is_incomplete_or_truncated(item)]
        briefing[list_key] = cleaned

    for str_key in ("why_it_matters", "problem_statement"):
        val = briefing.get(str_key, "")
        if is_incomplete_or_truncated(val):
            for pat in TRUNCATED_PATTERNS:
                val = pat.sub("[incomplete claim omitted: evidence does not provide complete value]", val)
            briefing[str_key] = val.strip()

    if not briefing.get("key_results_claims"):
        briefing["key_results_claims"] = [
            "The available evidence does not provide complete quantitative results or benchmark claims."
        ]

    return briefing


def assemble_briefing_evidence(
    paper_meta: PaperMetadata,
    chunks: List[DocumentChunk],
    max_total_words: int = 2200,
) -> List[DocumentChunk]:
    """Assemble a curated set of evidence chunks covering the paper's key aspects.

    Avoids dumping the entire PDF into the prompt by selecting:
    1. Early context chunks (Abstract & Introduction)
    2. Method/Approach chunks
    3. Results/Evaluation chunks
    4. Limitations/Discussion chunks

    Enforces a strict total word budget (~2200 words / ~2800 tokens) to ensure
    prompts fit comfortably within LLM rate limits (e.g. Groq 8000 TPM limit).
    Completes trailing sentences at chunk boundaries to preserve complete numerical claims.
    """
    if not chunks:
        return []

    paper_id = paper_meta.get("arxiv_id", "")
    chunk_index_map = {c["metadata"]["chunk_id"]: idx for idx, c in enumerate(chunks)}
    selected_chunk_ids = set()
    evidence_chunks: List[DocumentChunk] = []
    current_word_count = 0

    def add_chunk(chunk: DocumentChunk) -> bool:
        nonlocal current_word_count
        cid = chunk["metadata"]["chunk_id"]
        if cid in selected_chunk_ids:
            return False

        # Attempt to complete trailing sentence if chunk cuts off mid-sentence
        idx = chunk_index_map.get(cid)
        text = chunk["text"]
        if idx is not None and idx + 1 < len(chunks):
            next_chunk = chunks[idx + 1]
            if next_chunk["metadata"].get("paper_id") == chunk["metadata"].get("paper_id"):
                text = complete_trailing_sentence(text, next_chunk["text"])

        chunk_to_add: DocumentChunk = {
            "text": text,
            "metadata": dict(chunk["metadata"]),
        }

        words = len(text.split())
        if current_word_count + words > max_total_words and evidence_chunks:
            return False

        selected_chunk_ids.add(cid)
        evidence_chunks.append(chunk_to_add)
        current_word_count += words
        return True

    # 1. Early chunk (first chunk provides abstract / introduction)
    if chunks:
        add_chunk(chunks[0])

    # 2. Section-based filtering (select 1 high-signal chunk per key aspect)
    method_found = False
    result_found = False
    limit_found = False

    for c in chunks:
        sec = (c["metadata"].get("section") or "").lower()
        if not method_found and any(k in sec for k in ["method", "approach", "architecture"]):
            if add_chunk(c):
                method_found = True
        elif not result_found and any(k in sec for k in ["result", "evaluation", "experiment"]):
            if add_chunk(c):
                result_found = True
        elif not limit_found and any(k in sec for k in ["limitation", "discussion", "conclusion"]):
            if add_chunk(c):
                limit_found = True

    # 3. Vector-similarity fallback/supplement for aspects not yet found
    aspect_queries = []
    if not method_found:
        aspect_queries.append("What is the proposed method, algorithm, or architecture?")
    if not result_found:
        aspect_queries.append("What are the key experimental results and benchmark claims?")
    if not limit_found:
        aspect_queries.append("What are the explicit limitations, assumptions, or failure cases?")

    for q in aspect_queries:
        try:
            hits = query_similar_chunks(paper_id=paper_id, query=q, top_k=1)
            for hit in hits:
                for c in chunks:
                    if c["metadata"]["chunk_id"] == hit["chunk_id"]:
                        add_chunk(c)
                        break
        except Exception as e:
            logger.debug("Vector search supplement for '%s' skipped: %s", q, e)

    # 4. If still under budget and fewer than 3 chunks, add second intro chunk
    if len(evidence_chunks) < 3 and len(chunks) > 1:
        add_chunk(chunks[1])

    logger.info(
        "Assembled %d targeted evidence chunks (%d words) for executive briefing of %s",
        len(evidence_chunks),
        current_word_count,
        paper_id,
    )
    return evidence_chunks


def format_evidence_block(evidence_chunks: List[DocumentChunk]) -> str:
    """Format evidence chunks into a clean prompt string with explicit source citations."""
    blocks = []
    for idx, c in enumerate(evidence_chunks, 1):
        meta = c["metadata"]
        sec_label = meta.get("section") or "Unlabeled Section"
        pages = f"Page {meta.get('page_start', '?')}-{meta.get('page_end', '?')}"
        blocks.append(
            f"--- EVIDENCE PIECE {idx} [Chunk ID: {meta.get('chunk_id')} | {pages} | Section: {sec_label}] ---\n"
            f"{c['text']}\n"
        )
    return "\n".join(blocks)


def generate_executive_briefing(
    paper_meta: PaperMetadata,
    evidence_chunks: List[DocumentChunk],
    groq_client: Optional[Any] = None,
) -> ExecutiveBriefing:
    """Generate structured executive briefing grounded in assembled paper evidence.

    Authoritative metadata (Title, Authors, arXiv ID, Date, Link) is populated
    directly from paper_meta, not left to LLM recall.
    """
    evidence_text = format_evidence_block(evidence_chunks)

    system_prompt = (
        "You are an expert scientific analyst creating a rigorous, grounded executive briefing for a research paper.\n"
        "CRITICAL ANTI-HALLUCINATION RULES:\n"
        "1. Base your summary and claims EXCLUSIVELY on the provided paper evidence.\n"
        "2. Do NOT use general background knowledge to invent or assume details not present in the text.\n"
        "3. If a specific aspect (e.g. limitations, problem statement, or results) is NOT explicitly discussed in the evidence, "
        "explicitly state that the paper does not clearly state or provide this information in the supplied sections.\n"
        "4. For limitations: do NOT invent limitations. If the paper explicitly discusses limitations, summarize them. "
        "If limitations cannot be reliably identified from the provided text, state: "
        "'The paper does not explicitly discuss limitations in the provided sections.'\n"
        "5. NEVER emit incomplete, fragmented, or truncated numerical claims, ranges, or comparisons "
        "(such as 'reduced by 1.43× to (truncated)' or dangling 'to ...'). "
        "If the complete value or range is present in the paper evidence, state the complete value (e.g., '1.43 to 3.77×'). "
        "If the complete value is not available in the evidence, either omit the incomplete claim entirely or explicitly state "
        "that the available evidence does not provide the complete value. Never emit phrases like '(value truncated in evidence)'.\n"
        "6. Output your response as a valid JSON object matching the required schema."
    )

    user_prompt = f"""
PAPER METADATA:
- Title: {paper_meta.get('title')}
- Authors: {", ".join(paper_meta.get('authors', []))}
- arXiv ID: {paper_meta.get('arxiv_id')}
- Abstract: {paper_meta.get('abstract')}

RETRIEVED PAPER EVIDENCE CHUNKS:
{evidence_text if evidence_text else "No content chunks available."}

INSTRUCTIONS:
Generate a structured briefing in JSON format with the following keys:
{{
  "why_it_matters": "A concise 1-paragraph plain-English summary explaining why this paper matters and its significance.",
  "problem_statement": "The core problem or bottleneck the paper aims to solve.",
  "method_approach": [
    "Bullet point describing component 1 of the approach",
    "Bullet point describing component 2"
  ],
  "key_results_claims": [
    "Key result or quantitative claim 1 (complete values only; omit incomplete or truncated comparisons)",
    "Key result or claim 2"
  ],
  "limitations": [
    "Explicit limitation mentioned in the text (or 'The paper does not explicitly state limitations in the provided sections.')"
  ],
  "suggested_followup_questions": [
    "Thoughtful follow-up question 1 a reader might ask",
    "Thoughtful follow-up question 2",
    "Thoughtful follow-up question 3"
  ]
}}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response_json = call_groq_json(messages=messages, client=groq_client)

    # Validate and build final ExecutiveBriefing using authoritative metadata
    briefing: ExecutiveBriefing = {
        "title": paper_meta.get("title", ""),
        "authors": paper_meta.get("authors", []),
        "arxiv_id": paper_meta.get("arxiv_id", ""),
        "publish_date": paper_meta.get("published", ""),
        "link": paper_meta.get("entry_id") or f"https://arxiv.org/abs/{paper_meta.get('arxiv_id', '')}",
        "why_it_matters": str(response_json.get("why_it_matters", "")).strip(),
        "problem_statement": str(response_json.get("problem_statement", "")).strip(),
        "method_approach": [str(item).strip() for item in response_json.get("method_approach", []) if str(item).strip()],
        "key_results_claims": [str(item).strip() for item in response_json.get("key_results_claims", []) if str(item).strip()],
        "limitations": [str(item).strip() for item in response_json.get("limitations", []) if str(item).strip()],
        "suggested_followup_questions": [
            str(item).strip() for item in response_json.get("suggested_followup_questions", []) if str(item).strip()
        ],
    }

    # Sanitize claims against incomplete/truncated expressions
    briefing = sanitize_briefing_claims(briefing)

    # Ensure empty lists default to informative statements
    if not briefing["limitations"]:
        briefing["limitations"] = ["The paper does not explicitly state limitations in the provided sections."]

    return briefing
