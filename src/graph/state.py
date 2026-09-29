"""Explicit state schema for the Autonomous arXiv Paper Digest & QA Agent.

Defines the AgentState TypedDict and PaperMetadata structures that are passed
across LangGraph nodes. Each field is explicitly documented with its purpose.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict


class ChunkMetadata(TypedDict, total=False):
    """Metadata preserved with each chunk for retrieval and grounding."""
    chunk_id: str
    paper_id: str
    page_start: int
    page_end: int
    section: Optional[str]  # Detected heading/section name if reliable, else None
    word_count: int  # Explicit word count of the chunk window
    token_count: Optional[int]  # Optional alias for backward compatibility


class DocumentChunk(TypedDict):
    """Represents a chunk of text extracted from a parsed paper."""
    text: str
    metadata: ChunkMetadata


class PaperMetadata(TypedDict, total=False):
    """Structured metadata retrieved from arXiv API."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published: str
    updated: Optional[str]
    pdf_url: str
    entry_id: str
    primary_category: str
    categories: List[str]
    comment: Optional[str]
    journal_ref: Optional[str]
    doi: Optional[str]


class ExecutiveBriefing(TypedDict, total=False):
    """Structured executive briefing required by the assignment specification."""
    title: str
    authors: List[str]
    arxiv_id: str
    publish_date: str
    link: str
    why_it_matters: str  # 1-paragraph plain-English summary
    problem_statement: str
    method_approach: List[str]  # Bullet points
    key_results_claims: List[str]  # Bullet points
    limitations: List[str]  # Explicitly listed limitations
    suggested_followup_questions: List[str]  # List of suggested questions


class QAExchange(TypedDict):
    """Record of a single QA interaction."""
    question: str
    answer: str
    sources: List[Dict[str, Any]]


class AgentState(TypedDict, total=False):
    """The central state passed across LangGraph nodes.

    Design rationale:
    - user_input: Raw query received from CLI.
    - query_type: Classified intent ('topic' vs 'paper').
    - topic: Cleaned search string if query_type is 'topic'.
    - arxiv_id / arxiv_url: Target paper identifier if query_type is 'paper'.
    - candidate_papers: Candidate metadata list returned by arXiv search.
    - selected_paper: The chosen paper metadata to process.
    - pdf_path: Local filesystem path where the PDF is stored.
    - raw_text: Extracted full text from the PDF.
    - chunks: List of DocumentChunks ready for embedding and vector store.
    - vector_collection_name: Unique collection name in Chroma for this paper.
    - briefing: Structured executive briefing.
    - current_question: Active question being answered in QA mode.
    - retrieved_chunks: Retrieved context chunks for current_question.
    - qa_answer: Grounded LLM response to current_question.
    - conversation_history: History of QA exchanges for stateful interaction.
    - error: Human-readable error message if any node fails.
    - status: Operational status ('init', 'retrieving', 'parsing', 'indexing',
             'summarizing', 'completed', 'error').
    """
    user_input: str
    query_type: Optional[Literal["topic", "paper"]]
    topic: Optional[str]
    arxiv_id: Optional[str]
    arxiv_url: Optional[str]

    candidate_papers: List[PaperMetadata]
    selected_paper: Optional[PaperMetadata]
    pdf_path: Optional[str]
    raw_text: Optional[str]
    chunks: List[DocumentChunk]

    vector_collection_name: Optional[str]
    briefing: Optional[ExecutiveBriefing]

    current_question: Optional[str]
    retrieved_chunks: List[DocumentChunk]
    qa_answer: Optional[str]
    conversation_history: List[QAExchange]

    error: Optional[str]
    status: Literal[
        "init",
        "understanding_query",
        "retrieving",
        "ranking",
        "fetching_pdf",
        "parsing_pdf",
        "chunking",
        "indexing",
        "summarizing",
        "completed",
        "qa_retrieving",
        "qa_answering",
        "error",
    ]


def create_initial_state(user_input: str) -> AgentState:
    """Helper to initialize an AgentState dictionary."""
    return {
        "user_input": user_input,
        "query_type": None,
        "topic": None,
        "arxiv_id": None,
        "arxiv_url": None,
        "candidate_papers": [],
        "selected_paper": None,
        "pdf_path": None,
        "raw_text": None,
        "chunks": [],
        "vector_collection_name": None,
        "briefing": None,
        "current_question": None,
        "retrieved_chunks": [],
        "qa_answer": None,
        "conversation_history": [],
        "error": None,
        "status": "init",
    }
