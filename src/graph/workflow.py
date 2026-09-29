"""LangGraph workflow construction and orchestration.

Separated cleanly from CLI and individual node implementations.
"""

from typing import Any, Dict, Literal
from langgraph.graph import StateGraph, END

from src.graph.state import AgentState
from src.nodes.query_understanding import query_understanding_node
from src.nodes.arxiv_retrieval import arxiv_retrieval_node
from src.nodes.pdf_fetch import pdf_fetch_node
from src.nodes.pdf_parse import pdf_parse_node
from src.nodes.chunk_and_index import chunk_and_index_node
from src.nodes.summarize import summarize_node
from src.nodes.qa import qa_retrieval_node, qa_answer_node


def error_handler_node(state: AgentState) -> Dict[str, Any]:
    """Terminal error handler node logging issues."""
    return {"status": "error"}


def route_after_query(state: AgentState) -> Literal["arxiv_retrieval", "error_handler"]:
    if state.get("status") == "error" or state.get("error"):
        return "error_handler"
    return "arxiv_retrieval"


def route_after_retrieval(state: AgentState) -> Literal["pdf_fetch", "error_handler"]:
    if state.get("status") == "error" or state.get("error"):
        return "error_handler"
    return "pdf_fetch"


def route_after_fetch(state: AgentState) -> Literal["pdf_parse", "error_handler"]:
    if state.get("status") == "error" or state.get("error"):
        return "error_handler"
    return "pdf_parse"


def route_after_parse(state: AgentState) -> Literal["chunk_and_index", "error_handler"]:
    if state.get("status") == "error" or state.get("error"):
        return "error_handler"
    return "chunk_and_index"


def route_after_index(state: AgentState) -> Literal["summarize", "error_handler"]:
    if state.get("status") == "error" or state.get("error"):
        return "error_handler"
    return "summarize"


def build_digest_graph() -> StateGraph:
    """Build and compile the StateGraph for the paper digest workflow.

    Pipeline:
    query_understanding -> arxiv_retrieval -> pdf_fetch ->
    pdf_parse -> chunk_and_index -> summarize -> END
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("query_understanding", query_understanding_node)
    workflow.add_node("arxiv_retrieval", arxiv_retrieval_node)
    workflow.add_node("pdf_fetch", pdf_fetch_node)
    workflow.add_node("pdf_parse", pdf_parse_node)
    workflow.add_node("chunk_and_index", chunk_and_index_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("error_handler", error_handler_node)

    workflow.set_entry_point("query_understanding")

    workflow.add_conditional_edges(
        "query_understanding",
        route_after_query,
        {
            "arxiv_retrieval": "arxiv_retrieval",
            "error_handler": "error_handler",
        },
    )
    workflow.add_conditional_edges(
        "arxiv_retrieval",
        route_after_retrieval,
        {
            "pdf_fetch": "pdf_fetch",
            "error_handler": "error_handler",
        },
    )
    workflow.add_conditional_edges(
        "pdf_fetch",
        route_after_fetch,
        {
            "pdf_parse": "pdf_parse",
            "error_handler": "error_handler",
        },
    )
    workflow.add_conditional_edges(
        "pdf_parse",
        route_after_parse,
        {
            "chunk_and_index": "chunk_and_index",
            "error_handler": "error_handler",
        },
    )
    workflow.add_conditional_edges(
        "chunk_and_index",
        route_after_index,
        {
            "summarize": "summarize",
            "error_handler": "error_handler",
        },
    )

    workflow.add_edge("summarize", END)
    workflow.add_edge("error_handler", END)

    return workflow


def route_after_qa_retrieval(state: AgentState) -> Literal["qa_answer", "error_handler"]:
    if state.get("status") == "error" or state.get("error"):
        return "error_handler"
    return "qa_answer"


def build_qa_graph() -> StateGraph:
    """Build and compile the StateGraph for Grounded Question Answering.

    Pipeline:
    qa_retrieval -> qa_answer -> END
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("qa_retrieval", qa_retrieval_node)
    workflow.add_node("qa_answer", qa_answer_node)
    workflow.add_node("error_handler", error_handler_node)

    workflow.set_entry_point("qa_retrieval")

    workflow.add_conditional_edges(
        "qa_retrieval",
        route_after_qa_retrieval,
        {
            "qa_answer": "qa_answer",
            "error_handler": "error_handler",
        },
    )
    workflow.add_edge("qa_answer", END)
    workflow.add_edge("error_handler", END)

    return workflow
