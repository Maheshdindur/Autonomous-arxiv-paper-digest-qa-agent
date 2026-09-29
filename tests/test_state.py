"""Unit tests for state initialization and structure."""

from src.graph.state import AgentState, create_initial_state


def test_initial_state_creation():
    """Verify that create_initial_state creates a valid, fully initialized state dictionary."""
    user_query = "2401.12345"
    state = create_initial_state(user_query)

    assert state["user_input"] == user_query
    assert state["status"] == "init"
    assert state["candidate_papers"] == []
    assert state["chunks"] == []
    assert state["conversation_history"] == []
    assert state["retrieved_chunks"] == []
    assert state["selected_paper"] is None
    assert state["briefing"] is None
    assert state["error"] is None


def test_state_keys_conform_to_schema():
    """Verify all expected keys are present in initial state."""
    state = create_initial_state("test query")
    expected_keys = {
        "user_input",
        "query_type",
        "topic",
        "arxiv_id",
        "arxiv_url",
        "candidate_papers",
        "selected_paper",
        "pdf_path",
        "raw_text",
        "chunks",
        "vector_collection_name",
        "briefing",
        "current_question",
        "retrieved_chunks",
        "qa_answer",
        "conversation_history",
        "error",
        "status",
    }
    assert set(state.keys()) == expected_keys
