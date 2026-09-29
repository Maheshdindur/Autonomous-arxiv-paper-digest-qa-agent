"""Unit tests for CLI interface and argument handling."""

from src.cli import run_cli


def test_cli_no_args_returns_error(capsys):
    """Verify that running CLI with no input arguments returns exit code 1."""
    code = run_cli([])
    assert code == 1
    captured = capsys.readouterr()
    assert "Please provide an input query" in captured.err


def test_cli_check_config_fails_without_keys(capsys, monkeypatch):
    """Verify that --check-config exits with code 1 when no GROQ_API_KEY is configured."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    code = run_cli(["--check-config"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Configuration check failed" in captured.err
    assert "Missing GROQ_API_KEY" in captured.err


def test_cli_check_config_succeeds_with_groq_key(capsys, monkeypatch):
    """Verify that --check-config exits with code 0 when non-placeholder GROQ_API_KEY is set."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_testDummyKeyForTestingPurposes67890")

    code = run_cli(["--check-config"])
    assert code == 0
    captured = capsys.readouterr()
    assert "[OK] Configuration valid" in captured.out
    assert "Groq" in captured.out


from unittest.mock import MagicMock, patch


@patch("src.nodes.summarize.generate_executive_briefing")
def test_cli_workflow_trigger_with_key(mock_generate, monkeypatch):
    """Verify that the digest graph can be invoked through CLI when key is configured."""
    mock_generate.return_value = {
        "title": "Test Title",
        "authors": ["Author"],
        "arxiv_id": "2401.12345",
        "publish_date": "2024-01-01",
        "link": "https://arxiv.org/abs/2401.12345",
        "why_it_matters": "Matters",
        "problem_statement": "Problem",
        "method_approach": ["Method"],
        "key_results_claims": ["Claim"],
        "limitations": ["Limitation"],
        "suggested_followup_questions": ["Question"],
    }
    monkeypatch.setenv("GROQ_API_KEY", "gsk_testDummyKeyForTestingPurposes67890")

    code = run_cli(["2401.12345"])
    assert code == 0


def test_cli_ask_without_paper_fails(capsys, monkeypatch):
    """Verify that asking a question without specifying a paper returns exit code 1."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_testDummyKeyForTestingPurposes67890")
    code = run_cli(["--ask", "What is the speedup?"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Please specify a paper" in captured.err


@patch("src.cli.get_or_create_paper_collection")
@patch("src.cli.build_qa_graph")
def test_cli_ask_with_paper_id_succeeds(mock_qa_graph_builder, mock_get_col, capsys, monkeypatch):
    """Verify that asking a question with --paper-id executes QA and displays the answer."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_testDummyKeyForTestingPurposes67890")

    mock_col = MagicMock()
    mock_col.count.return_value = 10
    mock_get_col.return_value = mock_col

    mock_compiled_graph = MagicMock()
    mock_compiled_graph.invoke.return_value = {
        "status": "completed",
        "qa_answer": "EVICPRESS achieves 2.19x faster TTFT [Chunk ID: 2401.12345_c0001, Page 2].",
        "conversation_history": [
            {
                "question": "What is the speedup?",
                "answer": "EVICPRESS achieves 2.19x faster TTFT [Chunk ID: 2401.12345_c0001, Page 2].",
                "sources": [{"chunk_id": "2401.12345_c0001", "page_start": 2, "page_end": 2, "section": "Evaluation"}],
            }
        ],
    }
    mock_qa_graph_builder.return_value.compile.return_value = mock_compiled_graph

    code = run_cli(["--paper-id", "2401.12345", "--ask", "What is the speedup?"])
    assert code == 0
    captured = capsys.readouterr()
    assert "EVICPRESS achieves 2.19x faster TTFT" in captured.out
    assert "2401.12345_c0001" in captured.out
    assert "Sources / Provenance" in captured.out


@patch("src.cli.interactive_followup_loop")
@patch("src.nodes.summarize.generate_executive_briefing")
def test_cli_interactive_no_args_prompts_and_runs_pipeline(mock_generate, mock_loop, capsys, monkeypatch):
    """Verify that running CLI with no args in an interactive terminal prompts for input and runs pipeline."""
    mock_generate.return_value = {
        "title": "Test Title",
        "authors": ["Author"],
        "arxiv_id": "2401.12345",
        "publish_date": "2024-01-01",
        "link": "https://arxiv.org/abs/2401.12345",
        "why_it_matters": "Matters",
        "problem_statement": "Problem",
        "method_approach": ["Method"],
        "key_results_claims": ["Claim"],
        "limitations": ["Limitation"],
        "suggested_followup_questions": ["Question"],
    }
    monkeypatch.setenv("GROQ_API_KEY", "gsk_testDummyKeyForTestingPurposes67890")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "2401.12345")

    code = run_cli([])
    assert code == 0
    captured = capsys.readouterr()
    assert "Autonomous arXiv Paper Digest & QA Agent" in captured.out
    assert "Enter arXiv paper ID, URL, or research topic:" in captured.out
    mock_loop.assert_called_once()


def test_cli_interactive_empty_input_exits_cleanly(capsys, monkeypatch):
    """Verify that submitting empty input in interactive startup exits cleanly with code 0."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    code = run_cli([])
    assert code == 0
    captured = capsys.readouterr()
    assert "No input provided. Exiting." in captured.out


def test_cli_interactive_exit_quit_exits_cleanly(capsys, monkeypatch):
    """Verify that entering 'exit' or 'quit' at the interactive startup prompt exits with code 0."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "exit")

    code = run_cli([])
    assert code == 0
    captured = capsys.readouterr()
    assert "Goodbye." in captured.out


def test_cli_interactive_eof_or_interrupt_exits_cleanly(capsys, monkeypatch):
    """Verify that Ctrl+C / EOF at interactive startup exits cleanly with code 0 and no traceback."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def raise_interrupt(prompt=""):
        raise KeyboardInterrupt()

    monkeypatch.setattr("builtins.input", raise_interrupt)

    code = run_cli([])
    assert code == 0
    captured = capsys.readouterr()
    assert "Goodbye." in captured.out

