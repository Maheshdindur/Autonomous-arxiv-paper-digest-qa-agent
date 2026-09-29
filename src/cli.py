"""CLI entry point for Autonomous arXiv Paper Digest & QA Agent.

Parses command line arguments, validates environment/configuration,
and triggers the LangGraph workflow or QA execution.
"""

import argparse
import sys
from typing import Any, Optional

from src.config import ConfigurationError, load_config, setup_logging
from src.graph.state import AgentState, create_initial_state
from src.graph.workflow import build_digest_graph, build_qa_graph
from src.utils.vector_store import get_or_create_paper_collection


def format_qa_response(question: str, answer: str, sources: list) -> str:
    """Format QA answer and source provenance for console display."""
    lines = [
        f"\n**Question**: {question}",
        "",
        f"**Answer**:\n{answer}",
    ]
    if sources:
        lines.append("")
        lines.append("**Sources / Provenance**:")
        for s in sources:
            cid = s.get("chunk_id", "Unknown")
            sec = s.get("section") or "Unlabeled Section"
            p_start = s.get("page_start", "?")
            p_end = s.get("page_end", "?")
            lines.append(f"- `[{cid}]` Pages {p_start}-{p_end} (Section: {sec})")
    return "\n".join(lines)


def interactive_followup_loop(state: AgentState, qa_graph: Any) -> None:
    """Interactive console loop for grounded follow-up QA."""
    paper_id = (state.get("selected_paper") or {}).get("arxiv_id") or state.get("arxiv_id") or ""
    print("\n" + "-" * 60)
    print(f"Interactive Grounded QA Mode for paper: {paper_id}")
    print("Type your question and press Enter.")
    print("Type 'exit', 'quit', or Ctrl+C to stop.")
    print("-" * 60)

    current_state = dict(state)
    while True:
        try:
            user_q = input("\nAsk a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_q:
            continue
        if user_q.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        current_state["current_question"] = user_q
        current_state["status"] = "qa_retrieving"
        qa_result = qa_graph.invoke(current_state)

        if qa_result.get("status") == "error":
            print(f"\n[ERROR] QA Error: {qa_result.get('error')}", file=sys.stderr)
            continue

        answer = qa_result.get("qa_answer", "")
        history = qa_result.get("conversation_history", [])
        latest_sources = history[-1].get("sources", []) if history else []

        print(format_qa_response(user_q, answer, latest_sources))
        current_state = qa_result


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="arxiv-agent",
        description="Autonomous arXiv Paper Digest & QA Agent: Ingests topics or papers, produces executive briefings, and provides grounded QA.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Research topic, arXiv ID (e.g. 2401.12345), or arXiv URL (e.g. https://arxiv.org/abs/2401.12345)",
    )
    parser.add_argument(
        "--input",
        "-i",
        dest="input_flag",
        default=None,
        help="Alternative flag to specify input query/ID/URL",
    )
    parser.add_argument(
        "--ask",
        "-q",
        dest="question",
        default=None,
        help="Follow-up question to ask about a processed paper (grounded QA)",
    )
    parser.add_argument(
        "--paper-id",
        dest="paper_id",
        default=None,
        help="Specify arXiv ID explicitly when asking follow-up questions",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate configuration and API keys without executing a query",
    )
    return parser


def run_cli(args: Optional[list] = None) -> int:
    """Run CLI application with given arguments or sys.argv."""
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    # 1. Config check mode
    if parsed_args.check_config:
        try:
            config = load_config(require_llm_key=True)
            print(f"[OK] Configuration valid. Provider: Groq (Model: {config.groq_model})")
            print(f"[OK] Local credential check passed (key format validated).")
            print(f"[OK] Embedding Model: {config.embedding_model}")
            print(f"[OK] Storage: Chroma: {config.chroma_persist_dir}, PDFs: {config.pdf_download_dir}")
            return 0
        except ConfigurationError as e:
            print(f"[ERROR] Configuration check failed: {e}", file=sys.stderr)
            return 1

    # 2. Resolve query and paper inputs
    query = parsed_args.input_flag or parsed_args.query
    target_paper = parsed_args.paper_id or query

    if not query and not parsed_args.question and not parsed_args.paper_id:
        if sys.stdin.isatty():
            print("\n" + "=" * 60)
            print("Autonomous arXiv Paper Digest & QA Agent")
            print("=" * 60)
            try:
                print("\nEnter arXiv paper ID, URL, or research topic: ", end="", flush=True)
                user_entry = input().strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                return 0

            if not user_entry:
                print("No input provided. Exiting.")
                return 0

            if user_entry.lower() in ("exit", "quit", "q"):
                print("Goodbye.")
                return 0

            query = user_entry
            target_paper = user_entry
        else:
            parser.print_help()
            print("\n[ERROR] Please provide an input query (topic, arXiv ID, or URL) or use --ask for QA.", file=sys.stderr)
            return 1

    # 3. Load config and validate LLM credentials
    try:
        config = load_config(require_llm_key=True)
        logger = setup_logging(config.log_level)
    except ConfigurationError as e:
        print(f"[ERROR] Configuration error: {e}", file=sys.stderr)
        return 1

    # 4. Handle dedicated QA mode (--ask)
    if parsed_args.question:
        if not target_paper:
            print("[ERROR] Please specify a paper via --paper-id or positional query when using --ask.", file=sys.stderr)
            return 1

        logger.info("Executing grounded QA for paper '%s', question: '%s'", target_paper, parsed_args.question)

        # Check if paper is already indexed in Chroma
        try:
            col = get_or_create_paper_collection(target_paper)
            is_indexed = col.count() > 0
        except Exception:
            is_indexed = False

        state: AgentState
        if not is_indexed:
            logger.info("Paper '%s' not yet indexed; running digest ingestion pipeline first...", target_paper)
            initial_state = create_initial_state(target_paper)
            digest_graph = build_digest_graph().compile()
            state = digest_graph.invoke(initial_state)
            if state.get("status") == "error":
                print(f"[ERROR] Pipeline halted during paper ingestion: {state.get('error')}", file=sys.stderr)
                return 1
        else:
            state = create_initial_state(target_paper)
            state["arxiv_id"] = target_paper

        state["current_question"] = parsed_args.question
        qa_graph = build_qa_graph().compile()
        qa_result = qa_graph.invoke(state)

        if qa_result.get("status") == "error":
            print(f"\n[ERROR] QA Error: {qa_result.get('error')}", file=sys.stderr)
            return 1

        answer = qa_result.get("qa_answer", "")
        history = qa_result.get("conversation_history", [])
        latest_sources = history[-1].get("sources", []) if history else []

        print("\n" + "=" * 60)
        print(format_qa_response(parsed_args.question, answer, latest_sources))
        print("=" * 60)

        # If interactive terminal, allow continuous follow-up questions
        if sys.stdin.isatty():
            interactive_followup_loop(qa_result, qa_graph)

        return 0

    # 5. Execute Digest Workflow
    logger.info("Starting Paper Digest workflow for input: '%s'", query)
    initial_state = create_initial_state(query)
    graph = build_digest_graph().compile()

    # Run compiled LangGraph state machine
    final_state = graph.invoke(initial_state)
    logger.info("Digest workflow finished with status: %s", final_state.get("status"))

    if final_state.get("status") == "error":
        error_msg = final_state.get("error", "Unknown pipeline error occurred.")
        print(f"\n[ERROR] Pipeline halted: {error_msg}", file=sys.stderr)
        return 1

    # Print Executive Briefing if generated
    briefing = final_state.get("briefing")
    if briefing:
        from src.nodes.summarize import format_briefing_markdown
        print("\n" + "=" * 60)
        print(format_briefing_markdown(briefing))
        print("=" * 60 + "\n")

    # If running in an interactive terminal, offer follow-up QA loop
    if sys.stdin.isatty():
        qa_graph = build_qa_graph().compile()
        interactive_followup_loop(final_state, qa_graph)

    return 0


def main():
    """Main function entry point."""
    sys.exit(run_cli())


if __name__ == "__main__":
    main()
