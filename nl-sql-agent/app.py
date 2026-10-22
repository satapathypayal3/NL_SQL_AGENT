"""Command-line chat with the NL-to-SQL agent.

Usage:
    python app.py                          # interactive mode
    python app.py "Which city has the most customers?"
    python app.py --model google_genai:gemini-2.5-flash
"""

import argparse
import sys

from dotenv import load_dotenv

from agent.core import DEFAULT_DB, SQLAgent


def print_answer(ans, show_result: bool = True) -> None:
    print("\n" + "=" * 70)
    print(f"Answer: {ans.answer}")
    if ans.sql:
        print("\nSQL used:")
        print("  " + ans.sql.replace("\n", "\n  "))
    if show_result and ans.result:
        print("\nResult:")
        print("  " + ans.result.replace("\n", "\n  "))
    print(f"\n({ans.steps} tool calls)")
    print("=" * 70)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ask questions about a database in plain English.")
    parser.add_argument("question", nargs="?", help="Ask one question and exit.")
    parser.add_argument("--model", help='LLM as "provider:model", e.g. groq:openai/gpt-oss-120b')
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to a SQLite database file.")
    parser.add_argument("--hide-result", action="store_true", help="Do not print the raw result table.")
    args = parser.parse_args()

    agent = SQLAgent(db_path=args.db, model=args.model)

    if args.question:
        print_answer(agent.ask(args.question), not args.hide_result)
        return

    print("Ask a question about the data (type 'exit' to quit).")
    print("Tables:", ", ".join(agent.db.table_names()))
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        try:
            print_answer(agent.ask(question), not args.hide_result)
        except Exception as e:  # network errors, rate limits, bad API key
            print(f"Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
