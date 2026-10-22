"""Measure how often the agent gets the right answer.

For each question in questions.json we run the agent, take the last SQL it executed, and
compare that query's result with the result of a hand-written "gold" query
(execution accuracy, the standard metric for text-to-SQL).

Run:  python -m eval.run_eval              (from the project folder)
      python -m eval.run_eval --limit 5 --sleep 3
Writes eval/results.csv and prints accuracy overall and by difficulty.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from itertools import permutations
from pathlib import Path

from dotenv import load_dotenv

from agent.core import DEFAULT_DB, SQLAgent
from agent.sql_tools import Database

HERE = Path(__file__).parent


def _norm(v):
    """Normalise a value so 4999 == 4999.0 and 123.4567 == 123.46."""
    if isinstance(v, float):
        v = round(v, 2)
        return int(v) if v.is_integer() else v
    if isinstance(v, str):
        return v.strip().lower()
    return v


def results_match(gold_rows: list[tuple], pred_rows: list[tuple]) -> bool:
    """True if the prediction contains the gold columns (in any order) with the same rows.

    Extra columns in the prediction are allowed (e.g. a model that also returns a count).
    Row order is ignored.
    """
    if len(gold_rows) != len(pred_rows):
        return False
    if not gold_rows:
        return True

    gold = sorted((tuple(_norm(v) for v in r) for r in gold_rows), key=repr)
    pred = [tuple(_norm(v) for v in r) for r in pred_rows]
    n_gold, n_pred = len(gold[0]), len(pred[0])
    if n_pred < n_gold or n_pred > 8:  # keep the column search small
        return False

    for cols in permutations(range(n_pred), n_gold):
        projected = sorted((tuple(r[c] for c in cols) for r in pred), key=repr)
        if projected == gold:
            return True
    return False


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help='LLM as "provider:model"')
    parser.add_argument("--limit", type=int, help="Only run the first N questions.")
    parser.add_argument("--sleep", type=float, default=0, help="Seconds to wait between questions (rate limits).")
    args = parser.parse_args()

    questions = json.loads((HERE / "questions.json").read_text())[: args.limit]
    agent = SQLAgent(model=args.model)
    checker = Database(DEFAULT_DB)

    rows_out = []
    for q in questions:
        start = time.time()
        error, pred_sql, correct = "", None, False
        try:
            ans = agent.ask(q["question"])
            pred_sql = ans.sql
            if pred_sql:
                gold_rows = checker.run(q["gold_sql"], max_rows=10_000).rows
                pred_rows = checker.run(pred_sql, max_rows=10_000).rows
                correct = results_match(gold_rows, pred_rows)
            else:
                error = "agent ran no successful query"
            answer = ans.answer
        except Exception as e:
            answer, error = "", f"{type(e).__name__}: {e}"
        secs = time.time() - start

        mark = "PASS" if correct else "FAIL"
        print(f"[{mark}] #{q['id']:>2} ({q['difficulty']:<6}) {q['question']}  [{secs:.1f}s]")
        if error:
            print(f"        {error}")
        rows_out.append({"id": q["id"], "difficulty": q["difficulty"], "question": q["question"],
                         "correct": correct, "agent_sql": pred_sql or "", "gold_sql": q["gold_sql"],
                         "answer": answer, "error": error, "seconds": round(secs, 1)})
        if args.sleep:
            time.sleep(args.sleep)

    out = HERE / "results.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    total = sum(r["correct"] for r in rows_out)
    print(f"\nExecution accuracy: {total}/{len(rows_out)} = {100 * total / len(rows_out):.1f}%")
    by_diff, n_diff = Counter(), Counter()
    for r in rows_out:
        n_diff[r["difficulty"]] += 1
        by_diff[r["difficulty"]] += r["correct"]
    for d in ("easy", "medium", "hard"):
        if n_diff[d]:
            print(f"  {d:<6}: {by_diff[d]}/{n_diff[d]}")
    print(f"Details saved to {out}")


if __name__ == "__main__":
    main()
