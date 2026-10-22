"""Database access and the tools the agent can call.

Safety layers (defence in depth):
  1. The SQLite connection is opened read-only (mode=ro), so writes fail at the database level.
  2. Every query is parsed with sqlglot before it runs: one statement only, and it must be a
     SELECT (or WITH ... SELECT / UNION). Anything else is rejected with a clear message.
  3. Results are capped at MAX_ROWS rows, and long-running queries are interrupted.
Errors are returned to the model as text (not raised) so the agent can read them and retry.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import sqlglot
from sqlglot import exp
from langchain_core.tools import tool

MAX_ROWS = 50
QUERY_TIMEOUT_SECONDS = 5

# Statement types that must never run, even inside a CTE or subquery.
FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.Command, exp.Merge, exp.TruncateTable, exp.Attach, exp.Detach, exp.Pragma,
)


class UnsafeQueryError(ValueError):
    """Raised when a query fails validation."""


def validate_sql(query: str) -> str:
    """Return the cleaned query if it is a single read-only SELECT; otherwise raise."""
    cleaned = query.strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeQueryError("Query is empty.")

    try:
        statements = sqlglot.parse(cleaned, read="sqlite")
    except sqlglot.errors.ParseError as e:
        raise UnsafeQueryError(f"Could not parse SQL: {e}") from e

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise UnsafeQueryError("Only one SQL statement is allowed per query.")

    tree = statements[0]
    if not isinstance(tree, exp.Query):  # Select, Union, Intersect, Except
        raise UnsafeQueryError("Only SELECT queries are allowed (read-only access).")

    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise UnsafeQueryError(f"Query contains a forbidden operation: {type(node).__name__}.")

    return cleaned


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool


@dataclass
class Database:
    path: Path
    query_log: list[str] = field(default_factory=list)  # every SQL query that ran successfully

    def __post_init__(self):
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"Database not found: {self.path}. Run `python data/create_db.py` first.")

    def _connect(self) -> sqlite3.Connection:
        # check_same_thread=False: tools may run on a worker thread inside the agent framework.
        conn = sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True, check_same_thread=False)
        deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS
        # Returning a truthy value from the progress handler aborts the running query.
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        return conn

    def table_names(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        return [r[0] for r in rows]

    def describe(self, tables: list[str], sample_rows: int = 3) -> str:
        known = set(self.table_names())
        unknown = [t for t in tables if t not in known]
        if unknown:
            return f"Unknown table(s): {', '.join(unknown)}. Available tables: {', '.join(sorted(known))}."

        parts = []
        with self._connect() as conn:
            for t in tables:
                ddl = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone()[0]
                cur = conn.execute(f'SELECT * FROM "{t}" LIMIT {int(sample_rows)}')
                cols = [d[0] for d in cur.description]
                sample = format_table(cols, cur.fetchall())
                parts.append(f"{ddl.strip()}\n\n/* {sample_rows} sample rows from {t}:\n{sample}\n*/")
        return "\n\n".join(parts)

    def run(self, query: str, max_rows: int = MAX_ROWS) -> QueryResult:
        safe_query = validate_sql(query)
        with self._connect() as conn:
            cur = conn.execute(safe_query)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(max_rows + 1)
        self.query_log.append(safe_query)
        return QueryResult(columns, rows[:max_rows], truncated=len(rows) > max_rows)


def format_table(columns: list[str], rows: list[tuple]) -> str:
    """Plain-text table that is easy for both people and the model to read."""
    if not rows:
        return "(no rows)"
    str_rows = [["NULL" if v is None else str(v) for v in row] for row in rows]
    widths = [max(len(c), *(len(r[i]) for r in str_rows)) for i, c in enumerate(columns)]
    line = " | ".join(c.ljust(w) for c, w in zip(columns, widths))
    sep = "-+-".join("-" * w for w in widths)
    body = "\n".join(" | ".join(v.ljust(w) for v, w in zip(r, widths)) for r in str_rows)
    return f"{line}\n{sep}\n{body}"


def make_tools(db: Database):
    """Create the LangChain tools bound to one database."""

    @tool
    def list_tables() -> str:
        """List all tables in the database. Call this first to see what data exists."""
        return ", ".join(db.table_names())

    @tool
    def describe_tables(tables: str) -> str:
        """Show the schema (columns, types, keys) and a few sample rows for the given tables.

        Args:
            tables: Comma-separated table names, e.g. "orders, customers".
        """
        names = [t.strip() for t in tables.split(",") if t.strip()]
        return db.describe(names)

    @tool
    def run_sql(query: str) -> str:
        """Run a single read-only SQLite SELECT query and return the result rows.

        If the query fails, the error message is returned so you can fix the query and try again.

        Args:
            query: A SQLite SELECT statement.
        """
        try:
            result = db.run(query)
        except UnsafeQueryError as e:
            return f"REJECTED: {e}"
        except sqlite3.Error as e:
            msg = "Query took too long and was stopped." if "interrupt" in str(e).lower() else str(e)
            return f"SQL ERROR: {msg} Check table and column names with describe_tables, then fix the query."

        text = format_table(result.columns, result.rows)
        if result.truncated:
            text += f"\n(Showing first {MAX_ROWS} rows only. Use aggregation or LIMIT for large results.)"
        return text

    return [list_tables, describe_tables, run_sql]
