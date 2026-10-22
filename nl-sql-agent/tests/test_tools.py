"""Tests for SQL safety checks and tools. These need no API key.

Run:  python -m pytest -q
"""

import sqlite3

import pytest

from agent.sql_tools import Database, UnsafeQueryError, make_tools, validate_sql
from data.create_db import build


@pytest.fixture(scope="session")
def db(tmp_path_factory):
    path = build(tmp_path_factory.mktemp("data") / "test.db")
    return Database(path)


@pytest.mark.parametrize("query", [
    "SELECT * FROM customers",
    "select count(*) from orders;",
    "WITH t AS (SELECT city FROM customers) SELECT city, COUNT(*) FROM t GROUP BY city",
    "SELECT city FROM customers UNION SELECT category FROM products",
])
def test_allows_read_queries(query):
    assert validate_sql(query)


@pytest.mark.parametrize("query", [
    "DELETE FROM customers",
    "DROP TABLE orders",
    "UPDATE products SET price = 0",
    "INSERT INTO customers VALUES (999, 'x', 'y', '2025-01-01')",
    "SELECT 1; DROP TABLE orders",
    "CREATE TABLE x (a INT)",
    "ALTER TABLE customers ADD COLUMN x INT",
    "PRAGMA table_info(customers)",
    "ATTACH DATABASE 'other.db' AS other",
    "",
])
def test_blocks_unsafe_queries(query):
    with pytest.raises(UnsafeQueryError):
        validate_sql(query)


def test_connection_is_read_only(db):
    # Even if validation were bypassed, the read-only connection refuses writes.
    with pytest.raises(sqlite3.OperationalError):
        with db._connect() as conn:
            conn.execute("DELETE FROM customers")


def test_run_sql_tool_returns_rows(db):
    run_sql = {t.name: t for t in make_tools(db)}["run_sql"]
    out = run_sql.invoke({"query": "SELECT COUNT(*) AS n FROM customers"})
    assert "200" in out


def test_run_sql_tool_reports_errors_as_text(db):
    tools = {t.name: t for t in make_tools(db)}
    assert tools["run_sql"].invoke({"query": "SELECT nope FROM customers"}).startswith("SQL ERROR")
    assert tools["run_sql"].invoke({"query": "DROP TABLE customers"}).startswith("REJECTED")


def test_run_sql_caps_rows(db):
    run_sql = {t.name: t for t in make_tools(db)}["run_sql"]
    out = run_sql.invoke({"query": "SELECT * FROM orders"})
    assert "Showing first 50 rows" in out


def test_describe_tables(db):
    describe = {t.name: t for t in make_tools(db)}["describe_tables"]
    out = describe.invoke({"tables": "orders, products"})
    assert "CREATE TABLE orders" in out and "CREATE TABLE products" in out
    assert "Unknown table" in describe.invoke({"tables": "sales"})
