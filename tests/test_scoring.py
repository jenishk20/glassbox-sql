"""The comparison rules must match BIRD's official evaluation.py exactly."""

import sqlite3

import pytest

from glassbox.score import classify, extract_sql
from glassbox.sqlexec import ExecResult, error_type, execute, results_match


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (a INTEGER, b TEXT, c REAL)")
    conn.executemany("INSERT INTO t VALUES (?, ?, ?)", [(1, "x", 1.0), (2, "y", 2.5), (2, "y", 2.5)])
    conn.commit()
    conn.close()
    return path


def same(db, pred, gold):
    return results_match(execute(db, pred), execute(db, gold))


def test_row_order_and_duplicates_are_ignored(db):
    assert same(db, "SELECT a FROM t ORDER BY a DESC", "SELECT a FROM t ORDER BY a")
    assert same(db, "SELECT DISTINCT b FROM t", "SELECT b FROM t")


def test_column_order_matters(db):
    assert not same(db, "SELECT a, b FROM t", "SELECT b, a FROM t")


def test_int_equals_float_but_not_text(db):
    assert same(db, "SELECT 1", "SELECT 1.0")
    assert not same(db, "SELECT '1'", "SELECT 1")


def test_empty_results_match(db):
    assert same(db, "SELECT a FROM t WHERE a > 99", "SELECT b FROM t WHERE 0")


def test_multiple_statements_are_an_error(db):
    r = execute(db, "SELECT 1; SELECT 2")
    assert not r.ok and error_type(r.error) == "multiple statements"


def test_database_is_read_only(db):
    r = execute(db, "DROP TABLE t")
    assert not r.ok
    assert execute(db, "SELECT count(*) FROM t").rows == [(3,)]


def test_timeout(db):
    r = execute(db, "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT count(*) FROM n",
                timeout=0.5)
    assert r.timed_out and not r.ok


def test_error_types():
    assert error_type("no such column: T1.foo") == "no such column"
    assert error_type('near "FROM": syntax error') == "syntax error"


def test_extract_takes_last_sql_block():
    text = "Draft:\n```sql\nSELECT 1\n```\nFinal:\n```sql\nSELECT 2;\n```"
    assert extract_sql({"text": text}) == ("SELECT 2;", "sql_fence")


def test_extract_fallbacks():
    assert extract_sql({"text": "```\nSELECT 3\n```"}) == ("SELECT 3", "plain_fence")
    assert extract_sql({"text": "<think>```sql\nSELECT 0\n```</think>```SQL\nSELECT 4\n```"})[0] == "SELECT 4"
    assert extract_sql({"text": "```sql\nSELECT 5"}) == ("", "none")          # unterminated
    assert extract_sql({"text": "SELECT 6\n", "prefilled": True}) == ("SELECT 6", "prefilled")
    assert extract_sql({"text": None, "error": "boom"}) == ("", "none")


def test_classify_categories(db):
    gold = execute(db, "SELECT a FROM t")
    ok = lambda sql: classify({"text": "x"}, sql, execute(db, sql), gold)  # noqa: E731
    assert ok("SELECT a FROM t")["category"] == "correct"
    assert ok("SELECT a, b FROM t")["wrong_kind"] == "columns"
    assert ok("SELECT a FROM t WHERE a = 1")["wrong_kind"] == "row_count"
    assert ok("SELECT a + 10 FROM t")["wrong_kind"] == "values"
    assert ok("SELECT nope FROM t")["error_type"] == "no such column"
    assert classify({"text": "", "finish_reason": "length"}, "", None, gold)["category"] == "truncated"
    assert classify({"text": "hi", "finish_reason": "stop"}, "", None, gold)["category"] == "no_sql"
    assert classify({"error": "x"}, "", None, gold)["category"] == "api_error"
    assert classify({"text": "x"}, "SELECT 1", None, ExecResult(ok=False, error="timeout"))["category"] == "gold_failed"
