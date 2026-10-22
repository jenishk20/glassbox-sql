"""Run SQL against a BIRD database and compare results the way the official BIRD script does.

Official rule (BIRD evaluation.py): a prediction is correct iff
    set(predicted_rows) == set(gold_rows)
on the raw Python values from sqlite3's fetchall(). So row order and duplicate rows are
ignored, column order matters, 1 == 1.0, '1' != 1, and two empty results match.

Differences from the official script, all deliberate and small:
  * databases are opened read-only, so a generated DROP/UPDATE cannot damage them;
  * a model's query gets 30 s, the official limit. A gold (answer-key) query gets 5 minutes,
    because it must always produce the answer: on a laptop, dev question 701's gold query
    takes about 3 minutes, and with a 30 s limit the score would depend on the machine;
  * results over MAX_ROWS rows are cut off and flagged. No BIRD dev gold query is close
    to that size, so such a prediction is wrong either way.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

TIMEOUT_S = 30.0          # model queries (official BIRD limit)
GOLD_TIMEOUT_S = 300.0    # answer-key queries
MAX_ROWS = 1_000_000


@dataclass
class ExecResult:
    ok: bool
    rows: list[tuple] = field(default_factory=list)
    error: str | None = None
    timed_out: bool = False
    too_large: bool = False
    seconds: float = 0.0


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True, check_same_thread=False)


def execute(db_path: Path, sql: str, timeout: float = TIMEOUT_S) -> ExecResult:
    start = time.monotonic()
    deadline = start + timeout
    conn = connect_readonly(db_path)
    # SQLite calls this every 10k VM steps; a non-zero return aborts the query.
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = conn.execute(sql)
        rows: list[tuple] = []
        while True:
            chunk = cur.fetchmany(10_000)
            if not chunk:
                break
            rows.extend(chunk)
            if len(rows) > MAX_ROWS:
                return ExecResult(ok=True, rows=rows[:MAX_ROWS], too_large=True,
                                  seconds=time.monotonic() - start)
        return ExecResult(ok=True, rows=rows, seconds=time.monotonic() - start)
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e) and time.monotonic() > deadline:
            return ExecResult(ok=False, error="timeout", timed_out=True, seconds=time.monotonic() - start)
        return ExecResult(ok=False, error=str(e), seconds=time.monotonic() - start)
    except Exception as e:  # ProgrammingError for multiple statements, Warning, etc.
        return ExecResult(ok=False, error=str(e) or type(e).__name__, seconds=time.monotonic() - start)
    finally:
        conn.close()


def results_match(pred: ExecResult, gold: ExecResult) -> bool:
    """The official BIRD comparison. Both queries must have run."""
    if not (pred.ok and gold.ok) or pred.too_large:
        return False
    return set(pred.rows) == set(gold.rows)


def error_type(message: str) -> str:
    """Collapse an SQLite error message into a short, countable label."""
    m = message.strip()
    if "syntax error" in m:
        return "syntax error"
    if "one statement at a time" in m:
        return "multiple statements"
    if "readonly" in m or "read-only" in m:
        return "tried to write"
    if ":" in m:
        return m.split(":", 1)[0].strip()   # "no such column: T1.x" -> "no such column"
    return m[:60]
