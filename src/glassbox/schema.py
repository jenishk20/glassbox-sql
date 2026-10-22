"""Turn a BIRD database into schema text for a prompt.

Everything here comes from BIRD's own files:
  * table and column structure: the .sqlite file (PRAGMA table_info / foreign_key_list)
  * column descriptions: dev_databases/<db>/database_description/<table>.csv
  * example values and question-matched values: read from the database itself
"""

from __future__ import annotations

import csv
import re
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .sqlexec import connect_readonly

_PLAIN_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# SQLite's keyword list (https://sqlite.org/lang_keywords.html).
_KEYWORDS = set("""ABORT ACTION ADD AFTER ALL ALTER ALWAYS ANALYZE AND AS ASC ATTACH AUTOINCREMENT BEFORE
BEGIN BETWEEN BY CASCADE CASE CAST CHECK COLLATE COLUMN COMMIT CONFLICT CONSTRAINT CREATE CROSS CURRENT
CURRENT_DATE CURRENT_TIME CURRENT_TIMESTAMP DATABASE DEFAULT DEFERRABLE DEFERRED DELETE DESC DETACH
DISTINCT DO DROP EACH ELSE END ESCAPE EXCEPT EXCLUDE EXCLUSIVE EXISTS EXPLAIN FAIL FILTER FIRST FOLLOWING
FOR FOREIGN FROM FULL GENERATED GLOB GROUP GROUPS HAVING IF IGNORE IMMEDIATE IN INDEX INDEXED INITIALLY
INNER INSERT INSTEAD INTERSECT INTO IS ISNULL JOIN KEY LAST LEFT LIKE LIMIT MATCH MATERIALIZED NATURAL NO
NOT NOTHING NOTNULL NULL NULLS OF OFFSET ON OR ORDER OTHERS OUTER OVER PARTITION PLAN PRAGMA PRECEDING
PRIMARY QUERY RAISE RANGE RECURSIVE REFERENCES REGEXP REINDEX RELEASE RENAME REPLACE RESTRICT RETURNING
RIGHT ROLLBACK ROW ROWS SAVEPOINT SELECT SET TABLE TEMP TEMPORARY THEN TIES TO TRANSACTION TRIGGER
UNBOUNDED UNION UNIQUE UPDATE USING VACUUM VALUES VIEW VIRTUAL WHEN WHERE WINDOW WITH WITHOUT""".split())


def quote(name: str) -> str:
    """How a name is shown to the model: backticked only when SQLite needs it."""
    if _PLAIN_NAME.match(name) and name.upper() not in _KEYWORDS:
        return name
    return ident(name)


def ident(name: str) -> str:
    """Always-quoted name, for the queries this module runs itself."""
    return "`" + name.replace("`", "``") + "`"


@dataclass
class Column:
    name: str
    type: str
    pk: bool


@dataclass
class Table:
    name: str
    columns: list[Column]
    foreign_keys: list[tuple[list[str], str, list[str]]] = field(default_factory=list)


@lru_cache(maxsize=None)
def read_tables(db_path: str) -> list[Table]:
    conn = connect_readonly(Path(db_path))
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]
        tables = []
        for t in names:
            cols = [Column(name=r[1], type=(r[2] or "").upper(), pk=r[5] > 0)
                    for r in conn.execute(f"PRAGMA table_info({ident(t)})")]
            fks: dict[int, tuple[list[str], str, list[str]]] = {}
            for r in conn.execute(f"PRAGMA foreign_key_list({ident(t)})"):
                fk_id, ref_table, src, dst = r[0], r[2], r[3], r[4]
                entry = fks.setdefault(fk_id, ([], ref_table, []))
                entry[0].append(src)
                entry[2].append(dst or src)
            tables.append(Table(name=t, columns=cols, foreign_keys=list(fks.values())))
        return tables
    finally:
        conn.close()


def raw_ddl(db_path: str) -> str:
    """The CREATE TABLE text exactly as stored in the database (what v1 used)."""
    conn = connect_readonly(Path(db_path))
    try:
        rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL").fetchall()
    finally:
        conn.close()
    return "\n\n".join(s.strip() + ";" for (s,) in rows if "sqlite_sequence" not in s.lower())


def _clean(text: str | None, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


@lru_cache(maxsize=None)
def read_descriptions(db_folder: str) -> dict[tuple[str, str], str]:
    """(table, column) -> one-line description, from BIRD's database_description CSVs."""
    out: dict[tuple[str, str], str] = {}
    for path in sorted(Path(db_folder, "database_description").glob("*.csv")):
        for encoding in ("utf-8-sig", "latin-1"):
            try:
                rows = list(csv.DictReader(path.open(encoding=encoding, newline="")))
                break
            except UnicodeDecodeError:
                continue
        table = path.stem.lower()
        for row in rows:
            row = {(k or "").strip().lower(): v for k, v in row.items()}
            original = (row.get("original_column_name") or "").strip()
            if not original:
                continue
            # Keep only text that adds something beyond the column's own name.
            parts: list[str] = []
            seen = {original.lower()}
            for key, limit in (("column_name", 100), ("column_description", 200), ("value_description", 200)):
                piece = _clean(row.get(key), limit)
                if piece and piece.lower() not in seen:
                    parts.append(piece)
                    seen.add(piece.lower())
            if parts:
                out[(table, original.lower())] = "; ".join(parts)
    return out


def _format_value(v) -> str | None:
    if isinstance(v, bytes):
        return None
    if isinstance(v, str):
        return "'" + _clean(v, 40).replace("'", "''") + "'"
    return repr(v)


@lru_cache(maxsize=None)
def example_values(db_path: str, table: str, column: str, k: int = 2) -> tuple[str, ...]:
    conn = connect_readonly(Path(db_path))
    try:
        rows = conn.execute(
            f"SELECT DISTINCT {ident(column)} FROM {ident(table)} "
            f"WHERE {ident(column)} IS NOT NULL AND {ident(column)} != '' LIMIT {k}").fetchall()
    except sqlite3.Error:
        return ()
    finally:
        conn.close()
    return tuple(s for s in (_format_value(r[0]) for r in rows) if s)


@lru_cache(maxsize=None)
def render_schema(db_path: str, db_folder: str, descriptions: bool, examples: bool) -> str:
    """CREATE TABLE-style schema, optionally with a description and examples comment per column."""
    descs = read_descriptions(db_folder) if descriptions else {}
    blocks = []
    for t in read_tables(db_path):
        lines = []
        for c in t.columns:
            comment = []
            if descs.get((t.name.lower(), c.name.lower())):
                comment.append(descs[(t.name.lower(), c.name.lower())])
            if examples:
                vals = example_values(db_path, t.name, c.name)
                if vals:
                    comment.append("examples: " + ", ".join(vals))
            line = f"  {quote(c.name)} {c.type}".rstrip() + ","
            lines.append(line + (" -- " + " | ".join(comment) if comment else ""))
        pks = [c.name for c in t.columns if c.pk]
        if pks:
            lines.append(f"  PRIMARY KEY ({', '.join(quote(p) for p in pks)}),")
        for src, ref, dst in t.foreign_keys:
            lines.append(f"  FOREIGN KEY ({', '.join(map(quote, src))}) "
                         f"REFERENCES {quote(ref)} ({', '.join(map(quote, dst))}),")
        # Drop the trailing comma of the last entry (it sits before any "--" comment).
        last = lines[-1]
        head, sep, tail = last.partition(" -- ")
        lines[-1] = head.rstrip(",") + (sep + tail if sep else "")
        blocks.append(f"CREATE TABLE {quote(t.name)} (\n" + "\n".join(lines) + "\n);")
    return "\n\n".join(blocks)


# ---- values that appear in the question -------------------------------------------------

_STOPWORDS = set("""a an and are as at be by for from has have how in is it its of on or that the
their them there these this to was were what when where which who whom whose why will with
list name names give show find many much number total average all each any per""".split())

_TEXT_TYPES = ("CHAR", "TEXT", "CLOB")   # SQLite's rules for text affinity


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[\w'&.+-]+", s.lower())).strip(".")


@lru_cache(maxsize=None)
def value_index(db_path: str, max_len: int = 60) -> dict[str, list[tuple[str, str, str]]]:
    """normalized value -> [(table, column, original value)] for short text values in the DB."""
    index: dict[str, list[tuple[str, str, str]]] = {}
    conn = connect_readonly(Path(db_path))
    try:
        for t in read_tables(db_path):
            for c in t.columns:
                is_text = c.type == "" or any(k in c.type for k in _TEXT_TYPES)
                if not is_text:
                    continue
                try:
                    rows = conn.execute(
                        f"SELECT DISTINCT {ident(c.name)} FROM {ident(t.name)} "
                        f"WHERE typeof({ident(c.name)}) = 'text' "
                        f"AND length({ident(c.name)}) BETWEEN 2 AND {max_len}").fetchall()
                except sqlite3.Error:
                    continue
                for (v,) in rows:
                    key = _norm(v)
                    if len(key) >= 3:
                        index.setdefault(key, []).append((t.name, c.name, v))
    finally:
        conn.close()
    return index


def matched_values(db_path: str, question: str, max_ngram: int = 6, limit: int = 10) -> list[tuple[str, str, str]]:
    """Database values whose text appears word-for-word in the question, longest matches first."""
    index = value_index(db_path)
    words = _norm(question).split()
    found: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for n in range(min(max_ngram, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            gram = " ".join(words[i:i + n])
            if n == 1 and (gram in _STOPWORDS or len(gram) < 3 or gram.isdigit()):
                continue
            for hit in index.get(gram, [])[:3]:
                if hit not in seen:
                    seen.add(hit)
                    found.append(hit)
                    if len(found) >= limit:
                        return found
    return found
