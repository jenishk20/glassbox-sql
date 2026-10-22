"""Prompts. The p0..p5 ladder adds exactly one ingredient per step:

  p0-schema        schema + question
  p1-evidence      + BIRD's evidence hint
  p2-descriptions  + a description comment on every column
  p3-examples      + two example values per column
  p4-matches       + database values that appear in the question
  p5-reasoning     + "think step by step" before the final SQL

`v1` reproduces the prompt the v1 study used (raw DDL, "return only the SQL"), so its
27% result can be explained rather than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import schema as S
from .data import Example, db_dir, db_file

LADDER = ["p0-schema", "p1-evidence", "p2-descriptions", "p3-examples", "p4-matches", "p5-reasoning"]
PROMPTS = LADDER + ["v1"]

DIRECT = "Return only the SQL query, in a ```sql code block."
REASONING = ("First think step by step about which tables, columns and conditions you need. "
             "Then write the final SQL query in a ```sql code block.")

V1_SYSTEM = ("You are an expert SQL query generator. Given a database schema and a natural "
             "language question, generate a single valid SQL query that answers the question. "
             "Output ONLY the SQL query, nothing else.")


@dataclass(frozen=True)
class Prompt:
    system: str | None
    user: str
    reasoning: bool        # True if the model is asked to think before answering

    def completion_text(self) -> str:
        """Plain-text version for base models: the answer fence is opened for the model."""
        if self.reasoning:
            raise ValueError("reasoning prompts need --format chat; a base model is given "
                             "an opened ```sql fence and should write SQL immediately")
        head = f"{self.system}\n\n" if self.system else ""
        return f"{head}{self.user}\n\n```sql\n"


def build(name: str, ex: Example, bird_dir: Path) -> Prompt:
    if name not in PROMPTS:
        raise ValueError(f"unknown prompt {name!r}; choose from {PROMPTS}")
    path = str(db_file(bird_dir, ex.db_id))

    if name == "v1":
        evidence = f"External Knowledge:\n{ex.evidence}\n\n" if ex.evidence else ""
        user = ("Convert the following natural language question into a valid SQL query.\n\n"
                f"Database Schema:\n{S.raw_ddl(path)}\n\n{evidence}Question: {ex.question}\n\n"
                "Return only the SQL query with no explanation.")
        return Prompt(system=V1_SYSTEM, user=user, reasoning=False)

    step = LADDER.index(name)
    schema = S.render_schema(path, str(db_dir(bird_dir, ex.db_id)),
                             descriptions=step >= 2, examples=step >= 3)
    parts = ["You are given a SQLite database and a question about it. "
             "Write one SQL query that answers the question.",
             f"Database schema:\n{schema}"]
    if step >= 4:
        hits = S.matched_values(path, ex.question)
        if hits:
            parts.append("Values in the database that appear in the question:\n" + "\n".join(
                f"  {S.quote(t)}.{S.quote(c)} = '{v}'" for t, c, v in hits))
    if step >= 1 and ex.evidence:
        parts.append(f"Evidence: {ex.evidence}")
    parts.append(f"Question: {ex.question}")
    parts.append(REASONING if step >= 5 else DIRECT)
    return Prompt(system=None, user="\n\n".join(parts), reasoning=step >= 5)
