"""Load BIRD dev questions and check that every database they need is present."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

# Split name -> file inside the BIRD dev folder. Both splits use the same 11 databases.
SPLITS = {
    "dev": "dev.json",                  # original release (dev_20240627), what most papers report
    "dev1106": "dev_20251106.json",     # cleaned re-release, Nov 2025 (`glassbox fetch-dev1106`)
}

DEV1106_URL = (
    "https://huggingface.co/datasets/birdsql/bird_sql_dev_20251106/resolve/main/"
    "data/dev_20251106-00000-of-00001.json"
)


@dataclass(frozen=True)
class Example:
    idx: int            # position in the split file; the official scorer aligns on this
    question_id: int
    db_id: str
    question: str
    evidence: str       # BIRD's "external knowledge" hint, may be empty
    gold_sql: str
    difficulty: str


def db_dir(bird_dir: Path, db_id: str) -> Path:
    return Path(bird_dir) / "dev_databases" / db_id


def db_file(bird_dir: Path, db_id: str) -> Path:
    return db_dir(bird_dir, db_id) / f"{db_id}.sqlite"


def _parse_json_or_jsonl(raw: bytes) -> list[dict]:
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise ValueError("expected a JSON list of questions")
    return rows


def load_split(bird_dir: Path, split: str) -> tuple[list[Example], dict]:
    """Return the examples and a fingerprint of the file they came from."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; choose from {sorted(SPLITS)}")
    path = Path(bird_dir) / SPLITS[split]
    if not path.exists():
        hint = " Run `python -m glassbox fetch-dev1106` first." if split == "dev1106" else ""
        raise FileNotFoundError(f"{path} not found.{hint}")

    raw = path.read_bytes()
    examples = [
        Example(
            idx=i,
            question_id=int(r["question_id"]),
            db_id=r["db_id"],
            question=r["question"].strip(),
            evidence=(r.get("evidence") or "").strip(),
            gold_sql=r["SQL"],
            difficulty=r.get("difficulty") or "unknown",
        )
        for i, r in enumerate(_parse_json_or_jsonl(raw))
    ]

    missing = sorted({e.db_id for e in examples if not db_file(bird_dir, e.db_id).exists()})
    if missing:
        raise FileNotFoundError(f"databases missing under {bird_dir}/dev_databases: {missing}")

    info = {
        "split": split,
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "n": len(examples),
    }
    return examples, info


def fetch_dev1106(bird_dir: Path) -> Path:
    """Download the Nov 2025 dev questions next to the original dev.json (databases are shared)."""
    import urllib.request

    out = Path(bird_dir) / SPLITS["dev1106"]
    with urllib.request.urlopen(DEV1106_URL) as resp:
        rows = _parse_json_or_jsonl(resp.read())
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    return out
