"""Score a run: extract SQL, execute it, compare with gold, and label every failure.

Every sample gets exactly one category:
  correct       result set equals gold (official BIRD rule)
  api_error     the model call failed
  no_sql        no ```sql block in the answer
  truncated     no SQL because the answer hit max_tokens
  sql_error     SQLite rejected the query (error_type says why)
  timeout       the query ran longer than 30 s
  wrong_result  ran fine, wrong answer (wrong_kind: columns / row_count / values)
  gold_failed   the gold query itself failed; counted wrong, flagged so you can check it
"""

from __future__ import annotations

import json
import pickle
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .data import Example, db_file
from .sqlexec import GOLD_TIMEOUT_S, ExecResult, error_type, execute, results_match
from .stats import wilson_interval

_SQL_FENCE = re.compile(r"```(?:sql|sqlite)\b\s*(.*?)```", re.S | re.I)
_PLAIN_FENCE = re.compile(r"```[ \t]*\n(.*?)```", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)


def extract_sql(sample: dict) -> tuple[str, str]:
    """(sql, how). Takes the LAST sql block, since models often draft before answering."""
    text = sample.get("text")
    if text is None:
        return "", "none"
    if sample.get("prefilled"):              # base model: we opened the fence, it wrote SQL
        return text.strip(), "prefilled"
    text = _THINK.sub("", text)
    for pattern, how in ((_SQL_FENCE, "sql_fence"), (_PLAIN_FENCE, "plain_fence")):
        blocks = pattern.findall(text)
        if blocks and blocks[-1].strip():
            return blocks[-1].strip(), how
    return "", "none"


def _preview(res: ExecResult, rows: int = 5) -> dict:
    distinct = set(res.rows)
    return {
        "n_rows": len(res.rows),
        "n_distinct": len(distinct),
        "n_cols": len(res.rows[0]) if res.rows else None,
        "rows": [[str(v)[:60] for v in r] for r in res.rows[:rows]],
    }


def classify(sample: dict, sql: str, pred: ExecResult | None, gold: ExecResult) -> dict:
    if sample.get("error"):
        return {"category": "api_error", "detail": sample["error"]}
    if not gold.ok:
        return {"category": "gold_failed", "detail": gold.error}
    if not sql:
        return {"category": "truncated" if sample.get("finish_reason") == "length" else "no_sql"}
    assert pred is not None
    if pred.timed_out:
        return {"category": "timeout"}
    if not pred.ok:
        return {"category": "sql_error", "error_type": error_type(pred.error or ""), "detail": pred.error}
    if results_match(pred, gold):
        return {"category": "correct"}
    pred_cols = len(pred.rows[0]) if pred.rows else None
    gold_cols = len(gold.rows[0]) if gold.rows else None
    if pred_cols and gold_cols and pred_cols != gold_cols:
        kind = "columns"
    elif pred.too_large or len(set(pred.rows)) != len(set(gold.rows)):
        kind = "row_count"
    else:
        kind = "values"
    return {"category": "wrong_result", "wrong_kind": kind}


def gold_results(examples: list[Example], bird_dir: Path, sha: str, cache_dir: Path,
                 workers: int, timeout: float = GOLD_TIMEOUT_S) -> dict[int, ExecResult]:
    """Run every gold query once and cache the results, keyed by the split file's hash.

    The cache holds plain Python values (no classes), so it survives code changes.
    """
    cache = cache_dir / f"gold_{sha[:16]}_{int(timeout)}s.pkl"
    if cache.exists():
        cached = pickle.loads(cache.read_bytes())
        if all(e.idx in cached for e in examples):
            return {i: ExecResult(**fields) for i, fields in cached.items()}
    print(f"running {len(examples)} gold queries, up to {timeout:.0f} s each (cached afterwards)...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(lambda e: execute(db_file(bird_dir, e.db_id), e.gold_sql, timeout), examples))
    out = {e.idx: r for e, r in zip(examples, res)}
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps({i: vars(r) for i, r in out.items()}))
    return out


def _vote(scored: list[dict], results: list[ExecResult | None]) -> int | None:
    """Majority vote over result sets of the samples that ran; ties go to the earliest sample."""
    groups: dict[frozenset, list[int]] = {}
    for k, res in enumerate(results):
        if res is not None and res.ok and not res.too_large:
            groups.setdefault(frozenset(res.rows), []).append(k)
    if not groups:
        return None
    best = max(groups.values(), key=lambda ks: (len(ks), -ks[0]))
    return best[0]


def score_run(run_dir: Path, examples: list[Example], bird_dir: Path, meta: dict,
              cache_dir: Path, workers: int = 16) -> dict:
    gens = {}
    for line in (run_dir / "generations.jsonl").open():
        if line.strip():
            rec = json.loads(line)
            gens[rec["idx"]] = rec
    by_idx = {e.idx: e for e in examples}
    scored_examples = [by_idx[i] for i in sorted(gens) if i in by_idx]
    gold = gold_results(examples, bird_dir, meta["dataset"]["sha256"], cache_dir, workers)

    # Execute every extracted SQL of every sample.
    jobs = []
    for e in scored_examples:
        for k, sample in enumerate(gens[e.idx]["samples"]):
            sql, how = extract_sql(sample)
            jobs.append((e, k, sample, sql, how))
    print(f"executing {len(jobs)} predicted queries...", flush=True)

    def run(job):
        e, _, sample, sql, _ = job
        if sample.get("error") or not sql or not gold[e.idx].ok:
            return None
        return execute(db_file(bird_dir, e.db_id), sql)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pred_results = list(pool.map(run, jobs))

    per_q: dict[int, dict] = {}
    for (e, k, sample, sql, how), pred in zip(jobs, pred_results):
        g = gold[e.idx]
        row = per_q.setdefault(e.idx, {
            "idx": e.idx, "question_id": e.question_id, "db_id": e.db_id, "difficulty": e.difficulty,
            "gold": _preview(g) if g.ok else {"error": g.error}, "samples": [], "_results": []})
        entry = {"k": k, "sql": sql, "extracted_by": how, "finish_reason": sample.get("finish_reason"),
                 **classify(sample, sql, pred, g)}
        if pred is not None:
            entry["result"] = _preview(pred) if pred.ok else {"error": pred.error}
        row["samples"].append(entry)
        row["_results"].append(pred)

    n_samples = meta["decoding"]["n"]
    records = []
    for idx in sorted(per_q):
        row = per_q[idx]
        results = row.pop("_results")
        row["correct"] = row["samples"][0]["category"] == "correct"
        if n_samples > 1:
            pick = _vote(row["samples"], results)
            row["vote_pick"] = pick
            row["vote_correct"] = pick is not None and row["samples"][pick]["category"] == "correct"
            row["pass_any"] = any(s["category"] == "correct" for s in row["samples"])
            row["frac_correct"] = sum(s["category"] == "correct" for s in row["samples"]) / len(row["samples"])
        records.append(row)

    with (run_dir / "scores.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    _write_official_predictions(run_dir, records, by_idx)
    summary = summarize(records, meta, n_expected=len(examples), gens=gens)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _write_official_predictions(run_dir: Path, records: list[dict], by_idx: dict[int, Example]) -> None:
    """predict_dev.json in BIRD's own format, so the official evaluation.py can re-check us."""
    sep = "\t----- bird -----\t"
    first = {str(r["idx"]): r["samples"][0]["sql"] + sep + by_idx[r["idx"]].db_id for r in records}
    (run_dir / "predict_dev.json").write_text(json.dumps(first, indent=1, ensure_ascii=False))
    if records and "vote_pick" in records[0]:
        voted = {}
        for r in records:
            k = r["vote_pick"] if r["vote_pick"] is not None else 0
            voted[str(r["idx"])] = r["samples"][k]["sql"] + sep + by_idx[r["idx"]].db_id
        (run_dir / "predict_dev_vote.json").write_text(json.dumps(voted, indent=1, ensure_ascii=False))


def _acc(correct: int, n: int) -> dict:
    lo, hi = wilson_interval(correct, n)
    return {"n": n, "correct": correct, "acc": correct / n if n else 0.0, "ci95": [lo, hi]}


def summarize(records: list[dict], meta: dict, n_expected: int, gens: dict) -> dict:
    n = len(records)
    first = [r["samples"][0] for r in records]
    by_diff = {}
    for d in ("simple", "moderate", "challenging"):
        rows = [r for r in records if r["difficulty"] == d]
        if rows:
            by_diff[d] = _acc(sum(r["correct"] for r in rows), len(rows))
    usage = Counter()
    for g in gens.values():
        usage.update(g.get("usage", {}))
    summary = {
        "run": meta["run_name"],
        "model": meta["model"],
        "prompt": meta["prompt"],
        "format": meta["format"],
        "split": meta["dataset"]["split"],
        "decoding": meta["decoding"],
        "complete": n == n_expected and not meta.get("limit"),
        "n_expected": n_expected,
        "accuracy": _acc(sum(r["correct"] for r in records), n),
        "by_difficulty": by_diff,
        "categories": dict(Counter(s["category"] for s in first).most_common()),
        "sql_error_types": dict(Counter(s["error_type"] for s in first if s.get("error_type")).most_common(12)),
        "wrong_kinds": dict(Counter(s["wrong_kind"] for s in first if s.get("wrong_kind")).most_common()),
        "extracted_by": dict(Counter(s["extracted_by"] for s in first).most_common()),
        "tokens": {**usage, "completion_per_question": round(usage["completion_tokens"] / n, 1) if n else 0},
    }
    if records and "vote_correct" in records[0]:
        k = meta["decoding"]["n"]
        summary["sampling"] = {
            "k": k,
            f"maj@{k}": _acc(sum(r["vote_correct"] for r in records), n),
            f"pass@{k}": _acc(sum(r["pass_any"] for r in records), n),
            "mean_per_sample": sum(r["frac_correct"] for r in records) / n,
        }
    return summary
