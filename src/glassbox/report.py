"""Read finished runs: summary tables, failure breakdowns, paired comparisons, traces."""

from __future__ import annotations

import json
import random
from pathlib import Path

from .stats import mcnemar_exact


def load_summary(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "summary.json").read_text())


def load_scores(run_dir: Path) -> dict[int, dict]:
    return {r["idx"]: r for r in map(json.loads, (Path(run_dir) / "scores.jsonl").open()) if r}


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}"


def print_table(run_dirs: list[Path]) -> None:
    rows = [load_summary(d) for d in run_dirs]
    width = max(12, *(len(r["run"]) for r in rows))
    print(f"{'run':<{width}}  {'n':>5}  {'EX%':>5}  {'95% CI':>13}  {'simple':>6}  {'moder.':>6}  "
          f"{'chall.':>6}  {'no_sql':>6}  {'sql_err':>7}  {'timeout':>7}  {'wrong':>6}  {'tok/q':>6}")
    for r in rows:
        a, cats, d = r["accuracy"], r["categories"], r["by_difficulty"]
        no_sql = cats.get("no_sql", 0) + cats.get("truncated", 0) + cats.get("api_error", 0)
        diff = [_pct(d[k]["acc"]) if k in d else "    -" for k in ("simple", "moderate", "challenging")]
        flag = "" if r["complete"] else "   <-- INCOMPLETE: not a full-split number"
        print(f"{r['run']:<{width}}  {a['n']:>5}  {_pct(a['acc'])}  "
              f"[{_pct(a['ci95'][0])},{_pct(a['ci95'][1])}]  {diff[0]:>6}  {diff[1]:>6}  {diff[2]:>6}  "
              f"{no_sql:>6}  {cats.get('sql_error', 0):>7}  {cats.get('timeout', 0):>7}  "
              f"{cats.get('wrong_result', 0):>6}  {r['tokens'].get('completion_per_question', 0):>6.0f}{flag}")
        if "sampling" in r:
            s, k = r["sampling"], r["sampling"]["k"]
            print(f"{'':<{width}}  maj@{k} {_pct(s[f'maj@{k}']['acc'])}%   pass@{k} {_pct(s[f'pass@{k}']['acc'])}%   "
                  f"mean single sample {_pct(s['mean_per_sample'])}%")


def print_breakdown(run_dir: Path) -> None:
    r = load_summary(run_dir)
    n = r["accuracy"]["n"]
    print(f"\n{r['run']}  ({r['model']}, prompt {r['prompt']}, {r['split']}, n={n})")
    print("  what happened to each question (first sample):")
    for cat, c in r["categories"].items():
        print(f"    {cat:<14} {c:>5}  {_pct(c / n)}%")
    if r["sql_error_types"]:
        print("  SQL errors by type:")
        for t, c in r["sql_error_types"].items():
            print(f"    {t:<40} {c:>5}")
    if r["wrong_kinds"]:
        print("  wrong results by kind (columns = wrong SELECT list, row_count = wrong filter/join, "
              "values = right shape, wrong numbers or strings):")
        for t, c in r["wrong_kinds"].items():
            print(f"    {t:<14} {c:>5}")
    print(f"  SQL found by: {r['extracted_by']}")
    gf = r["categories"].get("gold_failed", 0)
    if gf:
        print(f"  NOTE: {gf} gold (answer-key) queries failed even with a 5-minute limit; "
              "they count as wrong, as in the official scorer. Check them before reporting.")


def compare(run_a: Path, run_b: Path, show: int = 15) -> None:
    a, b = load_scores(run_a), load_scores(run_b)
    common = sorted(set(a) & set(b))
    both = [i for i in common if a[i]["correct"] and b[i]["correct"]]
    only_a = [i for i in common if a[i]["correct"] and not b[i]["correct"]]
    only_b = [i for i in common if b[i]["correct"] and not a[i]["correct"]]
    neither = len(common) - len(both) - len(only_a) - len(only_b)
    acc_a = (len(both) + len(only_a)) / len(common)
    acc_b = (len(both) + len(only_b)) / len(common)
    p = mcnemar_exact(len(only_a), len(only_b))
    name_a, name_b = Path(run_a).name, Path(run_b).name
    print(f"A = {name_a}\nB = {name_b}\non {len(common)} shared questions:")
    print(f"  A {_pct(acc_a)}%   B {_pct(acc_b)}%   difference B-A {100 * (acc_b - acc_a):+.1f} points")
    print(f"  both right {len(both)}, only A {len(only_a)}, only B {len(only_b)}, neither {neither}")
    verdict = "a real difference" if p < 0.05 else "NOT distinguishable from noise"
    print(f"  McNemar exact p = {p:.4f}  ->  {verdict} (at 0.05)")
    for label, ids in (("only A right", only_a), ("only B right", only_b)):
        if ids:
            print(f"  {label} (idx): {ids[:show]}{' ...' if len(ids) > show else ''}")


def write_traces(run_dir: Path, examples_by_idx: dict, category: str | None, n: int, seed: int,
                 difficulty: str | None, db: str | None, idxs: list[int] | None, out: Path) -> int:
    """Markdown file with full traces: question, gold vs predicted SQL, results, raw output, prompt."""
    scores = load_scores(run_dir)
    gens = {g["idx"]: g for g in map(json.loads, (Path(run_dir) / "generations.jsonl").open())}
    prompts = {}
    prompts_path = Path(run_dir) / "prompts.jsonl"
    if prompts_path.exists():
        prompts = {p["idx"]: p for p in map(json.loads, prompts_path.open())}

    pool = [s for s in scores.values()
            if (idxs is None or s["idx"] in idxs)
            and (category is None or s["samples"][0]["category"] == category)
            and (difficulty is None or s["difficulty"] == difficulty)
            and (db is None or s["db_id"] == db)]
    random.Random(seed).shuffle(pool)
    pool = sorted(pool[:n], key=lambda s: s["idx"])

    lines = [f"# Traces: {Path(run_dir).name}",
             f"{len(pool)} questions; filter: category={category}, difficulty={difficulty}, db={db}; seed {seed}\n"]
    for s in pool:
        ex, first = examples_by_idx[s["idx"]], s["samples"][0]
        sample = gens[s["idx"]]["samples"][0]
        label = first["category"] + (f" ({first.get('error_type') or first.get('wrong_kind')})"
                                     if first.get("error_type") or first.get("wrong_kind") else "")
        lines += [f"## idx {s['idx']} · {ex.db_id} · {ex.difficulty} · {label}",
                  f"**Question:** {ex.question}  ",
                  f"**Evidence:** {ex.evidence or '(none)'}\n",
                  "**Gold SQL**", f"```sql\n{ex.gold_sql.strip()}\n```",
                  "**Predicted SQL**", f"```sql\n{first['sql'] or '(nothing extracted)'}\n```"]
        if first.get("detail"):
            lines.append(f"**Error:** `{first['detail']}`\n")
        if "rows" in s.get("gold", {}):
            lines.append(f"**Gold result** ({s['gold']['n_distinct']} distinct rows): `{s['gold']['rows']}`  ")
        if "rows" in first.get("result", {}):
            lines.append(f"**Predicted result** ({first['result']['n_distinct']} distinct rows): "
                         f"`{first['result']['rows']}`\n")
        if sample.get("reasoning"):
            lines += ["<details><summary>Model reasoning</summary>\n", "```text",
                      sample["reasoning"].strip(), "```\n</details>\n"]
        lines += ["<details><summary>Full model output</summary>\n", "```text",
                  (sample.get("text") or sample.get("error") or "").strip(), "```\n</details>\n"]
        if s["idx"] in prompts:
            p = prompts[s["idx"]]
            shown = (f"[system]\n{p['system']}\n\n[user]\n" if p.get("system") else "") + p["user"]
            lines += ["<details><summary>Prompt sent to the model</summary>\n", "```text",
                      shown.strip(), "```\n</details>\n"]
        lines.append("---\n")
    out.write_text("\n".join(lines))
    return len(pool)
