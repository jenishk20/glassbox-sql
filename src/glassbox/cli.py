"""Command line: python -m glassbox <command> ...

  run           generate answers with a model, then score them
  score         re-score an existing run (no model calls)
  report        summary table for one or more runs (--detail for failure breakdowns)
  compare       paired comparison of two runs on the same questions
  traces        write failure traces to a markdown file you can read
  verify-official  re-score a run with BIRD's own evaluation.py and compare
  show-prompt   print the exact prompt for one question
  models        list model ids served by an endpoint
  fetch-dev1106 download the Nov 2025 cleaned dev questions
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .data import SPLITS, fetch_dev1106, load_split
from .generate import ENDPOINTS, Decoding, decoding_dict, make_client, run_generation, server_info, slug
from .prompts import PROMPTS, build
from .report import compare, print_breakdown, print_table, write_traces
from .score import score_run

DEFAULT_BIRD_DIR = os.environ.get("BIRD_DIR", "data/bird/dev_20240627")
RUNS_DIR = Path(os.environ.get("RUNS_DIR", "runs"))

# Settings that must match for a run to be resumed instead of started fresh.
RESUME_KEYS = ("model", "endpoint", "format", "prompt", "decoding", "dataset", "limit", "tag")


def _load_dotenv(path: Path = Path(".env")) -> None:
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if v.strip() and not os.environ.get(k.strip()):
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5)
        return (out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")) or None
    except Exception:
        return None


def _run_name(a, dec: Decoding) -> str:
    parts = [slug(a.model) if a.endpoint != "gold" else "gold", a.prompt, a.format, f"t{dec.temperature:g}"]
    if dec.n > 1:
        parts.append(f"n{dec.n}")
    if dec.repetition_penalty:
        parts.append(f"rp{dec.repetition_penalty:g}")
    if a.tag:
        parts.append(a.tag)
    if a.limit:
        parts.append(f"limit{a.limit}")
    return "__".join(parts)


def cmd_run(a) -> None:
    bird_dir = Path(a.bird_dir)
    examples, info = load_split(bird_dir, a.split)
    if a.limit:
        examples = examples[: a.limit]
        print(f"WARNING: --limit {a.limit}. This run is a smoke test, not a reportable number.")

    reasoning_prompt = a.prompt == "p5-reasoning"
    max_tokens = a.max_tokens or (16384 if a.endpoint == "wandb" else 2048 if reasoning_prompt else 1024)
    dec = Decoding(temperature=a.temperature, top_p=a.top_p, max_tokens=max_tokens, n=a.n,
                   repetition_penalty=a.repetition_penalty)
    if a.n > 1 and a.temperature == 0:
        raise SystemExit("--n > 1 with temperature 0 gives n identical answers; set --temperature (e.g. 0.8)")

    name = a.name or _run_name(a, dec)
    run_dir = RUNS_DIR / a.split / name
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_name": name, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": a.model if a.endpoint != "gold" else "gold-sql", "endpoint": a.endpoint,
        "base_url": a.base_url or ENDPOINTS[a.endpoint]["base_url"], "format": a.format,
        "prompt": a.prompt, "decoding": decoding_dict(dec),
        "dataset": info, "limit": a.limit, "tag": a.tag,
        "harness_version": __version__, "git_commit": _git_commit(),
        "host": platform.node(), "python": sys.version.split()[0],
    }
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        diffs = [k for k in RESUME_KEYS if old.get(k) != meta.get(k)]
        if diffs:
            raise SystemExit(f"{run_dir} exists with different settings ({diffs}). "
                             "Use --tag or --name to start a separate run.")
        meta["created"] = old["created"]
        meta["resumed"] = old.get("resumed", 0) + 1

    print(f"building {len(examples)} prompts ({a.prompt})...", flush=True)
    prompts = {e.idx: build(a.prompt, e, bird_dir) for e in examples}
    if a.format == "completion":
        for p in prompts.values():
            p.completion_text()          # fails early if the prompt needs chat format
    with (run_dir / "prompts.jsonl").open("w") as f:
        for e in examples:
            p = prompts[e.idx]
            sent = {"system": p.system, "user": p.user} if a.format == "chat" else {"user": p.completion_text()}
            f.write(json.dumps({"idx": e.idx, **sent}, ensure_ascii=False) + "\n")

    client = None if a.endpoint == "gold" else make_client(a.endpoint, a.base_url)
    if client is not None:
        meta["server"] = server_info(client, a.endpoint)
    meta_path.write_text(json.dumps(meta, indent=2))

    workers = a.workers or (64 if a.endpoint == "local" else 16)
    run_generation(examples, prompts, client, a.endpoint, a.model, a.format, dec,
                   run_dir / "generations.jsonl", workers)
    score_run(run_dir, examples, bird_dir, meta, RUNS_DIR / ".cache", a.score_workers)
    print()
    print_table([run_dir])
    print_breakdown(run_dir)
    print(f"\nrun folder: {run_dir}")


def cmd_score(a) -> None:
    for d in a.run_dirs:
        run_dir = Path(d)
        meta = json.loads((run_dir / "meta.json").read_text())
        bird_dir = Path(a.bird_dir or Path(meta["dataset"]["path"]).parent)
        examples, info = load_split(bird_dir, meta["dataset"]["split"])
        if info["sha256"] != meta["dataset"]["sha256"]:
            raise SystemExit(f"{bird_dir} has a different {meta['dataset']['split']} file than this run used")
        if meta.get("limit"):
            examples = examples[: meta["limit"]]
        score_run(run_dir, examples, bird_dir, meta, RUNS_DIR / ".cache", a.score_workers)
    print_table([Path(d) for d in a.run_dirs])


def cmd_report(a) -> None:
    print_table([Path(d) for d in a.run_dirs])
    if a.detail:
        for d in a.run_dirs:
            print_breakdown(Path(d))


def cmd_traces(a) -> None:
    run_dir = Path(a.run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    bird_dir = Path(a.bird_dir or Path(meta["dataset"]["path"]).parent)
    examples, _ = load_split(bird_dir, meta["dataset"]["split"])
    idxs = [int(i) for i in a.idx.split(",")] if a.idx else None
    out = Path(a.out) if a.out else run_dir / f"traces_{a.category or 'all'}.md"
    n = write_traces(run_dir, {e.idx: e for e in examples}, a.category, a.n, a.seed,
                     a.difficulty, a.db, idxs, out)
    print(f"wrote {n} traces to {out}")


def cmd_verify_official(a) -> None:
    from .official import run_official
    from .report import load_summary

    run_dir = Path(a.run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    ours = load_summary(run_dir)
    if not ours["complete"]:
        raise SystemExit("verify-official needs a complete run (all questions of the split)")
    bird_dir = Path(a.bird_dir or Path(meta["dataset"]["path"]).parent)
    split_file = Path(meta["dataset"]["path"])
    print("running BIRD's official evaluation.py on this run's predict_dev.json ...", flush=True)
    official = run_official(run_dir, split_file, bird_dir, RUNS_DIR / ".cache", a.cpus)
    mine = {d: 100 * ours["by_difficulty"][d]["acc"] for d in ("simple", "moderate", "challenging")}
    mine["total"] = 100 * ours["accuracy"]["acc"]
    print(f"\n{'':<12} {'glassbox':>9} {'official':>9}")
    for k in ("simple", "moderate", "challenging", "total"):
        print(f"{k:<12} {mine[k]:>9.2f} {official[k]:>9.2f}")
    gap = abs(mine["total"] - official["total"]) * ours["accuracy"]["n"] / 100
    print(f"\ndifference: {gap:.0f} question(s)" + ("" if gap < 0.5 else
          "  (expected only for questions whose answer-key query takes over 30 s; see sqlexec.py)"))
    (run_dir / "official_check.json").write_text(json.dumps({"glassbox": mine, "official": official}, indent=2))


def cmd_show_prompt(a) -> None:
    examples, _ = load_split(Path(a.bird_dir), a.split)
    p = build(a.prompt, examples[a.idx], Path(a.bird_dir))
    text = p.completion_text() if a.format == "completion" else (
        (f"[system]\n{p.system}\n\n[user]\n" if p.system else "") + p.user)
    print(text)
    print(f"\n[{len(text):,} characters]")


def cmd_models(a) -> None:
    client = make_client(a.endpoint, a.base_url)
    for m in sorted(client.models.list().data, key=lambda m: m.id):
        print(m.id)


def main(argv: list[str] | None = None) -> None:
    _load_dotenv()
    p = argparse.ArgumentParser(prog="glassbox", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="generate with a model, then score")
    r.add_argument("--model", required=True, help="model id, e.g. Qwen/Qwen2.5-Coder-7B-Instruct")
    r.add_argument("--endpoint", choices=sorted(ENDPOINTS), default="local")
    r.add_argument("--base-url", help="override the endpoint URL (e.g. another port)")
    r.add_argument("--format", choices=["chat", "completion"], default="chat",
                   help="chat for instruct/frontier models; completion for base models")
    r.add_argument("--prompt", choices=PROMPTS, default="p1-evidence")
    r.add_argument("--split", choices=sorted(SPLITS), default="dev")
    r.add_argument("--bird-dir", default=DEFAULT_BIRD_DIR)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--top-p", type=float, default=1.0)
    r.add_argument("--max-tokens", type=int, help="default: 1024 direct, 2048 reasoning, 16384 on W&B")
    r.add_argument("--n", type=int, default=1, help="samples per question (for majority vote / pass@k)")
    r.add_argument("--repetition-penalty", type=float, help="only to reproduce v1 (it used 1.1)")
    r.add_argument("--limit", type=int, help="first N questions only (smoke tests)")
    r.add_argument("--tag", help="extra label for the run name, e.g. 4bit")
    r.add_argument("--name", help="explicit run name")
    r.add_argument("--workers", type=int, help="parallel requests (default 64 local, 16 W&B)")
    r.add_argument("--score-workers", type=int, default=16)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("score", help="re-score existing runs")
    s.add_argument("run_dirs", nargs="+")
    s.add_argument("--bird-dir")
    s.add_argument("--score-workers", type=int, default=16)
    s.set_defaults(func=cmd_score)

    rp = sub.add_parser("report", help="summary table")
    rp.add_argument("run_dirs", nargs="+")
    rp.add_argument("--detail", action="store_true", help="also print failure breakdowns")
    rp.set_defaults(func=cmd_report)

    c = sub.add_parser("compare", help="paired comparison of two runs")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.set_defaults(func=lambda a: compare(Path(a.run_a), Path(a.run_b)))

    t = sub.add_parser("traces", help="write readable traces to markdown")
    t.add_argument("run_dir")
    t.add_argument("--category", help="e.g. wrong_result, sql_error, no_sql, correct")
    t.add_argument("--difficulty", choices=["simple", "moderate", "challenging"])
    t.add_argument("--db", help="only this database")
    t.add_argument("--idx", help="comma-separated question indexes")
    t.add_argument("--n", type=int, default=25)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out")
    t.add_argument("--bird-dir")
    t.set_defaults(func=cmd_traces)

    v = sub.add_parser("verify-official", help="re-score a run with BIRD's own evaluation.py")
    v.add_argument("run_dir")
    v.add_argument("--bird-dir")
    v.add_argument("--cpus", type=int, default=16)
    v.set_defaults(func=cmd_verify_official)

    sp = sub.add_parser("show-prompt", help="print the prompt for one question")
    sp.add_argument("--prompt", choices=PROMPTS, default="p1-evidence")
    sp.add_argument("--format", choices=["chat", "completion"], default="chat")
    sp.add_argument("--idx", type=int, default=0)
    sp.add_argument("--split", choices=sorted(SPLITS), default="dev")
    sp.add_argument("--bird-dir", default=DEFAULT_BIRD_DIR)
    sp.set_defaults(func=cmd_show_prompt)

    m = sub.add_parser("models", help="list model ids on an endpoint")
    m.add_argument("--endpoint", choices=["local", "wandb"], default="wandb")
    m.add_argument("--base-url")
    m.set_defaults(func=cmd_models)

    f = sub.add_parser("fetch-dev1106", help="download the Nov 2025 dev questions")
    f.add_argument("--bird-dir", default=DEFAULT_BIRD_DIR)
    f.set_defaults(func=lambda a: print(f"saved {fetch_dev1106(Path(a.bird_dir))}"))

    args = p.parse_args(argv)
    args.func(args)
