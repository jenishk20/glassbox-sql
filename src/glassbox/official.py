"""Re-check a run with BIRD's own scoring script, unchanged.

BIRD's evaluation.py (MIT license) lives at
  https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/evaluation.py
It is downloaded once, run on the run's predict_dev.json, and its accuracy is printed next
to ours. The two can differ only in the cases listed in sqlexec.py (answer-key queries
slower than 30 s, which the official script counts as wrong).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

OFFICIAL_URL = ("https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/main/"
                "bird/llm/src/evaluation.py")


def official_script(cache_dir: Path) -> Path:
    path = cache_dir / "official" / "evaluation.py"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(OFFICIAL_URL) as r:
            path.write_bytes(r.read())
    return path


def run_official(run_dir: Path, split_file: Path, bird_dir: Path, cache_dir: Path, cpus: int = 16) -> dict:
    """Run BIRD's evaluation.py on run_dir/predict_dev.json; return its four accuracies."""
    try:
        import func_timeout  # noqa: F401  (the official script needs it)
    except ImportError:
        raise SystemExit("The official script needs `func_timeout`: run with `uv run --extra verify ...`")

    script = official_script(cache_dir)
    rows = json.loads(split_file.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        (tmp_dir / "predict_dev.json").write_text((run_dir / "predict_dev.json").read_text())
        # The script reads gold SQL as one "SQL<tab>db_id" line per question.
        gold_lines = [" ".join(r["SQL"].split("\n")).replace("\t", " ") + "\t" + r["db_id"] for r in rows]
        (tmp_dir / "dev_gold.sql").write_text("\n".join(gold_lines) + "\n")
        cmd = [sys.executable, str(script),
               "--predicted_sql_path", f"{tmp_dir}/", "--ground_truth_path", f"{tmp_dir}/",
               "--data_mode", "dev", "--db_root_path", f"{bird_dir / 'dev_databases'}/",
               "--num_cpus", str(cpus), "--meta_time_out", "30.0", "--diff_json_path", str(split_file)]
        out = subprocess.run(cmd, capture_output=True, text=True)
    line = next((l for l in out.stdout.splitlines() if l.startswith("accuracy")), None)
    if out.returncode != 0 or line is None:
        raise SystemExit(f"official script failed:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
    simple, moderate, challenging, total = map(float, re.findall(r"[\d.]+", line)[:4])
    return {"simple": simple, "moderate": moderate, "challenging": challenging, "total": total}
