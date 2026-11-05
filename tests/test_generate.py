"""Resuming a run must ask again any question whose request failed."""

import json

from glassbox.data import Example
from glassbox.generate import Decoding, run_generation
from glassbox.prompts import Prompt


def test_resume_retries_failed_questions(tmp_path):
    examples = [Example(idx=i, question_id=i, db_id="db", question="q", evidence="",
                        gold_sql=f"SELECT {i}", difficulty="simple") for i in range(2)]
    out = tmp_path / "generations.jsonl"
    out.write_text(
        json.dumps({"idx": 0, "question_id": 0, "samples": [{"text": None, "error": "APIConnectionError"}]}) + "\n"
        + json.dumps({"idx": 1, "question_id": 1, "samples": [{"text": "```sql\nSELECT 1\n```"}]}) + "\n")

    prompts = {e.idx: Prompt(system=None, user="q", reasoning=False) for e in examples}
    run_generation(examples, prompts, None, "gold", "gold", "chat", Decoding(), out, workers=1)

    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["idx"] for r in lines] == [0, 1, 0]          # only question 0 was asked again
    assert "error" not in lines[-1]["samples"][0]          # and the new answer replaced the failure


def test_reuse_copies_finished_answers_only(tmp_path):
    from glassbox.generate import reuse_answers

    meta = {"model": "m", "endpoint": "wandb", "format": "chat", "prompt": "p0-schema", "dataset": {"sha256": "x"},
            "limit": None, "decoding": {"temperature": 0.0, "top_p": 1.0, "n": 1, "repetition_penalty": None,
                                        "max_tokens": 16}}
    old = tmp_path / "old"
    old.mkdir()
    (old / "meta.json").write_text(json.dumps({**meta, "run_name": "old"}))
    (old / "generations.jsonl").write_text(
        json.dumps({"idx": 0, "samples": [{"text": "```sql\nSELECT 1\n```", "finish_reason": "stop"}]}) + "\n"
        + json.dumps({"idx": 1, "samples": [{"text": "", "finish_reason": "length"}]}) + "\n")

    new_meta = {**meta, "decoding": {**meta["decoding"], "max_tokens": 64}}
    out = tmp_path / "generations.jsonl"
    info = reuse_answers(old, new_meta, out)
    assert info["copied"] == 1 and info["asked_again"] == [1]
    assert [json.loads(line)["idx"] for line in out.read_text().splitlines()] == [0]
