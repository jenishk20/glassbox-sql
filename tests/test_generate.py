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
