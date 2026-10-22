"""Ask a model for SQL. One code path for every model, through an OpenAI-compatible API:

  local  a vLLM server on the cluster (`vllm serve <model> --generation-config vllm`)
  wandb  W&B Inference (frontier and open models)
  gold   no model at all: answers with the gold SQL, to check data loading and scoring

Every sampling setting is sent explicitly with each request. A model's own
generation_config.json is never allowed to fill gaps (v1 silently picked up
repetition_penalty=1.1 that way).
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import Example
from .prompts import Prompt

ENDPOINTS = {
    "local": {"base_url": "http://localhost:8000/v1", "key_env": None},
    "wandb": {"base_url": "https://api.inference.wandb.ai/v1", "key_env": "WANDB_API_KEY"},
    "gold": {"base_url": None, "key_env": None},
}


@dataclass(frozen=True)
class Decoding:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    n: int = 1
    repetition_penalty: float | None = None   # None: not sent to W&B; 1.0 (off) for local

    def extra_body(self, endpoint: str) -> dict:
        """vLLM-only knobs, pinned to their neutral values unless asked otherwise."""
        if endpoint == "local":
            return {"top_k": -1, "min_p": 0.0, "repetition_penalty": self.repetition_penalty or 1.0}
        if endpoint == "wandb" and self.repetition_penalty is not None:
            return {"repetition_penalty": self.repetition_penalty}
        return {}


def make_client(endpoint: str, base_url: str | None = None):
    from openai import OpenAI

    cfg = ENDPOINTS[endpoint]
    key = "not-needed"
    if cfg["key_env"]:
        key = os.environ.get(cfg["key_env"], "")
        if not key:
            raise SystemExit(f"Set {cfg['key_env']} (in the environment or a .env file).")
    headers = {}
    if endpoint == "wandb" and os.environ.get("WANDB_PROJECT"):
        headers["OpenAI-Project"] = os.environ["WANDB_PROJECT"]   # "team/project", optional
    return OpenAI(base_url=base_url or cfg["base_url"], api_key=key,
                  default_headers=headers or None, timeout=900, max_retries=0)


def server_info(client, endpoint: str) -> dict:
    """What the server says about itself, saved with the run (vLLM version, context length)."""
    info: dict = {}
    try:
        info["models"] = [m.model_dump() for m in client.models.list().data][:5] if endpoint == "local" else None
    except Exception as e:
        info["models_error"] = repr(e)
    if endpoint == "local":
        import urllib.request
        root = str(client.base_url).rstrip("/").removesuffix("/v1")
        try:
            with urllib.request.urlopen(f"{root}/version", timeout=10) as r:
                info["vllm_version"] = json.loads(r.read()).get("version")
        except Exception as e:
            info["version_error"] = repr(e)
    return info


def _reasoning_text(message) -> str | None:
    for attr in ("reasoning_content", "reasoning"):
        value = getattr(message, attr, None) or (getattr(message, "model_extra", None) or {}).get(attr)
        if value:
            return value
    return None


def _retryable(e: Exception) -> bool:
    import openai
    return isinstance(e, (openai.RateLimitError, openai.APIConnectionError,
                          openai.APITimeoutError, openai.InternalServerError))


def generate_one(client, endpoint: str, model: str, fmt: str, prompt: Prompt,
                 dec: Decoding, gold_sql: str) -> dict:
    """All samples for one question. Errors are recorded, never silently re-sampled."""
    samples, usage = [], {"prompt_tokens": 0, "completion_tokens": 0}
    for k in range(dec.n):
        if endpoint == "gold":
            samples.append({"text": f"```sql\n{gold_sql}\n```", "finish_reason": "stop"})
            continue
        seed = {"seed": k} if endpoint == "local" else {}
        for attempt in range(6):
            try:
                t0 = time.monotonic()
                if fmt == "chat":
                    msgs = ([{"role": "system", "content": prompt.system}] if prompt.system else [])
                    msgs.append({"role": "user", "content": prompt.user})
                    r = client.chat.completions.create(
                        model=model, messages=msgs, temperature=dec.temperature, top_p=dec.top_p,
                        max_tokens=dec.max_tokens, extra_body=dec.extra_body(endpoint), **seed)
                    choice = r.choices[0]
                    sample = {"text": choice.message.content or "",
                              "reasoning": _reasoning_text(choice.message)}
                else:
                    r = client.completions.create(
                        model=model, prompt=prompt.completion_text(), temperature=dec.temperature,
                        top_p=dec.top_p, max_tokens=dec.max_tokens, stop=["```"],
                        extra_body=dec.extra_body(endpoint), **seed)
                    choice = r.choices[0]
                    sample = {"text": choice.text, "prefilled": True}
                sample["finish_reason"] = choice.finish_reason
                sample["seconds"] = round(time.monotonic() - t0, 2)
                if r.usage:
                    usage["prompt_tokens"] += r.usage.prompt_tokens or 0
                    usage["completion_tokens"] += r.usage.completion_tokens or 0
                samples.append(sample)
                break
            except Exception as e:
                if _retryable(e) and attempt < 5:
                    time.sleep(min(60, 2 ** attempt) + random.random())
                    continue
                samples.append({"text": None, "error": f"{type(e).__name__}: {str(e)[:400]}"})
                break
    return {"samples": samples, "usage": usage}


def run_generation(examples: list[Example], prompts: dict[int, Prompt], client, endpoint: str,
                   model: str, fmt: str, dec: Decoding, out_path: Path, workers: int) -> None:
    """Append one JSON line per question to out_path. Re-running resumes where it stopped,
    and asks again any question whose request failed (the new line replaces the old one)."""
    done, failed_before = set(), 0
    if out_path.exists():
        latest = {}
        for line in out_path.open():
            if line.strip():
                rec = json.loads(line)
                latest[rec["idx"]] = rec
        done = {i for i, rec in latest.items() if not any(s.get("error") for s in rec["samples"])}
        failed_before = len(latest) - len(done)
    todo = [e for e in examples if e.idx not in done]
    if not todo:
        print(f"generation: all {len(examples)} questions already done")
        return
    print(f"generation: {len(todo)} to do ({len(done)} already done, "
          f"{failed_before} failed last time and will be asked again), {workers} workers")

    lock, finished, failures, t0 = threading.Lock(), 0, 0, time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool, out_path.open("a") as f:
        futures = {pool.submit(generate_one, client, endpoint, model, fmt, prompts[e.idx], dec, e.gold_sql): e
                   for e in todo}
        for fut in as_completed(futures):
            e = futures[fut]
            rec = {"idx": e.idx, "question_id": e.question_id, **fut.result()}
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                finished += 1
                failures += all(s.get("error") for s in rec["samples"])
                if finished == 10 and failures == 10:
                    pool.shutdown(cancel_futures=True)
                    raise SystemExit("The first 10 questions all failed; check the model id and endpoint:\n"
                                     + rec["samples"][0]["error"])
                if finished % 50 == 0 or finished == len(todo):
                    rate = finished / (time.monotonic() - t0)
                    print(f"  {finished}/{len(todo)}  ({rate:.1f} q/s, {failures} failed)", flush=True)


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", text.split("/")[-1]).strip("-").lower()


def decoding_dict(dec: Decoding) -> dict:
    return asdict(dec)
