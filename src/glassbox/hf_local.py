"""Run an open model on the local GPU with Hugging Face transformers (no server, no vLLM).

Needs only torch and transformers (tested with torch 2.4.1 + transformers 4.46.3 on Python 3.8).
Every generation setting is set explicitly in a fresh GenerationConfig; only the model's stop
tokens are taken from its own config. v1 silently picked up repetition_penalty=1.1 from there.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .data import Example
from .generate import Decoding, already_done
from .prompts import Prompt


def load(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"                      # decoder-only models pad on the left for batching
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    gpu = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16 if gpu else torch.float32, attn_implementation="sdpa")
    model.to("cuda" if gpu else "cpu").eval()
    return tok, model


def generation_config(model, tok, dec: Decoding):
    from transformers import GenerationConfig

    settings = dict(max_new_tokens=dec.max_tokens, num_return_sequences=dec.n,
                    repetition_penalty=dec.repetition_penalty or 1.0,
                    eos_token_id=model.generation_config.eos_token_id, pad_token_id=tok.pad_token_id)
    if dec.temperature > 0:
        settings.update(do_sample=True, temperature=dec.temperature, top_p=dec.top_p, top_k=0)
    else:
        settings.update(do_sample=False)            # greedy: temperature/top_p/top_k are unused
    return GenerationConfig(**settings)


def prompt_text(tok, prompt: Prompt, fmt: str) -> str:
    if fmt == "completion":
        return prompt.completion_text()
    messages = ([{"role": "system", "content": prompt.system}] if prompt.system else [])
    messages.append({"role": "user", "content": prompt.user})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _generate_batch(tok, model, cfg, texts: list[str], fmt: str) -> list[list[dict]]:
    """Samples for each text. On out-of-memory, split the batch in half and try again."""
    import torch

    try:
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        extra = {"stop_strings": ["```"], "tokenizer": tok} if fmt == "completion" else {}
        t0 = time.monotonic()
        with torch.no_grad():
            out = model.generate(**enc, generation_config=cfg, **extra)
        seconds = round((time.monotonic() - t0) / len(texts), 2)
    except torch.cuda.OutOfMemoryError:
        if len(texts) == 1:
            raise
        torch.cuda.empty_cache()
        half = len(texts) // 2
        return (_generate_batch(tok, model, cfg, texts[:half], fmt)
                + _generate_batch(tok, model, cfg, texts[half:], fmt))

    stop_ids = set(cfg.eos_token_id if isinstance(cfg.eos_token_id, list) else [cfg.eos_token_id])
    stop_ids.add(tok.pad_token_id)
    prompt_len = enc["input_ids"].shape[1]
    prompt_tokens = enc["attention_mask"].sum(dim=1).tolist()
    results = []
    for i in range(len(texts)):
        samples = []
        for k in range(cfg.num_return_sequences):
            new = out[i * cfg.num_return_sequences + k, prompt_len:].tolist()
            n_new = next((j for j, t in enumerate(new) if t in stop_ids), len(new))
            text = tok.decode(new[:n_new], skip_special_tokens=True)
            sample = {"text": text, "seconds": seconds,
                      "finish_reason": "length" if n_new >= cfg.max_new_tokens else "stop",
                      "tokens": n_new, "prompt_tokens": int(prompt_tokens[i])}
            if fmt == "completion":                  # the model wrote SQL into the fence we opened
                sample["text"] = text.split("```")[0]
                sample["prefilled"] = True
            samples.append(sample)
        results.append(samples)
    return results


def run_generation_hf(examples: list[Example], prompts: dict[int, Prompt], model_id: str, fmt: str,
                      dec: Decoding, out_path: Path, batch_size: int) -> dict:
    """Same output format and resume rule as generate.run_generation. Returns engine info for meta.json."""
    done, failed_before = already_done(out_path)
    todo = [e for e in examples if e.idx not in done]
    if not todo:
        print(f"generation: all {len(examples)} questions already done")
        return {}

    import torch
    import transformers

    print(f"loading {model_id} with transformers {transformers.__version__} ...", flush=True)
    tok, model = load(model_id)
    cfg = generation_config(model, tok, dec)
    texts = {e.idx: prompt_text(tok, prompts[e.idx], fmt) for e in todo}
    # Longest prompts first: similar lengths batch together, and memory problems show up early.
    todo.sort(key=lambda e: len(texts[e.idx]), reverse=True)
    print(f"generation: {len(todo)} to do ({len(done)} already done, {failed_before} failed last time), "
          f"batch size {batch_size}", flush=True)

    t0, finished = time.monotonic(), 0
    with out_path.open("a") as f:
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            for e, samples in zip(batch, _generate_batch(tok, model, cfg, [texts[e.idx] for e in batch], fmt)):
                usage = {"prompt_tokens": samples[0]["prompt_tokens"],
                         "completion_tokens": sum(s["tokens"] for s in samples)}
                f.write(json.dumps({"idx": e.idx, "question_id": e.question_id,
                                    "samples": samples, "usage": usage}, ensure_ascii=False) + "\n")
            f.flush()
            finished += len(batch)
            if finished % 100 < batch_size or finished == len(todo):
                rate = finished / (time.monotonic() - t0)
                print(f"  {finished}/{len(todo)}  ({rate:.2f} q/s)", flush=True)

    return {"engine": "transformers", "transformers": transformers.__version__, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "batch_size": batch_size, "generation_config": cfg.to_diff_dict()}
