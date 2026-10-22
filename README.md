# Glass Box: Post-Training a Small Model in Public

A from-scratch study of how much each post-training technique really improves a small open model at
turning English questions into SQL, measured on the [BIRD](https://bird-bench.github.io/) benchmark.

Every number in the series comes from a run folder you can open: the exact prompt, the model's full
answer, the SQL it wrote, whether that SQL returned the right rows, and why not when it failed. The
scorer is checked against BIRD's own evaluation script on every published run.

The series is written up at [jenishkothari.pages.dev](https://jenishkothari.pages.dev/). Each part has
one experiment, its results, and the maths behind it.

## The series

| Part | Question | The maths | Status |
|---|---|---|---|
| 1 | Starting over: how is BIRD scored, and how can a score be wrong? | Accuracy as a proportion; 95% confidence intervals (why about ±2.5 points on 1,534 questions) | writing |
| 2 | How good are Qwen2.5-Coder-7B Base and Instruct with no training, and how do they fail? | How a model picks each token (softmax, greedy decoding); paired McNemar test | next |
| 3 | What is each prompt ingredient worth? | Testing each change for significance; repetition penalty as an operation on logits | planned |
| 4 | How good are frontier models on the same prompt? | Tokens to dollars; cost per correct answer | planned |
| 5 | Supervised fine-tuning (SFT) | Cross-entropy loss; LoRA's low-rank update ΔW = (α/r)·BA | planned |
| 6 | Distilling reasoning from a larger model | Rejection sampling; distillation as minimising KL divergence | planned |
| 7 | More samples, better answers: voting and pass@k | Majority-vote probability; pass@k = 1 − C(n−c,k)/C(n,k) | planned |
| 8 | Direct Preference Optimization (DPO) | Bradley-Terry model; deriving the DPO loss | planned |
| 9 | Reinforcement learning with GRPO | Policy gradient; group-relative advantage; why binary rewards saturate | planned |
| 10 | The final exam: BIRD hidden test set | What carries over from dev to test | planned |

Rules every part follows:

1. **Full split or it does not count.** A run that did not answer all 1,534 questions is labelled INCOMPLETE.
2. **Every decoding setting is explicit.** Nothing is inherited from a model's own config file.
3. **Confidence intervals on every number,** and a paired test whenever two runs are compared.
4. **Two dev splits:** the original BIRD dev (comparable with papers) and the cleaned dev-1106 re-release.
5. **Choices are made before seeing dev results,** and written down in the post.
6. **Every number in a post comes from a run folder** that is published with it.

## How it works

```
question + database ──► prompt ──► model ──► answer text ──► SQL ──► run on the database ──► compare with the answer key
                       (prompts.py)  (generate.py)     (score.py)       (sqlexec.py)              (sqlexec.py, score.py)
```

Each answer gets exactly one label:

| Label | Meaning |
|---|---|
| `correct` | The rows match the answer key (BIRD's official rule, below) |
| `wrong_result` | The SQL ran but returned the wrong rows: wrong `columns`, wrong `row_count`, or wrong `values` |
| `sql_error` | SQLite rejected the SQL, with the error type (`no such column`, `syntax error`, ...) |
| `timeout` | The SQL ran longer than 30 seconds |
| `no_sql` / `truncated` | No SQL in the answer (truncated: the answer hit the token limit) |
| `api_error` | The model call failed (resuming the run asks again) |
| `gold_failed` | The answer-key SQL itself failed; flagged so it can be checked |

## Quick start (any computer, no GPU)

You need [uv](https://docs.astral.sh/uv/) (it installs the exact library versions in `uv.lock`).

```bash
git clone https://github.com/jenishk20/glassbox-sql
cd glassbox-sql
uv sync --extra dev --extra verify
```

Get BIRD dev (350 MB download, checked against the exact release used here):

```bash
bash scripts/get_bird_dev.sh
uv run python -m glassbox fetch-dev1106
```

Check that everything works, without any model. This run "answers" every question with the answer key:

```bash
uv run python -m glassbox run --endpoint gold --model gold
uv run pytest
```

Expected: **1,532 of 1,534 correct**. The two misses are dev questions 518 and 701, whose answer-key SQL
takes longer than the 30-second limit on a laptop (about 34 s and 177 s). Under BIRD's rules a correct
answer must also be fast enough, so the slow answer counts as wrong. A model can still get these two
right by writing faster SQL.

Then confirm our scorer agrees with BIRD's own script on the same run:

```bash
uv run python -m glassbox verify-official runs/dev/gold__p1-evidence__chat__t0
```

## Evaluating a model

### Through an API (W&B Inference)

Put `WANDB_API_KEY=...` in a `.env` file, then:

```bash
uv run python -m glassbox models --endpoint wandb                    # model ids on offer
uv run python -m glassbox run --endpoint wandb --model <model-id> --limit 5   # 5-question test first
uv run python -m glassbox run --endpoint wandb --model <model-id>             # the full split
```

### On your own GPU (any machine with vLLM)

Start a server, then point glassbox at it. `--generation-config vllm` stops vLLM from silently applying
the model's own default sampling settings.

```bash
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --dtype bfloat16 --max-model-len 16384 --generation-config vllm --seed 0
uv run python -m glassbox run --endpoint local --model Qwen/Qwen2.5-Coder-7B-Instruct --format chat
```

Base models are not chat models, so they get the prompt as plain text with the SQL fence already opened:
add `--format completion`.

### On a SLURM cluster (how Part 2 was run)

We never `pip install` vLLM on the cluster. Installing it into a shared Python environment is where
version conflicts come from. Instead, everything runs inside vLLM's official container image (a single
file that already has vLLM, PyTorch, CUDA libraries and Python), through Apptainer. glassbox only needs
Python and `openai`, which the image already has.

**One-time setup, on a login node** (if your cluster needs a proxy for internet access, set
`https_proxy` and `http_proxy` first; if `apptainer` is not found, try `module load apptainer`):

```bash
WORK=/scratch/$USER/glassbox && mkdir -p $WORK && cd $WORK
git clone https://github.com/jenishk20/glassbox-sql
export APPTAINER_CACHEDIR=$WORK/apptainer-cache APPTAINER_TMPDIR=$WORK/apptainer-tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

# 1. The container (pinned version; the -cu129 build works with older GPU drivers than the default)
apptainer pull $WORK/vllm-openai-v0.31.0-cu129.sif docker://vllm/vllm-openai:v0.31.0-x86_64-cu129

# 2. Model weights (GPU nodes may have no internet, so download them here)
for m in Qwen/Qwen2.5-Coder-7B-Instruct Qwen/Qwen2.5-Coder-7B; do
  apptainer exec --cleanenv --env HF_HOME=$WORK/hf,https_proxy=${https_proxy:-} \
    $WORK/vllm-openai-v0.31.0-cu129.sif \
    python3 -c "from huggingface_hub import snapshot_download as d; print(d('$m'))"
done

# 3. BIRD dev and dev-1106
cd glassbox-sql
bash scripts/get_bird_dev.sh
apptainer exec --cleanenv --env PYTHONPATH=src,https_proxy=${https_proxy:-} \
  $WORK/vllm-openai-v0.31.0-cu129.sif python3 -m glassbox fetch-dev1106
```

**Check, then run** (from the repo folder; edit the paths at the top of each job file if yours differ):

```bash
mkdir -p logs
sbatch scripts/slurm/check.sbatch        # ~10 min; must end with "ALL CHECKS PASSED"
sbatch scripts/slurm/part2_qwen.sbatch   # Part 2: Base and Instruct on dev and dev-1106
```

If the check fails at step 2 with a CUDA error, the cluster's GPU driver is older than the container's
CUDA version (`nvidia-smi` in the log shows the driver). Pull an older vLLM release and try again.

Copy the results to your own computer to read them:

```bash
rsync -av <cluster>:/scratch/<user>/glassbox/glassbox-sql/runs/ runs/
```

## Reading results

```bash
uv run python -m glassbox report runs/dev/* --detail          # table + failure breakdowns
uv run python -m glassbox compare runs/dev/A runs/dev/B         # paired: which questions only A or B got right
uv run python -m glassbox traces runs/dev/A --category wrong_result --n 25   # readable mistakes, as markdown
uv run python -m glassbox show-prompt --prompt p1-evidence --idx 0           # the exact prompt for one question
```

Every run folder (`runs/<split>/<run name>/`) contains:

| File | What is in it |
|---|---|
| `meta.json` | Model, prompt, every decoding setting, dataset checksum, code version, vLLM version |
| `prompts.jsonl` | The exact text sent to the model, per question |
| `generations.jsonl` | The model's full answers (and reasoning, if the API returns it), token counts |
| `scores.jsonl` | Per question: extracted SQL, label, error, first rows of both results |
| `summary.json` | Accuracy with 95% confidence interval, by difficulty, failure counts |
| `predict_dev.json` | The predictions in BIRD's own submission format |
| `official_check.json` | Our accuracy next to BIRD's script's accuracy (after `verify-official`) |

## How scoring works

BIRD's official rule ([evaluation.py](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/evaluation.py)):
an answer is correct when `set(predicted_rows) == set(answer_key_rows)`. So row order and duplicate rows
do not matter, column order does, `1` equals `1.0`, and `'1'` does not equal `1`. These rules are pinned
by tests in `tests/test_scoring.py`.

Three deliberate differences, all small:

1. Databases are opened read-only, so a generated `DROP TABLE` cannot damage them.
2. A model's SQL gets 30 seconds (the official limit). The answer-key SQL gets 5 minutes, because it
   must always produce the answer. The official script gives both queries one shared 30-second budget,
   which on a laptop marks questions 518 and 701 wrong even when the model's SQL is right.
3. Results larger than one million rows are cut off and counted wrong (no answer key is anywhere near that).

`verify-official` runs BIRD's script, unchanged, on a run and prints both numbers:

| Run | glassbox | BIRD's script | Difference |
|---|---|---|---|
| Answer key as the "model" | 99.87% | 99.87% | 0 questions |
| GLM 5.2 answers saved from an earlier study | 63.17% | 63.04% | 2 questions: 518 and 701, answered correctly with faster SQL |

### Known quirks in the benchmark itself

Found by running the answer key as if it were a model (`--endpoint gold`), which should score 100%:

| Split | Question | What happens |
|---|---|---|
| dev and dev-1106 | 518, 701 | The answer-key SQL takes longer than 30 seconds on a laptop (about 34 s and 177 s), so the same SQL from a model counts as a timeout. Faster correct SQL still scores. |
| dev-1106 | 1131 | Same: slow answer-key SQL. |
| dev-1106 | 178 | The answer-key SQL computes a card's age with `julianday('now')`, so the correct answer changes every second. No model can match it. |

So the highest possible score is 1,532 / 1,534 on dev and 1,530 / 1,534 on dev-1106 for SQL that is as
slow as the answer key.

## Prompts

`p1-evidence` is the standard BIRD setting and the one used for Parts 2 and 4. Part 3 walks the whole
ladder, where each step adds exactly one ingredient, all taken from BIRD's own files:

| Prompt | Adds |
|---|---|
| `p0-schema` | Tables and columns (built from the database) and the question |
| `p1-evidence` | BIRD's evidence hint |
| `p2-descriptions` | A description comment on every column (from BIRD's `database_description` files) |
| `p3-examples` | Two example values per column |
| `p4-matches` | Database values that appear word for word in the question (simple matching; it also catches some coincidences) |
| `p5-reasoning` | "Think step by step" before the final SQL |
| `v1` | The prompt from the earlier v1 study, to explain its numbers |

No system message is sent with the ladder prompts, so a chat model's template may add its own default
(Qwen's adds "You are Qwen, created by Alibaba Cloud..."). `prompts.jsonl` records what we sent.

## Licenses

The code is MIT licensed. BIRD is distributed by its authors under CC BY-SA 4.0 and is not included
here; published run folders contain BIRD text and are shared under CC BY-SA 4.0 as well.

## Citation

```bibtex
@inproceedings{li2023bird,
  title     = {Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs},
  author    = {Li, Jinyang and Hui, Binyuan and Qu, Ge and Yang, Jiaxi and Li, Binhua and Li, Bowen and Wang, Bailin and Qin, Bowen and Geng, Ruiying and Huo, Nan and others},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023}
}
```
