#!/bin/bash
# Quick check (about 10 minutes): GPU visible, vLLM serves Qwen, glassbox answers 5 questions.
# Run this once after setup, before the real jobs:   sbatch scripts/slurm/check.sbatch
#SBATCH --job-name=glassbox-check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail

# ---- edit these to match your cluster (same values as in part2_qwen.sbatch) ----------------
WORK=/scratch/$USER/glassbox                      # container, model weights, caches
REPO=$WORK/glassbox-sql                           # this repo, cloned on the cluster
SIF=$WORK/vllm-openai-v0.31.0-cu129.sif           # from `apptainer pull` (see README)
BIRD_DIR=$REPO/data/bird/dev_20240627             # from scripts/get_bird_dev.sh
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
# --------------------------------------------------------------------------------------------

PORT=$((8000 + SLURM_JOB_ID % 1000))
cd "$REPO"
mkdir -p logs "$WORK/cache"
# --cleanenv: nothing from your login shell (conda, PYTHONPATH...) leaks into the container.
AX=(--nv --cleanenv --bind /scratch
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0},HF_HOME=$WORK/hf,HF_HUB_OFFLINE=1,XDG_CACHE_HOME=$WORK/cache,PYTHONPATH=$REPO/src,BIRD_DIR=$BIRD_DIR,RUNS_DIR=$REPO/runs")

echo "== 1. GPU and driver"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "== 2. Inside the container"
apptainer exec "${AX[@]}" "$SIF" python3 -c "
import torch, vllm, openai
print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))
print('vllm', vllm.__version__, '| openai', openai.__version__)"

echo "== 3. Start vLLM with $MODEL on port $PORT"
apptainer exec "${AX[@]}" "$SIF" vllm serve "$MODEL" --port "$PORT" --dtype bfloat16 \
    --max-model-len 16384 --generation-config vllm --seed 0 > "logs/vllm_check_${SLURM_JOB_ID}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 90); do
  curl -sf "http://localhost:$PORT/health" > /dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "vLLM exited, last log lines:"; tail -40 "logs/vllm_check_${SLURM_JOB_ID}.log"; exit 1; }
  sleep 10
done
curl -sf "http://localhost:$PORT/health" > /dev/null || { echo "vLLM not ready after 15 min"; exit 1; }
echo "server ready"

echo "== 4. glassbox on 5 questions"
apptainer exec "${AX[@]}" "$SIF" python3 -m glassbox run --endpoint local --base-url "http://localhost:$PORT/v1" \
    --model "$MODEL" --format chat --prompt p1-evidence --limit 5

echo "== ALL CHECKS PASSED"
