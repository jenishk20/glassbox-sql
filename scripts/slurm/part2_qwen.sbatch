#!/bin/bash
# Part 2: Qwen2.5-Coder-7B Base and Instruct on BIRD dev and dev-1106, no training, greedy decoding.
# Four runs in total; expect well under an hour on one H200.   sbatch scripts/slurm/part2_qwen.sbatch
#SBATCH --job-name=glassbox-part2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail

# ---- edit these to match your cluster (same values as in check.sbatch) ----------------------
WORK=/scratch/$USER/glassbox
REPO=$WORK/glassbox-sql
SIF=$WORK/vllm-openai-v0.31.0-cu129.sif
BIRD_DIR=$REPO/data/bird/dev_20240627
# --------------------------------------------------------------------------------------------

# model id | how we talk to it | prompt   (base models get plain text, not a chat)
RUNS=(
  "Qwen/Qwen2.5-Coder-7B-Instruct chat       p1-evidence"
  "Qwen/Qwen2.5-Coder-7B          completion p1-evidence"
)
SPLITS=(dev dev1106)

PORT=$((8000 + SLURM_JOB_ID % 1000))
cd "$REPO"
mkdir -p logs "$WORK/cache"
AX=(--nv --cleanenv --bind /scratch
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0},HF_HOME=$WORK/hf,HF_HUB_OFFLINE=1,XDG_CACHE_HOME=$WORK/cache,PYTHONPATH=$REPO/src,BIRD_DIR=$BIRD_DIR,RUNS_DIR=$REPO/runs")
glassbox() { apptainer exec "${AX[@]}" "$SIF" python3 -m glassbox "$@"; }

SERVER=""
start_server() {
  local model=$1 log="logs/vllm_${SLURM_JOB_ID}_$(basename "$1").log"
  apptainer exec "${AX[@]}" "$SIF" vllm serve "$model" --port "$PORT" --dtype bfloat16 \
      --max-model-len 16384 --generation-config vllm --seed 0 > "$log" 2>&1 &
  SERVER=$!
  for i in $(seq 1 90); do
    curl -sf "http://localhost:$PORT/health" > /dev/null && { echo "vLLM ready: $model"; return 0; }
    kill -0 "$SERVER" 2>/dev/null || { echo "vLLM exited, last log lines:"; tail -40 "$log"; exit 1; }
    sleep 10
  done
  echo "vLLM not ready after 15 minutes"; exit 1
}
stop_server() { [ -n "$SERVER" ] && { kill "$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; SERVER=""; }; }
trap stop_server EXIT

RUN_DIRS=()
for spec in "${RUNS[@]}"; do
  read -r MODEL FORMAT PROMPT <<< "$spec"
  start_server "$MODEL"
  for SPLIT in "${SPLITS[@]}"; do
    glassbox run --endpoint local --base-url "http://localhost:$PORT/v1" \
        --model "$MODEL" --format "$FORMAT" --prompt "$PROMPT" --split "$SPLIT"
    name=$(echo "$MODEL" | awk -F/ '{print tolower($NF)}')
    RUN_DIRS+=("runs/$SPLIT/${name}__${PROMPT}__${FORMAT}__t0")
  done
  stop_server
done

echo; echo "================ SUMMARY ================"
glassbox report "${RUN_DIRS[@]}"
for SPLIT in "${SPLITS[@]}"; do
  echo; echo "---- Instruct vs Base on $SPLIT ----"
  glassbox compare "runs/$SPLIT/qwen2.5-coder-7b-instruct__p1-evidence__chat__t0" \
                   "runs/$SPLIT/qwen2.5-coder-7b__p1-evidence__completion__t0"
done
