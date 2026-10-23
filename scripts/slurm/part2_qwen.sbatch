#!/bin/bash
# Part 2: Qwen2.5-Coder-7B Base and Instruct on BIRD dev, no training, greedy decoding,
# without (p0-schema) and with (p1-evidence) BIRD's evidence hint. Four runs; under an hour on one H200.
# Runs that already finished are skipped (only re-scored).   sbatch scripts/slurm/part2_qwen.sbatch
# Setup steps are at the top of check.sbatch. Resubmitting resumes where a run stopped.

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --job-name=glassbox-part2
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/scratch/%u/glassbox_part2_%j.out
#SBATCH --error=/scratch/%u/glassbox_part2_%j.err

source /scratch/$USER/glassbox_env/bin/activate
set -eo pipefail
cd /scratch/$USER/glassbox-sql

export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1
export BIRD_DIR=/scratch/$USER/bird_dev/dev_20240627
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

INSTRUCT=Qwen/Qwen2.5-Coder-7B-Instruct    # chat model: gets the prompt as a chat message
BASE=Qwen/Qwen2.5-Coder-7B                 # base model: gets plain text with the SQL fence opened

for SPLIT in dev; do
  for PROMPT in p0-schema p1-evidence; do
    python -m glassbox run --endpoint hf --model $INSTRUCT --format chat       --prompt $PROMPT --split $SPLIT
    python -m glassbox run --endpoint hf --model $BASE     --format completion --prompt $PROMPT --split $SPLIT
  done
done

echo; echo "================ SUMMARY ================"
R=qwen2.5-coder-7b
for SPLIT in dev; do
  echo; echo "---- $SPLIT ----"
  python -m glassbox report runs/$SPLIT/$R-instruct__p0-schema__chat__t0 runs/$SPLIT/$R-instruct__p1-evidence__chat__t0 \
                            runs/$SPLIT/${R}__p0-schema__completion__t0  runs/$SPLIT/${R}__p1-evidence__completion__t0
  echo "Instruct vs Base, with evidence:"
  python -m glassbox compare runs/$SPLIT/$R-instruct__p1-evidence__chat__t0 runs/$SPLIT/${R}__p1-evidence__completion__t0
  echo "Instruct, without vs with evidence:"
  python -m glassbox compare runs/$SPLIT/$R-instruct__p0-schema__chat__t0 runs/$SPLIT/$R-instruct__p1-evidence__chat__t0
  echo "Base, without vs with evidence:"
  python -m glassbox compare runs/$SPLIT/${R}__p0-schema__completion__t0 runs/$SPLIT/${R}__p1-evidence__completion__t0
done
