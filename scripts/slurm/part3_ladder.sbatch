#!/bin/bash
# Part 3: what is each prompt ingredient worth? Qwen2.5-Coder-7B Base and Instruct, greedy, on
# BIRD dev. p0-schema and p1-evidence were run in Part 2. Also reproduces v1's prompt, with and
# without its repetition penalty of 1.1. 9 runs; about 2-3 hours on one H200.
# Resubmitting resumes where it stopped.   sbatch scripts/slurm/part3_ladder.sbatch

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --job-name=glassbox-part3
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/scratch/%u/glassbox_part3_%j.out
#SBATCH --error=/scratch/%u/glassbox_part3_%j.err

source /scratch/$USER/glassbox_env/bin/activate
set -eo pipefail
cd /scratch/$USER/glassbox-sql

export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1
export BIRD_DIR=/scratch/$USER/bird_dev/dev_20240627
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

INSTRUCT=Qwen/Qwen2.5-Coder-7B-Instruct
BASE=Qwen/Qwen2.5-Coder-7B

for SPLIT in dev; do
  for PROMPT in p2-descriptions p3-examples p4-matches p5-reasoning v1; do
    python -m glassbox run --endpoint hf --model $INSTRUCT --format chat --prompt $PROMPT --split $SPLIT
  done
  # v1's exact settings minus 4-bit weights: its prompt plus repetition_penalty 1.1
  python -m glassbox run --endpoint hf --model $INSTRUCT --format chat --prompt v1 --repetition-penalty 1.1 --split $SPLIT

  # base models get no "think step by step" (p5), only the plain-text prompts
  for PROMPT in p2-descriptions p3-examples p4-matches; do
    python -m glassbox run --endpoint hf --model $BASE --format completion --prompt $PROMPT --split $SPLIT
  done
done

echo; echo "================ SUMMARY (includes Part 2 runs) ================"
for SPLIT in dev; do
  echo; echo "---- $SPLIT ----"
  python -m glassbox report $(ls -d runs/$SPLIT/qwen2.5-coder-7b*__t0* | grep -v limit)
done
