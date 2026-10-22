#!/bin/bash
# Part 2: Qwen2.5-Coder-7B Base and Instruct on BIRD dev and dev-1106, no training, greedy decoding.
# Four runs (1,534 questions each); about 1-2 hours on one H200.   sbatch scripts/slurm/part2_qwen.sbatch
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

for SPLIT in dev dev1106; do
  python -m glassbox run --endpoint hf --model $INSTRUCT --format chat       --split $SPLIT
  python -m glassbox run --endpoint hf --model $BASE     --format completion --split $SPLIT
done

echo; echo "================ SUMMARY ================"
python -m glassbox report runs/dev/qwen2.5-coder-7b* runs/dev1106/qwen2.5-coder-7b* --detail
for SPLIT in dev dev1106; do
  echo; echo "---- Instruct (A) vs Base (B) on $SPLIT ----"
  python -m glassbox compare runs/$SPLIT/qwen2.5-coder-7b-instruct__p1-evidence__chat__t0 \
                             runs/$SPLIT/qwen2.5-coder-7b__p1-evidence__completion__t0
done
