#!/bin/bash
# Quick check (about 10 minutes): both Qwen models answer 20 BIRD questions each.
# Run once after setup, before part2_qwen.sbatch:   sbatch scripts/slurm/check.sbatch
#
# ONE-TIME SETUP (in an interactive shell on the cluster; the proxy is only needed for downloads):
#   python3 -m venv /scratch/$USER/glassbox_env
#   source /scratch/$USER/glassbox_env/bin/activate
#   export https_proxy=http://10.99.0.130:3128 http_proxy=http://10.99.0.130:3128
#   pip install --upgrade pip && pip install torch==2.4.1 transformers==4.46.3
#   export HF_HOME=/scratch/$USER/hf_cache
#   python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-Coder-7B-Instruct'); d('Qwen/Qwen2.5-Coder-7B')"
#   cd /scratch/$USER/glassbox-sql && bash scripts/get_bird_dev.sh /scratch/$USER/bird_dev
#   curl -L -o /scratch/$USER/bird_dev/dev_20240627/dev_20251106.json https://huggingface.co/datasets/birdsql/bird_sql_dev_20251106/resolve/main/data/dev_20251106-00000-of-00001.json

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:45:00
#SBATCH --job-name=glassbox-check
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/scratch/%u/glassbox_check_%j.out
#SBATCH --error=/scratch/%u/glassbox_check_%j.err

source /scratch/$USER/glassbox_env/bin/activate
set -eo pipefail
cd /scratch/$USER/glassbox-sql

export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1                              # weights were downloaded in setup
export BIRD_DIR=/scratch/$USER/bird_dev/dev_20240627
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch, transformers, sqlite3; print('torch', torch.__version__, '| GPU ok:', torch.cuda.is_available(), '| transformers', transformers.__version__, '| SQLite', sqlite3.sqlite_version)"

python -m glassbox run --endpoint hf --model Qwen/Qwen2.5-Coder-7B-Instruct --format chat --limit 20
python -m glassbox run --endpoint hf --model Qwen/Qwen2.5-Coder-7B --format completion --limit 20

echo "ALL CHECKS PASSED"
