#!/bin/bash
#SBATCH --job-name=relbench-small
#SBATCH --partition=epyc-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
mkdir -p logs

module load anaconda
eval "$(conda shell.bash hook)"
conda activate relbench
echo "Loaded conda"

export RELBENCH_CACHE_DIR=/hpc/gpfs2/scratch/u/thomasti/relbench_cache
export HF_HUB_OFFLINE=1

echo "[Job ${SLURM_JOB_ID}] Args: $@"

python gnn_recommendation.py \
    --cache_dir $RELBENCH_CACHE_DIR\
    --no-download\
    "$@"