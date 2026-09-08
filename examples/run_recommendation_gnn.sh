#!/bin/bash
#SBATCH --job-name=relb-rec-gnn
#SBATCH --partition=epyc-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
mkdir -p logs

module load anaconda
eval "$(conda shell.bash hook)"
conda activate relbench
echo "Loaded conda"

export HF_HOME=/hpc/gpfs2/scratch/u/thomasti/hf_cache
export HF_HUB_OFFLINE=1
export SENTENCE_TRANSFORMERS_HOME=/hpc/gpfs2/scratch/u/thomasti/hf_cache
export TRANSFORMERS_OFFLINE=1

echo "[Job ${SLURM_JOB_ID}] Args: $@"

python gnn_recommendation.py \
    --cache_dir /hpc/gpfs2/scratch/u/thomasti/relbench_examples_cache\
    "$@"