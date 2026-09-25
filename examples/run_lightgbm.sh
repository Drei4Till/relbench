#!/bin/bash
#SBATCH --job-name=relb-lgbm
#SBATCH --partition=epyc
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%A_%a.out

# Usage: sbatch run_lightgbm.sh <entity|recommendation|autocomplete> --dataset rel-f1 --task driver-dnf [--seed 42 ...]

set -euo pipefail
mkdir -p logs

KIND="${1:?first argument must be 'entity', 'recommendation' or 'autocomplete'}"
shift
case "$KIND" in
    entity|recommendation|autocomplete) ;;
    *) echo "unknown kind: $KIND (expected entity|recommendation|autocomplete)" >&2; exit 1 ;;
esac

module load anaconda
eval "$(conda shell.bash hook)"
conda activate relbench
echo "Loaded conda"

export HF_HOME=/hpc/gpfs2/scratch/u/thomasti/hf_cache
export HF_HUB_OFFLINE=1
export SENTENCE_TRANSFORMERS_HOME=/hpc/gpfs2/scratch/u/thomasti/hf_cache
export TRANSFORMERS_OFFLINE=1

echo "[Job ${SLURM_JOB_ID}] lightgbm_${KIND}.py Args: $@"

python "lightgbm_${KIND}.py" \
    --cache_dir /hpc/gpfs2/scratch/u/thomasti/relbench_examples_cache \
    --pred_dir "$HOME/relbench_preds/lightgbm_${KIND}" \
    "$@"
