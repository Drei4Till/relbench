#!/bin/bash
#SBATCH --job-name=relb-trivial
#SBATCH --partition=epyc
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%A_%a.out

# Usage: sbatch run_trivial.sh <entity|recommendation|autocomplete> --dataset rel-f1 --task driver-dnf [...]

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
export HF_HUB_OFFLINE=0

echo "[Job ${SLURM_JOB_ID}] trivial_${KIND}.py Args: $@"

# The trivial scripts use no graph/feature cache, so no --cache_dir.
python "trivial_${KIND}.py" \
    --pred_dir "$HOME/relbench_preds/trivial_${KIND}" \
    "$@"
