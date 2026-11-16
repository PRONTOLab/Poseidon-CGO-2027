#!/bin/bash
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -J poseidon-artifact
#SBATCH -o poseidon-artifact-%j.out
# Submit from the artifact root, site options on the command line, e.g.
#   sbatch -A <account> -C gpu -q regular --gpus-per-node=1 -c 32 slurm/run_artifact.sbatch --quick
#   sbatch -A <account> -p normal --uenv=prgenv-gnu/25.11:v1 --view=default slurm/run_artifact.sbatch --quick
module load cudatoolkit 2>/dev/null || true
module load python 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:?submit this script from the artifact root}"
[ -x ./run_artifact.sh ] || { echo "run_artifact.sh not found: submit from the artifact root" >&2; exit 1; }

# The allocation's device outranks a GPU_ID pinned in local.env.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
fi

exec ./run_artifact.sh "$@"
