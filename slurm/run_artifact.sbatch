#!/bin/bash
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -J poseidon-artifact
#SBATCH -o poseidon-artifact-%j.out
#
# Slurm driver for the Poseidon-GPU artifact.
#
# Only site-neutral directives are baked in: one node, a wallclock ceiling, a
# job name and a log file. A queue name, a node feature, a GPU count, a core
# count and a software environment are all different at every machine and a
# wrong one makes the script UNSUBMITTABLE rather than merely suboptimal
# (Perlmutter's `-q regular` is rejected outright at CSCS Alps, whose QOS is
# named `normal`), so they go on the sbatch command line, where they are
# checked against the site the user is actually on. Command-line options
# override the directives above, `-t` included.
#
# Submit FROM THE ARTIFACT ROOT so SLURM_SUBMIT_DIR points at it, with the
# artifact and the compiler build tree on a filesystem the compute nodes see
# (a scratch filesystem, not $HOME). The job finds the compiler through
# POSEIDON_BUILD (README: Setup): export it before sbatch, which hands the
# submitting shell's environment to the job, or pin it in local.env.
#
#   export POSEIDON_BUILD=/path/to/Enzyme/build
#   cd /path/to/Poseidon-CGO-2027
#
#   # NERSC Perlmutter (A100)
#   sbatch -A <account> -C gpu -q regular --gpus-per-node=1 -c 32 \
#          slurm/run_artifact.sbatch --quick
#
#   # CSCS Alps / Daint (GH200); the uenv supplies CUDA, gcc, cmake and ninja.
#   # Its python has neither numpy nor matplotlib: pin PY in local.env to one
#   # that has both (a venv made from the uenv's python works).
#   sbatch -A <account> -p debug -t 00:10:00 \
#          --uenv=prgenv-gnu/25.11:v1 --view=default \
#          slurm/run_artifact.sbatch --check          # preflight, 20 s
#   sbatch -A <account> -p normal -t 01:30:00 \
#          --uenv=prgenv-gnu/25.11:v1 --view=default \
#          slurm/run_artifact.sbatch --quick          # 20 min on a GH200
#
#   # a plain Slurm cluster with a gpu partition
#   sbatch -A <account> -p gpu --gres=gpu:1 slurm/run_artifact.sbatch --quick
#
# On a GPU other than an sm_120 card the run measures the device's cost model
# first (3 min on a GH200), and the GENGA drift and unified steps then need
# Herbie answers for that model: HERBIE_ANSWERS=<answer pack> or
# GENGA_HERBIE=run with a Herbie on the machine (README: Choices), exported
# like POSEIDON_BUILD. Without either, genga_nbody stops at its drift step and
# the other benchmarks still run.
#
# Everything (cost-model microbenchmark first, then the benchmarks) runs
# inside this one allocation; the binaries execute directly on the allocated
# GPU, no per-step srun. Results land in results_<device>_<arch>_<timestamp>.tar.gz.
#
# Modules: adjust to your site. The clang/LLVM + Enzyme toolchain must already
# be built (README: Setup); CUDA is found via CUDA_HOME from the module, or
# from a uenv/spack view already on PATH.
module load cudatoolkit 2>/dev/null || true
module load python 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:?submit this script from the artifact root}"
[ -x ./run_artifact.sh ] || { echo "run_artifact.sh not found — submit from the artifact root" >&2; exit 1; }

# Inside an allocation the allocation is the authority on which device to use.
# detect.sh reads GPU_ID before CUDA_VISIBLE_DEVICES and local.env sets GPU_ID,
# so a workstation pin would otherwise select a device outside the allocation.
# Exporting it here puts it in the caller environment, which outranks local.env.
# PARALLEL_GPUS (one benchmark per device) stays in charge when it is set.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ -z "${PARALLEL_GPUS:-}" ]; then
  export GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
fi

exec ./run_artifact.sh "$@"
