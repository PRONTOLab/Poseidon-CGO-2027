#!/bin/bash
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -J poseidon-artifact-parallel
#SBATCH -o poseidon-artifact-parallel-%j.out
#
# Slurm driver for the Poseidon-GPU artifact, ONE BENCHMARK PER GPU.
#
# Same shape as slurm/run_artifact.sbatch and the same rule about directives:
# only what is true everywhere is baked in (one node, a wallclock ceiling, a
# job name, a log file). A queue name, a node feature, a GPU count, a core
# count and a software environment differ at every machine and a wrong one
# makes the script UNSUBMITTABLE rather than merely suboptimal, so they go on
# the sbatch command line where the site checks them.
#
# The difference from run_artifact.sbatch is the device list. That script pins
# the whole run to one GPU of the allocation; this one hands the driver every
# GPU it owns as PARALLEL_GPUS, and run_artifact.sh gives each benchmark a
# device of its own. With more benchmarks than devices they run in waves, never
# two at once on one device: a shared device would time one benchmark's kernels
# inside another's, and no output would say so. On a four-GPU node the whole
# artifact therefore costs the wall clock of its slowest benchmark instead of
# the sum of all four.
#
# Submit FROM THE ARTIFACT ROOT so SLURM_SUBMIT_DIR points at it, with the
# artifact and the compiler build tree on a filesystem the compute nodes see
# (a scratch filesystem, not $HOME). POSEIDON_BUILD, PY and, off sm_120, the
# Herbie answers reach the job as run_artifact.sbatch's header describes.
#
#   export POSEIDON_BUILD=/path/to/Enzyme/build
#   cd /path/to/Poseidon-CGO-2027
#
#   # CSCS Alps / Daint (4x GH200 per node); the uenv supplies CUDA, gcc,
#   # cmake and ninja. The node is allocated whole, so all four GPUs are used.
#   sbatch -A <account> -p normal -t 04:00:00 \
#          --uenv=prgenv-gnu/25.11:v1 --view=default \
#          slurm/run_artifact_parallel.sbatch
#
#   # NERSC Perlmutter (4x A100 per node): ask for the whole node.
#   sbatch -A <account> -C gpu -q regular --gpus-per-node=4 -c 128 \
#          slurm/run_artifact_parallel.sbatch
#
#   # a plain Slurm cluster with a gpu partition
#   sbatch -A <account> -p gpu --gres=gpu:4 slurm/run_artifact_parallel.sbatch
#
# PARALLEL_GPUS and BENCHES are honoured when the caller sets them, which is
# how a node shared with something else, or a subset of the benchmarks, is
# given fewer devices:
#
#   sbatch ... --export=ALL,PARALLEL_GPUS="0 1 2" slurm/run_artifact_parallel.sbatch
#
# Everything (cost-model microbenchmark first, then the benchmarks) runs inside
# this one allocation; the binaries execute directly on the allocated GPUs, no
# per-step srun. Results land in results_<device>_<arch>_<timestamp>.tar.gz.
#
# Modules: adjust to your site. The clang/LLVM + Enzyme toolchain must already
# be built (README: Setup); CUDA is found via CUDA_HOME from the module, or
# from a uenv/spack view already on PATH.
module load cudatoolkit 2>/dev/null || true
module load python 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:?submit this script from the artifact root}"
[ -x ./run_artifact.sh ] || { echo "run_artifact.sh not found: submit from the artifact root" >&2; exit 1; }

# The devices this job owns. Slurm publishes them in CUDA_VISIBLE_DEVICES when
# it constrains the job to a subset of the node's GPUs, and sets nothing at all
# when it hands the node over whole, in which case the node's own list is the
# allocation. A caller-set PARALLEL_GPUS outranks both.
if [ -z "${PARALLEL_GPUS:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    PARALLEL_GPUS="$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' ')"
  else
    PARALLEL_GPUS="$(nvidia-smi -L 2>/dev/null | sed -n 's/^GPU \([0-9][0-9]*\):.*/\1/p' | tr '\n' ' ')"
  fi
fi
PARALLEL_GPUS="$(echo $PARALLEL_GPUS)"   # collapse to single-spaced
[ -n "$PARALLEL_GPUS" ] || {
  echo "no GPU is visible to this job: neither CUDA_VISIBLE_DEVICES nor nvidia-smi -L" >&2
  echo "names one. Submit inside a GPU allocation, or set PARALLEL_GPUS." >&2
  exit 1
}
export PARALLEL_GPUS

# Benchmark order decides which device each benchmark gets and which wave it
# lands in. fig2_drift is a hand-written precision comparison with no solver in
# it and it finishes in minutes, so it goes last: on a three-GPU allocation the
# three benchmarks that carry the paper's measured pipelines then each get a
# device to themselves and fig2_drift follows alone, instead of one of them
# sharing a device with it.
export BENCHES="${BENCHES:-genga_nbody ozp chase_eigensolver fig2_drift}"

# GPU_ID is deliberately NOT exported here: run_artifact.sh sets it per
# benchmark from PARALLEL_GPUS, and an exported one would pin every benchmark
# (and local.env's workstation pin would win over the allocation).
unset GPU_ID

# The mapping run_artifact.sh will apply, printed here so the job log carries
# it even for --check, which stops before the benchmarks start.
{
  read -ra _g <<< "$PARALLEL_GPUS"; read -ra _b <<< "$BENCHES"
  echo "==== packed mode: ${#_g[@]} GPU(s) [$PARALLEL_GPUS] for ${#_b[@]} benchmark(s) ===="
  for (( i=0; i<${#_b[@]}; i++ )); do
    printf '  wave %d  %-18s -> GPU %s\n' \
      $(( i / ${#_g[@]} + 1 )) "${_b[$i]}" "${_g[$(( i % ${#_g[@]} ))]}"
  done
}

exec ./run_artifact.sh "$@"
