# Floating up the Peak: Automatic Numerical Rewriting Enables Fast and Tensorized Accelerator Code

---

## Overview

The repository is the artifact of the paper "Floating up the Peak: Automatic
Numerical Rewriting Enables Fast and Tensorized Accelerator Code" (CGO 2027).
The paper presents a workflow, built on the Poseidon profile-guided numerical
optimization framework, that generates fast accelerator code for existing
applications. The artifact reproduces Fig. 3 to Fig. 6 and Table 1 of the
paper.

Archived copy: [doi:10.5281/zenodo.24288852](https://doi.org/10.5281/zenodo.24288852).

## Table of Contents

- [Requirements](#requirements)
- [Docker Image (Recommended)](#docker-image-recommended)
- [Build From Source](#build-from-source)
- [Reproducing Main Results](#reproducing-main-results)
- [Reusing the Artifact](#reusing-the-artifact)
- [Expected Results](#expected-results)
- [Miscellaneous](#miscellaneous)

## Requirements

The hardware depends on the results to reproduce:

1. Compiler decisions of all three GPUs (Table 1): any x86-64 Linux machine
   with Docker, 24 GB of memory and 20 GB of disk; no GPU.
2. RTX 5090 results (Fig. 3 to Fig. 6, RTX 5090 column of Table 1): an NVIDIA
   GPU with the NVIDIA Container Toolkit and a driver for CUDA 12.9 or newer.
   - An RTX 5090 reproduces Figures 3, 4, 5, and 6 and the RTX 5090 column of
     Table 1.
   - Other GeForce GPUs that also run FP64 at 1/64 of the FP32 rate and have
     tensor cores (for example RTX 5080, RTX 4090 or RTX 3090) may show
     similar trends ([GPUs](#gpus)).
3. Speedups of Table 1 on the B300 and the GH200: a B300 and a GH200
   (procedures in [GPUs](#gpus)).

## Docker Image (Recommended)

```sh
git clone --recursive https://github.com/PRONTOLab/Poseidon-CGO-2027.git
cd Poseidon-CGO-2027
```

The image `sbrantq/poseidon-cgo2027` comes from one of two sources; each
command runs in the repository root. The prebuilt image from Docker Hub:

```sh
docker pull sbrantq/poseidon-cgo2027
```

The image built from the sources (about 15 minutes on 192 cores):

```sh
scripts/docker.sh build
```

`scripts/docker.sh` runs the commands of the section
[Reproducing Main Results](#reproducing-main-results) in the container on a
fresh copy of the files of the clone and writes the results of a command to
`./out`.

## Build From Source

The build needs CMake 3.24 or newer, Ninja, git, a C++17 compiler, the CUDA
toolkit (12.x or 13.x), MPFR, GMP, zlib, zstd, libxml2 and liblzma development
files, and Python 3 with numpy and matplotlib. In the repository root:

```sh
scripts/build_toolchain.sh --prefix $PWD/toolchain
export POSEIDON_BUILD=$PWD/toolchain/poseidon-build
```

The commands of [Reproducing Main Results](#reproducing-main-results) without
Docker, in the repository root:

```sh
./run_artifact.sh --picks --device all     # decision check
./run_artifact.sh --check                  # setup check
./run_artifact.sh --quick                  # quick run
./run_artifact.sh                          # full evaluation
```

All modes and settings:

```sh
./run_artifact.sh -h
```

## Reproducing Main Results

### Kick the Tires (about 10 minutes)

The kick-the-tires phase of the artifact evaluation needs only these three
commands, in the repository root:

```sh
scripts/docker.sh picks --device all    # decision check, no GPU, 2 min
scripts/docker.sh run --check           # setup check: toolchain, GPU and cost model, seconds
scripts/docker.sh run --quick           # quick run: one strategy per benchmark, 3 min
```

The decision check compares every compiler decision with the decision
reported in the paper (`expected/picks*.tsv`) and prints `SAME` when they
agree. Expected output, among the other lines of the three commands:

```text
== rtx5090: 15 of 15 rows SAME
== b300: 5 of 5 rows SAME
== gh200: 5 of 5 rows SAME
==== setup check OK ====
==== [genga_nbody] OK ====
==== [ozp] OK ====
==== [chase_eigensolver] OK ====
==== DONE. Expected results: section Expected Results of README.md. ====
```

### Full Evaluation (about 2 hours)

Figures 3, 4, 5, and 6 and the RTX 5090 column of Table 1, on the RTX 5090, in
the repository root:

```sh
scripts/docker.sh run                   # all strategies and all figures
```

On the B300 and the GH200, for the speedups of Table 1 on those GPUs, after a
[build from source](#build-from-source), in the repository root:

```sh
./run_artifact.sh --table1
```

Without these GPUs, the decision check of [Kick the Tires](#kick-the-tires-about-10-minutes)
checks the decisions of all three GPUs on the CPU.

[Expected Results](#expected-results) lists the numbers reported in the paper.

The full evaluation produces all results of the following table. Each command
of the table reproduces one result alone and draws its figure into `figures/`.
The commands run in the repository root, in a build from source or in a shell
in the container, opened in the repository root (`/out` in the container is
`./out` on the host):

```sh
scripts/docker.sh shell
```

| Result | Command | Figure |
|---|---|---|
| Fig. 3 | `benchmarks/genga_nbody/run_fig3.sh` | `genga_force_drift.pdf` |
| Fig. 4 | `benchmarks/genga_nbody/run_fig4.sh` | `genga_drift_timeseries.pdf` |
| Sec. 4.2, force-then-drift | `benchmarks/genga_nbody/run_force_then_drift.sh` | |
| Fig. 5 | `benchmarks/ozp/run_fig5.sh` | `ozp_convergence.pdf` |
| Sec. 4.3, deployed sizes | `benchmarks/ozp/run_deployed_sizes.sh` | |
| Fig. 6 | `benchmarks/chase_eigensolver/run_fig6.sh` | `rchfsi_residuals.pdf` |
| Table 1, RTX 5090 column | `./run_artifact.sh --table1` | |

The commands of Sec. 4.2 and Sec. 4.3 end with a line that gives the measured
numbers and the row of [Expected Results](#expected-results) to compare them
with; the Table 1 command ends with the speedup and the accuracy of every row.
The lines of Sec. 4.2 and Sec. 4.3 again, from the tables of the last run:

```sh
scripts/text_results.sh
```

From the extracted results bundle of a full run (`./out/results_*.tar.gz`):

```sh
scripts/text_results.sh --dir <extracted bundle>/data
```

The figures, from the container to `./out` on the host:

```sh
cp -r figures /out/
```

## Reusing the Artifact

The workflow runs on other GPUs, with other settings, from fresh inputs, with
other compilers and tools, and on other benchmarks.

- Other GPUs: [GPUs](#gpus) lists the supported GPU classes. The cost model of
  the GPU in use:

  ```sh
  scripts/docker.sh run --bootstrap-only
  ```

- Other settings, among them the benchmarks (`BENCHES`), the ozp problem sizes
  (`SURROGATE_SIZES`) and the number of timed runs (`NUM_RUNS`):

  ```sh
  ./run_artifact.sh -h
  ```

- Fresh inputs: [Precomputed Inputs](#precomputed-inputs) gives the run that
  measures the cost model, profiles the benchmarks and searches with Herbie
  instead of using the shipped inputs.
- Other compilers and tools: the scripts run with any build of the compiler,
  selected by `POSEIDON_BUILD=<build directory>` in a build from source, or
  built into the image by `scripts/docker.sh build` after a checkout of another
  commit in `sources/Enzyme`. The decision check then reports each decision
  that differs from `expected/picks*.tsv`, and `--explain` shows why. Each
  figure is plotted from a table of speedup and error per strategy
  (`benchmarks/<benchmark>/*.tsv`), so another tool's results on the same
  benchmarks, timed with the same helper (`scripts/measure.sh`), can be added
  as one more curve by the plotting script of the benchmark
  (`genga_figures.py`, `plot_ozp.py` or `plot_chase.py`).
- Other benchmarks: `poseidon-clang++` optimizes a kernel marked with
  `POSEIDON_OPTIMIZE`, or one call annotated with `__poseidon_optimize`. The
  commands run in a shell in the container, which starts in the repository
  root:

  ```sh
  scripts/docker.sh shell
  ```

  The program, in `app.cu` in the repository root, marks the kernel to
  optimize:

  ```c++
  #include "poseidon/poseidon.h"
  POSEIDON_OPTIMIZE __global__ void scale(double *r, const double *x, int n) { ... }
  ```

  A profiled build, a run of it on a small input, and the optimized build
  for the accuracy target 1e-7 with the RTX 5090 cost model:

  ```sh
  CUDA_FLAGS="-O3 -ffp-contract=on --cuda-gpu-arch=sm_120 -L/usr/local/cuda/lib64 -lcudart"
  poseidon-clang++ $CUDA_FLAGS -poseidon-profile-generate app.cu -o app_profiled
  ./app_profiled                                           # writes ./poseidon.profile
  poseidon-clang++ $CUDA_FLAGS -poseidon-profile-use -poseidon-tau=1e-7 \
    -poseidon-cost-model=devices/rtx5090/cost_models/cm_sm_120_RTX5090.csv app.cu -o app
  ```

  The call form `__poseidon_optimize(fn, args...)` makes that one call the
  optimized site. The profiler sizes each pointer argument from the constant
  range the call reaches through it; a pointer whose range is known only at run
  time (a loop over a length the call is given) stops the compile with an
  error that names it. Such a call takes Enzyme's explicit argument list
  instead: `enzyme_dup` before the pointer and a shadow buffer of the same size
  after it, zeroed for an input and filled with 1.0 for an output, and
  `enzyme_const` before a pointer that needs no shadow:

  ```c++
  int enzyme_dup, enzyme_const;
  __poseidon_optimize(step, enzyme_dup, out, out_shadow, enzyme_const, x, n);
  ```

## Expected Results

The table lists the numbers that the paper reports for the NVIDIA RTX 5090.
The directories `benchmarks/<benchmark>/recorded/rtx5090/` hold the tables of
the recorded run. The compiler decisions and the accuracies are
deterministic; the RMSD of ozp is deterministic in the first two digits only,
because the purification loop of ozp sums with atomic operations. A speedup is
the run time of the scalar FP64 program divided by the run time of the
optimized program, both averaged over 5 runs; speedups vary with the host CPU,
the GPU clocks and the GPU load. A run of the artifact writes the measured
numbers to `data/<benchmark>/*.tsv` and `logs/<benchmark>.log` in the results
bundle.

| Result | In the paper | Reported |
|---|---|---|
| GENGA force kernel in FP32 | Fig. 3 left, Sec. 4.2 | 53x, L2 5e-6 |
| GENGA force kernel as a two-component FP32 expansion | Fig. 3 left, Table 1 | 5.3x, L2 1.8e-13 |
| GENGA drift kernel: Newton update as a quotient of two Taylor polynomials in FP32, f and g functions as a two-component FP32 expansion | Fig. 3 right, Table 1 | 5.6x, energy drift 3.0e-8 |
| GENGA drift kernel: naive FP32 downcast compared with the rewrite of the previous row | Fig. 3, Sec. 4.2 | 4600x more drift (1.4e-4 against 3.0e-8) |
| GENGA force-then-drift, one solve | Sec. 4.2 | 5.9x, L2 2.8e-14 |
| ozp, tau = 1e-7: Ozaki-II nm = 8 | Fig. 5, Sec. 4.3, Table 1 | 27.4x (15.8x over cuBLAS DGEMM), RMSD 1.4e-8 |
| ozp, tau = 1e-6: FP16TCEC | Fig. 5, Sec. 4.3, Table 1 | 31.4x, RMSD 1.7e-7 |
| ozp, tau = 1e-12: Ozaki-II nm = 12 | Fig. 5, Sec. 4.3 | 20.1x, RMSD 3.8e-13 |
| ozp, direct raise as a cuBLAS BF16 GEMM | Fig. 5, Sec. 4.3 | 71.1x, RMSD 4.8e-4 |
| ozp, direct raise as a cuBLAS TF32 GEMM | Fig. 5 | 54.0x, RMSD 1.6e-4 |
| ozp, Ozaki-II nm = 8 deployed at N = 1024 and N = 8192 (RMSD at N = 8192) | Sec. 4.3 | 12.9x and 53.3x, RMSD 2.5e-8 |
| R-ChFSI, TF32TCEC filter | Fig. 6, Sec. 4.4 | 1.85x, 64 of 64 locked |
| R-ChFSI, BF16TCEC filter | Fig. 6, Sec. 4.4 | 2.06x, eigenvalue L2 2.7e-15, 64 of 64 locked |
| R-ChFSI, BF16TCEC filter and Ozaki-II nm = 14 on a Rayleigh-Ritz product | Fig. 6, Sec. 4.4 | 2.15x, eigenvalue L2 4.6e-14, 64 of 64 locked |
| R-ChFSI, BF16 filter and Ozaki-II nm = 12 on a Rayleigh-Ritz product | Fig. 6, Sec. 4.4, Table 1 | 2.39x, 64 of 64 locked |
| R-ChFSI, sensitivity-blind strategy with the modelled cost of the previous row | Fig. 6, Sec. 4.4 | 0.48x, 0 of 64 locked |

## Miscellaneous

### GPUs

The decision check reproduces the compiler decisions of all three columns of
Table 1 without a GPU:

```sh
scripts/docker.sh picks --device all
```

Measuring the speedups of a column needs that GPU. On each of the three GPUs,
one command compiles every Table 1 row of that GPU with the configuration
reported in the paper and measures its speedup over scalar FP64 and its
accuracy (`table1/<gpu>/table1.tsv`), in the repository root:

```sh
scripts/docker.sh run --table1
```

The image runs on an x86-64 machine with a GPU of compute capability 12.0. A
B300 or a GH200 runs the same command without Docker after
[Build From Source](#build-from-source), in the repository root:

```sh
./run_artifact.sh --table1
```

On a Slurm cluster, the job runs it on the allocated GPU:

```sh
sbatch -A <account> <GPU options> slurm/run_artifact.sbatch --table1
```

A B300 rented on Modal runs it as well: [`modal/README.md`](modal/README.md).

Other GeForce RTX 50 GPUs run the prebuilt image with the RTX 5090 cost model.
Earlier GeForce GPUs (RTX 40 and RTX 30 series) need the image rebuilt on the
machine and measure their own cost model in the first run. The rebuild, in the
repository root:

```sh
scripts/docker.sh build
```

### Precomputed Inputs

The repository ships precomputed inputs; they are auxiliary and are not the
results that the artifact reproduces:

- Herbie answers (`devices/*/herbie_answers/`): cached Herbie searches, each
  about 10 minutes and up to 13 GB of memory.
- Cost models (`devices/*/cost_models/`): the operation timings of the RTX
  5090, the B300 and the GH200.
- Profiles (`benchmarks/*/profiles/`): the profiles of the three benchmarks,
  recorded on an RTX 5090.

A run that uses none of them measures the cost model, profiles the benchmarks
and searches with Herbie. The full evaluation, in the repository root:

```sh
scripts/docker.sh run --from-scratch
```

The quick run:

```sh
scripts/docker.sh run --quick --from-scratch
```

### Files

- `run_artifact.sh`: the entry point; `env.sh`, `config.mk`: the toolchain and
  GPU settings.
- `scripts/`: the Docker wrapper, the decision check and its explanation, the
  cost-model selection, and the timing and plotting helpers.
- `benchmarks/<name>/`: the annotated programs, their scripts, shipped
  profiles, and the recorded RTX 5090 run (`recorded/rtx5090/`).
- `devices/<gpu>/`, `expected/`: the cost model and Herbie answers of each of
  the three GPUs; the expected compiler decisions.
- `docker/`, `slurm/`, `modal/`: the image, the Slurm job, and the Modal scripts
  (currently configured for a B300).
- `sources/`: the compiler (Enzyme), cuMpSGEMM and Herbie as git submodules,
  and two patches that the build applies.
