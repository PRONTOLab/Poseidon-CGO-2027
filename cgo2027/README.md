# Poseidon-GPU — CGO 2027 Artifact

Reproduction package for the paper's evaluation: automatic numerical rewriting
of GPU kernels (algebraic rewrites, precision tuning, floating-point expansion,
tensor-core raising and tensor-core emulation) under a user accuracy target. The
artifact contains the four benchmarks the driver runs, the RTX 5090 cost model
the paper's solver decisions were made with, the profiles and Herbie answers
those decisions were made from, and scripts that regenerate every evaluation
figure and quoted number. **What a reviewer should see, row by row, is in
[`EXPECTED_RESULTS.md`](EXPECTED_RESULTS.md).**

The compiler is the Poseidon subproject of the Enzyme repository
(https://github.com/EnzymeAD/Enzyme, `enzyme/Poseidon`) at commit
`9e2e0572931248b4610940f2906e583914f9913e`, a git submodule next to this
directory (`sources/Enzyme`). Every
script here reaches it through `POSEIDON_BUILD=<that tree's build>`; the
container (Setup, A) presets it. Licences and provenance: `LICENSES.md`.

## Quick start

```sh
# in the container (Setup, A), or after building from source (Setup, B):
export POSEIDON_BUILD=<the Enzyme build tree configured with -DENZYME_POSEIDON=ON>   # preset in the container
./run_artifact.sh --check      # preflight: toolchain + GPU visible, nothing runs
./run_artifact.sh --picks      # every solver decision of EXPECTED_RESULTS.md, compile-only (no GPU)
./run_artifact.sh --picks --device all   # ... and Table 1's B300 and GH200 decisions (rtx5090|b300|gh200|all)
./run_artifact.sh --explain ozp-tau1e-7  # one row: candidates considered, applied, rejected (no GPU)
./run_artifact.sh --quick      # cost model + smoke tests (~15 min)
./run_artifact.sh              # cost model + full paper pipeline (hours)
GPU_ID=1 ./run_artifact.sh     # on GPU 1 (default 0, or Slurm's CUDA_VISIBLE_DEVICES)
./run_artifact.sh --from-scratch   # nothing precomputed: measured model, fresh profiles, Herbie searches
```

`--picks` and `--explain` start every solve from the shipped solve caches
(`devices/<device>/caches/`), which only save time; `--no-shipped-cache` solves
without them. `scripts/docker.sh build|run|picks|explain` runs the same in the
container of Setup, A.

### What a benchmark source says

Three user actions, and the source carries one attribute:

```c++
#include "poseidon/poseidon.h"

POSEIDON_OPTIMIZE __global__ void filter(double *W, const double *H, ...);
```

```sh
poseidon-clang++ $CUDA_FLAGS -poseidon-profile-generate app.cu -o app_profiled
./app_profiled                 # runs the workload; writes ./poseidon.profile
poseidon-clang++ $CUDA_FLAGS -poseidon-profile-use -poseidon-tau=1e-7 app.cu -o app
```

The annotated kernel's whole body is the optimization unit. Profile generation
supplies its own shadow buffers, and a source that also calls
`poseidon_metric("<name>", value)` once at the end of the profiling workload gets
each site's condition number measured against that number by the same run. The
benchmarks below pass explicit directories, cost models and budgets instead of
the defaults, which is why their command lines are longer. A site with no
`-poseidon-tau` is solved against `-poseidon-cost-budget=<share>`: the share of
the site's profiled cost a solution may add (negative: has to save). Every solve
writes the shares of its Pareto frontier to `budgets.txt` in its
`-poseidon-cache` directory, and prints the configuration it applied as a
`[poseidon] pin: <value>` line that `-poseidon-apply-rewrites=<value>` applies
again.

One command does everything, in order: preflight, **the cost model of the
selected GPU** (the shipped RTX 5090 model on an RTX 5090-class card, else a
measurement of this GPU), all four benchmarks, figure regeneration, and a
results bundle `results_<device>_<arch>_<timestamp>.tar.gz` (logs, TSVs, figures, the
cost model used, a config snapshot and which decisions matched
`EXPECTED_RESULTS.md`) ready to copy off the machine. A failing benchmark does
not abort the others; failures are listed at the end.

On a Slurm system, submit from the artifact root inside a GPU allocation. The
wrapper carries only site-neutral directives, so the queue, node feature, GPU
count and software environment go on the command line (see NERSC / Slurm
notes):

```sh
cd cgo2027
sbatch -A <account> -C gpu -q regular --gpus-per-node=1 -c 32 \
       slurm/run_artifact.sbatch          # or: ... --quick
```

## Layout

```
cgo2027/
  run_artifact.sh      # single-command driver (see Quick start)
  EXPECTED_RESULTS.md  # what a reviewer should see; tables generated from expected/
  expected/            # picks.tsv (every decision, by the compiler's own name),
                       # picks_b300.tsv, picks_gh200.tsv (Table 1's decisions of those devices) and
                       # claims.tsv (every checked number, its tolerance and source)
  LICENSES.md          # licences and provenance of every third-party part
  docker/Dockerfile    # the container image (Setup, A); Dockerfile.herbie adds Herbie
  env.sh               # sourceable environment for direct script invocation
  config.mk            # make-side twin of env.sh; benchmarks include it
  local.env            # OPTIONAL user pins (not shipped; see Configuration)
  devices/             # devices.tsv: the devices whose decisions --picks checks;
                       # b300/, gh200/: their saved cost models (+ x1e6/) and Herbie
                       # answers; <device>/caches/: the shipped DP tables and accuracy memos
  cost_models/         # cm_sm_120_RTX5090.csv + .herbie.rkt: the RTX 5090 model;
                       # x1e6/: the same model in 1e6 units, as GENGA prices with it;
                       # measured/ + target.env: written by the bootstrap
  scripts/
    detect.sh          # configuration resolver (all auto-detection lives here)
    bootstrap_cost_model.sh  # the cost model of this GPU (check or measure)
    build_toolchain.sh # Setup B as one resumable script (LLVM, cuMpSGEMM, compiler, --with-herbie)
    build_cumpsgemm.sh # its cuMpSGEMM step alone
    docker.sh          # build|run|picks|explain in the container
    check_picks.sh     # compile-only check of every decision (run_artifact.sh --picks [--device])
    explain.sh         # one row's candidates, compile-only (run_artifact.sh --explain)
    caches.sh, make_caches.sh  # the shipped solve caches: seeding, regeneration
    make_herbie_answers.sh     # regenerate a device's Herbie answers (needs Herbie)
    genga_x1e6.sh      # the x1e6 copy of a cost model GENGA prices with
    expected.sh, expected_results.py  # read expected/; regenerate EXPECTED_RESULTS.md
  slurm/run_artifact.sbatch  # Slurm submission wrapper (site options on the command line)
  benchmarks/
    fig2_drift/        # intro figure: FP64 vs FP32 vs 3xFP32 leapfrog drift
    genga_nbody/       # eval 5.1 — GENGA force + drift kernels (elementwise)
    ozp/               # eval 5.2 — density-matrix purification GEMM (Ozaki-II)
    chase_eigensolver/ # eval 5.3 — R-ChFSI eigensolver (sensitivity-guided)
  GEMM_RUNTIMES.md     # the GEMM runtimes a rewritten program calls
  deps/                # cuMpSGEMM, when built by scripts/build_cumpsgemm.sh
  figures/             # regenerated figure PDFs, named as in the paper's img/
  results/             # per-run result bundles (created by run_artifact.sh)
```

Each benchmark directory contains: verbatim kernel sources, a `Makefile` that
`include`s `../../config.mk`, a `scripts/` directory with the paper pipeline
(`scripts/rows.sh` compiles one row of `EXPECTED_RESULTS.md`), `quick.sh`
(kick-the-tires) and `run_all.sh` (full reproduction), `profiles/` (the profiles
the expected decisions were made from), `golden/rtx5090/` (the RTX 5090 run
`EXPECTED_RESULTS.md` cites),
`expected_results.md` (what the benchmark measures; its numbers are in the
top-level `EXPECTED_RESULTS.md`), and a plotting script that writes into
`../../figures/`. `genga_nbody/herbie_answers/` holds Herbie's answers for the
drift and unified kernels.

## Requirements

Hardware:
- **Required: an NVIDIA RTX 5090 or a similar card**, i.e. compute capability
  12.0 (`nvidia-smi --query-gpu=name,compute_cap --format=csv` prints `12.0`):
  the GeForce RTX 50 series and RTX PRO Blackwell workstation cards, on which
  FP64 runs at 1/64 of the FP32 rate. The shipped cost model applies to them
  (`EXPECTED_RESULTS.md`: Required). Other NVIDIA GPUs (sm_75 or newer) run the
  same pipeline on their own measured cost model; their decisions and numbers
  are theirs (`EXPECTED_RESULTS.md`: Optional).
- ~4 GB free disk for generated inputs (ozp synthesizes ~3 GB of matrices).

Software (all of it is in the container of Setup, A, which needs only Docker
with the NVIDIA Container Toolkit and a driver for CUDA 12.9 or newer):
- CUDA 12.x or 13.x (cuBLAS, cuSOLVER). Found via `CUDA_HOME` (module
  systems), `/usr/local/cuda*`, or `nvcc` on PATH. The paper used 12.9; on a
  toolkit newer than the clang supports, every compile prints `clang++: warning:
  CUDA version is newer than the latest partially supported version`, which is
  benign.
- The LLVM build the Poseidon plugin is built against (auto-detected from the
  build tree's CMakeCache; override with `CLANG`).
- CMake + Ninja to build the compiler. Herbie is needed only to search for
  algebraic rewrites yourself (`GENGA_HERBIE=run`); by default the shipped
  answers are replayed and no Herbie process starts.
- Python with numpy/matplotlib for plots (auto-detected; override with `PY`).

## Setup

The repository this directory is in holds everything needed except LLVM.
Clone it with its submodules:

```sh
git clone --recursive https://github.com/PRONTOLab/Poseidon-CGO-2027.git
# in an existing clone: git submodule update --init --recursive
```

```
Poseidon-CGO-2027/
  cgo2027/            this directory
  sources/Enzyme      submodule, the compiler: Enzyme at 9e2e0572931248b4610940f2906e583914f9913e
  sources/cuMpSGEMM   submodule: cuMpSGEMM 42dd92ea1c0e3d603bbb3f8f354f53e0967a9f4b + its submodules
  sources/herbie      submodule: Herbie 73ba1fe76f97d4cdb852a602addd82e3a983b209
  sources/herbie-num-taylor-terms.patch   adds the --num-taylor-terms option to it
```

Herbie (only for `GENGA_HERBIE=run`) is built from the `sources/herbie`
submodule with `sources/herbie-num-taylor-terms.patch` applied to a copy of it
(`scripts/build_toolchain.sh --with-herbie`, or `docker/Dockerfile.herbie`).

LLVM is llvm-project (https://github.com/llvm/llvm-project) at commit
`9d618fc0243103263ebf9bdcd08ef25e871d9c78`; both paths below build it from
GitHub's source tarball of that commit (sha256
`57eb91d0f644d74671300ea3e154fc22c9070aa87656844f0795c637693b4bdb`).

The results come from a compiler whose error-corrected (TCEC) GEMM dispatch
delegates to cuMpSGEMM, built as its own CMakeLists says (sm_80 code and PTX):
the shipped RTX 5090 cost model was measured against that build.

### A. Container (recommended)

Needs Docker with the NVIDIA Container Toolkit (a driver for CUDA 12.9 or
newer). The image holds CUDA 12.9.1, the pinned LLVM, cuMpSGEMM, the compiler,
Python with numpy and matplotlib, and this directory at `/artifact/cgo2027`
with `POSEIDON_BUILD` and `PY` set.

```sh
# either load the prebuilt image of the deposit
docker load -i <TODO: image file of the deposit>
# or build it, from the repository root (the directory holding cgo2027/ and sources/)
docker build -f cgo2027/docker/Dockerfile -t poseidon-cgo2027 .

mkdir -p out
docker run --rm -it --gpus all -v "$PWD/out:/out" poseidon-cgo2027
#   with CDI instead: --device nvidia.com/gpu=0
./run_artifact.sh --check          # inside the container
./run_artifact.sh --quick
cp results_*.tar.gz /out/          # the results bundle, onto the host
```

In the container on an RTX 5090, `--picks` took 14 minutes (no GPU needed)
and `--quick` 16 minutes. The full run takes about 2.5 hours on an RTX 5090
(the sum of the recorded parts of a run outside the container; not measured
in one piece). cuMpSGEMM's PTX is compiled for the GPU by the driver on first
use; `-v poseidon-cuda-cache:/root/.nv` keeps that result across containers.

Building the image took about 14 minutes on a 192-core machine at 96 jobs
(not counting the pull of the 10.2 GB base image) and gives an image of
13.9 GB (10.2 GB of it the CUDA base image, 3.1 GB the LLVM installation); the
build downloads the base image, the LLVM source tarball, Ubuntu packages and
two Python wheels. The image runs the RTX 5090 class (compute capability
12.0); for another GPU add `--build-arg CUDA_ARCHS=<cc>-real` (e.g. `90-real`
for an H100), which the Poseidon runtimes and calibrators are compiled for.
The default image has no Herbie: the artifact replays the shipped Herbie
answers. `docker build -f cgo2027/docker/Dockerfile.herbie -t
poseidon-cgo2027:herbie .` adds one for `GENGA_HERBIE=run` (it downloads
Racket, Rust and Herbie's dependencies).

### B. From source

Needs CMake (3.24 or newer), Ninja, git (Enzyme's configure step requires it;
nothing is cloned), a C++17 compiler, the CUDA toolkit (12.x
or 13.x), MPFR and GMP (`libmpfr-dev libgmp-dev`), zlib, zstd, libxml2 and
liblzma development files, and Python 3 with numpy and matplotlib. From the
repository root, with `P` any installation directory and `CUDA` the toolkit,
`scripts/build_toolchain.sh --prefix $P` does all of the following
(resumable; `-h` lists its options, e.g. `--llvm <existing LLVM 9d618fc>` and
`--with-herbie`); by hand:

```sh
P=$PWD/toolchain CUDA=/usr/local/cuda; mkdir -p $P

# 1. LLVM 9d618fc (clang, lld; X86 and NVPTX)
L=9d618fc0243103263ebf9bdcd08ef25e871d9c78
curl -fL -o llvm.tar.gz https://github.com/llvm/llvm-project/archive/$L.tar.gz
echo "57eb91d0f644d74671300ea3e154fc22c9070aa87656844f0795c637693b4bdb  llvm.tar.gz" | sha256sum -c -
tar -xzf llvm.tar.gz
cmake -G Ninja -S llvm-project-$L/llvm -B llvm-build \
    -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;lld" \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_ENABLE_PLUGINS=ON -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DCMAKE_INSTALL_PREFIX=$P/llvm
ninja -C llvm-build install

# 2. cuMpSGEMM 42dd92e (the shared library the compiler links)
cp -a sources/cuMpSGEMM $P/cuMpSGEMM                       # built next to its sources
cmake -S $P/cuMpSGEMM -B $P/cuMpSGEMM/build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc
make -C $P/cuMpSGEMM/build -j"$(nproc)" cumpsgemm

# 3. The compiler (Enzyme with Poseidon) at 9e2e05729
cmake -G Ninja -S sources/Enzyme/enzyme -B $P/poseidon-build \
    -DCMAKE_C_COMPILER=$P/llvm/bin/clang -DCMAKE_CXX_COMPILER=$P/llvm/bin/clang++ \
    -DLLVM_DIR=$P/llvm/lib/cmake/llvm \
    -DENZYME_POSEIDON=ON -DENZYME_CLANG=ON -DENZYME_MLIR=OFF \
    -DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=native \
    -DPOSEIDON_HERBIE_BINARY=$P/herbie/bin/herbie \
    -DPOSEIDON_TCEC_USE_CUMPSGEMM=ON -DPOSEIDON_CUMPSGEMM_ROOT=$P/cuMpSGEMM
ninja -C $P/poseidon-build

export POSEIDON_BUILD=$P/poseidon-build PY=<a python with numpy and matplotlib>
cd cgo2027 && ./run_artifact.sh --check
```

`CMAKE_CUDA_ARCHITECTURES=native` needs the GPU visible at configure time;
without one give it explicitly (`120-real` for the RTX 5090 class). The image
builds LLVM with Ubuntu's clang 18 and lld (`-DCMAKE_C_COMPILER=clang
-DCMAKE_CXX_COMPILER=clang++ -DLLVM_USE_LINKER=lld`); any C++17 compiler
works. In the image build, at 96 jobs, the LLVM step took 9 minutes (download,
configure, build, install), cuMpSGEMM 3 minutes and the compiler 20 seconds.
Disk: the LLVM source tree 2.8 GB, its build tree 4.4 GB and its installation
3.0 GB (the first two can be deleted after the install), the compiler's build
tree 330 MB, cuMpSGEMM 50 MB; a run generates about 4 GB of inputs.

`POSEIDON_HERBIE_BINARY` names where a Herbie is, or will be put: the
artifact replays the shipped answers and starts no Herbie, so the path may not
exist. Leaving the option out makes CMake clone and build upstream Herbie
(git, cargo and raco needed), which lacks the `--num-taylor-terms` option the
GENGA search uses. To search yourself (Choices, below), build Herbie from the
`sources/herbie` submodule with `sources/herbie-num-taylor-terms.patch` and the
commands of `docker/Dockerfile.herbie` (Racket 8.15 and Rust 1.85.1),
which `scripts/build_toolchain.sh --with-herbie` runs, and install it at that
path.

### One build tree — read this before mixing trees

**The driver (`poseidon-clang++`), the plugin (`Poseidon-*.so`) and the profiler
runtimes it links must come from the SAME build tree.** `poseidon-clang++` takes
the plugin and the runtimes from the tree (or installation) it runs from, and
`scripts/detect.sh` resolves the driver from `POSEIDON_BUILD`, so the default
path is consistent. Two symptoms appear when a second compiler tree is
introduced by hand (a second checkout, a plugin copied in, `POSEIDON_PLUGIN` set
in the environment):

- **An empty solve.** The compile succeeds, the solve enumerates one budget or
  none, and the log says every site is `uncovered — skipping under
  -poseidon-loose-coverage`: a profile written by a different plugin, whose site
  keys do not match. Re-run the profiling step with this compiler.
- **`nvlink: Prototype doesn't match`** for a profiler entry point: the plugin
  emits calls against one revision of the profiler runtime while the link takes
  another tree's archive.

`./run_artifact.sh --check` prints the driver and the plugin (with its md5) it
resolved.

## Configuration

There is nothing to edit. Every knob is resolved by `scripts/detect.sh` with
this precedence:

1. the caller's environment / make command line (`GPU_ID=0 ./run_artifact.sh`,
   `make CLANG=...`),
2. `local.env` in the artifact root — optional guard-style pins
   (`GPU_ID="${GPU_ID:-1}"`) for a machine you use repeatedly,
3. `cost_models/target.env` — written by the bootstrap when it measured this
   GPU (the measured cost model and this device's arch),
4. auto-detection from `POSEIDON_BUILD` (required): the source tree its
   CMakeCache names (for the public header and the GEMM runtime sources the
   reference arms compile), the driver and `poseidon-calibrate` of that build,
   the plugin-matching `clang++` from its CMakeCache; CUDA from
   `CUDA_HOME`/`/usr/local/cuda*`/PATH; GPU arch and name from `nvidia-smi`;
   `GPU_ID` from Slurm's `CUDA_VISIBLE_DEVICES` if set, else 0; the cost model
   as described next.

Variables: `POSEIDON_ROOT CLANG POSEIDON_CLANGXX POSEIDON_CALIBRATE CUDA
GPU_ARCH GPU_ID PY POSEIDON_INC OZAKI_RT TCEC_RT DIRECT_RT COST_MODEL`. Both
`make` targets and directly invoked pipeline scripts (`bash scripts/run_force.sh`)
resolve identically (via `config.mk` and `env.sh`, which share `detect.sh`).

## The cost model

The solver's decisions are driven by a per-device cost model (reciprocal
throughputs measured by `poseidon-calibrate`). The artifact never prices one
device's code with another device's model:

- On an **sm_120 GPU** (RTX 5090 class), the shipped
  `cost_models/cm_sm_120_RTX5090.csv` is used: the model measured on the RTX
  5090 the paper ran on. The bootstrap checks it against the device
  (`poseidon-calibrate --check`) and never rewrites it. With it, every decision
  in `EXPECTED_RESULTS.md` is reproduced exactly (`./run_artifact.sh --picks`
  checks that without a GPU).
- `./run_artifact.sh --calibrate` measures this GPU's own model instead (into
  `cost_models/measured/`, recorded in `cost_models/target.env`; delete that
  file to return to the shipped model). The decisions are then this card's own
  and may differ from the table; `EXPECTED_RESULTS.md` (Choices) says what is
  still promised.
- On **any other GPU** there is no default: `run_artifact.sh` measures it first
  (~15-30 min; the bootstrap passes the two inputs that belong to the artifact,
  the ozp surrogate profile the in-kernel arm enumerates from and R-ChFSI's
  rectangular product shapes). A direct `make` or pipeline script on such a GPU
  stops with the command to run.

GENGA prices with the device's model scaled uniformly by 1e6 (for the shipped
model `cost_models/x1e6/`, otherwise derived into `benchmarks/genga_nbody/measured/`):
a uniform scale leaves every decision unchanged, and the shipped Herbie answers
were searched against the Herbie platform generated from the scaled RTX 5090
model.

## Choices

| Knob | Default | Alternative |
|---|---|---|
| cost model | shipped RTX 5090 model on sm_120 | `./run_artifact.sh --calibrate` (this GPU's own) |
| Herbie (GENGA drift and unified) | `GENGA_HERBIE=replay`: the shipped answers in `benchmarks/genga_nbody/herbie_answers/`, no Herbie process | `GENGA_HERBIE=run`: Herbie searches itself (`HERBIE=<binary>` optional) |
| profiles | `PROFILES=fresh`: profiled on this GPU (GENGA drift and unified: the shipped profiles in replay mode, see below) | `PROFILES=shipped`: every solve reads `benchmarks/*/profiles/` |

The shipped answers were searched for the shipped RTX 5090 model: an answer is
keyed by the device family of the cost model and by its Herbie platform, so on
another device the drift and unified solves search (`GENGA_HERBIE=run`) or
replay an answer pack made for that device (`HERBIE_ANSWERS=<dir>`, a directory
with `kepler/` and `unified/`). A Herbie question also carries the profiled
value ranges and sensitivities, so in replay mode the GENGA
drift and unified solves read the shipped profiles the answers were searched
for, and each profiling step prints whether this GPU's fresh profile has the
same ranges. `GENGA_HERBIE=run`
searches from the fresh profile. It needs a Herbie that accepts
`--num-taylor-terms` (the drift kernel's Newton-step rewrite is a 4-term series,
and Herbie's own cap is 2 terms); with a Herbie that does not, the compile stops
naming the option, and `GENGA_TAYLOR_TERMS=` (empty) searches with Herbie's
own cap, which reproduces every GENGA row except the ones that need the 4-term
series (`EXPECTED_RESULTS.md` names them). A search takes about 10 minutes and
up to 13 GB of memory per solve.

## Kick the tires (~15 minutes)

`./run_artifact.sh --quick`, or per benchmark:

```sh
cd benchmarks/fig2_drift        && make all        # ~2 min: builds, runs, plots
cd ../genga_nbody               && ./quick.sh      # one force point + a short drift check
cd ../ozp                       && ./quick.sh      # the tau = 1e-7 auto-pick
cd ../chase_eigensolver         && ./quick.sh      # profile + the knee solve + verify
```

Each `quick.sh` builds one Poseidon-optimized configuration, prints whether the
compiler applied the configuration of `EXPECTED_RESULTS.md` (`[expected]`
lines), and measures it. Not a paper measurement.

## Full reproduction

`./run_artifact.sh`, which runs each benchmark's `run_all.sh` sequentially
(they assume exclusive use of the GPU). Figures land in `figures/`, named as in
the paper; the expected numbers, their tolerances and what reproduced means are
in `EXPECTED_RESULTS.md`. The heaviest step is GENGA's drift frontier (about 120 points, each timed and
integrated for 20k steps), then ozp's surrogate sweep, whose `N = 8192` leg is
FP64-bound (`SURROGATE_SIZES="1024 2048 4096"` skips it).

## Where the data lands, and how to redraw from it

**Per-benchmark, during a run.** Each benchmark writes its tables into its own
directory as TSV, one header row naming the columns, one row per measured point
(`force_full_g0.tsv`, `kepler_full_g0.tsv`, `unified.tsv`, `pareto.tsv`,
`dispatch_rungs.tsv`, `autopick.tsv`, `scvh_probe_on.tsv`, `ablation_knee.tsv`,
...). Frontier tables carry each point's share (`budget`) and the configuration
applied there (`pin`). The one exception is `benchmarks/ozp/traj_*.tsv`, the
per-iteration convergence traces: bare two-column `iteration<TAB>value` files.

**Collected, at the end of a run.** `run_artifact.sh` gathers everything into
one tree named for the device it ran on:

```
results/<GPU>_<arch>_<timestamp>/
  config.txt              date, host, GPU, clang, CUDA, cost model, compiler, knobs
  expected_picks.txt      every [expected] line: rows whose configuration was checked
  cm_*.csv                the cost model actually used
  target.env              the measured model's record, when this GPU was measured
  logs/<bench>.log        one log per benchmark
  figures/                the PDFs as regenerated by that run
  data/<bench>/*.tsv      every table that benchmark produced
  data/<bench>/traces/    per-run stdout captures
```

and tars it as `results_<GPU>_<arch>_<timestamp>.tar.gz`.

**Redrawing every figure, from one command:**

```sh
./scripts/plot_all.sh            # all figures into figures/
./scripts/plot_all.sh ozp        # one benchmark
```

No GPU and no prior run needed: each plotting script falls back to the shipped
`golden/rtx5090/` data, so this also works on a fresh unpack. It **overwrites**
`figures/`. The scripts: `ozp/plot_ozp.py` (Fig. 5),
`chase_eigensolver/plot_chase.py` (Fig. 6), `genga_nbody/genga_figures.py`
(Fig. 3, Fig. 4) and `fig2_drift/plot_drift.py` (Fig. 1); all import `scripts/figstyle.py`. Each takes
`--out` (genga: `--out-dir`) and `--data-dir` / `--trace-dir` to plot another
bundle elsewhere.

## How results map to the paper

| Paper claim / figure | Artifact location |
|---|---|
| Every number and decision, with tolerances | `EXPECTED_RESULTS.md` |
| Intro figure (drift panels + plot) | `benchmarks/fig2_drift`, `figures/intro_drift.pdf` |
| GENGA force/drift Pareto + ablations (Fig. 3) | `benchmarks/genga_nbody/scripts/run_force*.sh`, `run_kepler*.sh` |
| GENGA drift time series (Fig. 4) | `benchmarks/genga_nbody/scripts/run_kepler_timeseries.sh` |
| GENGA unified force-then-drift solve | `benchmarks/genga_nbody/scripts/run_unified.sh` |
| ozp convergence + frontier figure (Fig. 5) | `benchmarks/ozp/plot_ozp.py`; data from `scripts/run_pareto.sh`, `run_dispatch_rungs.sh`, `run_direct_tc_lib.sh`, `run_autopick.sh` |
| ozp tau auto-picks | `benchmarks/ozp/scripts/run_autopick.sh` |
| ozp surrogate figure (SCVH) | `benchmarks/ozp/scripts/run_surrogate.sh` |
| R-ChFSI residual-spectrum figure (Fig. 6), ablation | `benchmarks/chase_eigensolver/scripts/run_ladder.sh`, `plot_chase.py` |
| Per-site condition numbers (`Kappa` in each `.fpprofile`) | `benchmarks/chase_eigensolver`: written by `make profile`; `make probe` prints them |

## NERSC / Slurm notes

- Put the artifact **and** the compiler build tree on a filesystem the compute
  nodes see (a scratch filesystem; `$HOME` is slow). Build the LLVM + Enzyme
  toolchain once per system, in a batch job.
- The job finds the compiler through `POSEIDON_BUILD`: export it before
  `sbatch` (which hands the submitting shell's environment to the job) or pin
  it in `local.env`.
- `module load cudatoolkit python` (the sbatch wrapper does this); CUDA is
  found through `CUDA_HOME`. Where the site supplies its toolchain another way
  (a Spack view, a CSCS uenv), whatever is already on `PATH` is used.
- Submit `slurm/run_artifact.sbatch` **from the artifact root**. Everything,
  including the cost-model measurement, runs inside the one allocation; binaries
  run directly on the allocated GPU (no per-step `srun`).
- The wrapper's `#SBATCH` block is site-neutral (`-N 1`, a walltime, a job
  name, a log file). Queue, node feature, GPU count, core count and software
  environment belong on the sbatch command line; the header of the wrapper
  carries a worked example per site. Command-line options override the
  directives, `-t` included.
- Do not set `GPU_ID`: the wrapper adopts Slurm's `CUDA_VISIBLE_DEVICES` and
  exports it, which outranks a `local.env` pin left over from a workstation.
- The benchmarks assume exclusive use of their GPU; one GPU of one node
  (`--gpus-per-node=1`) is the intended shape. The default walltime is 12 h;
  `--quick` fits in well under 1 h. On a node with several GPUs,
  `slurm/run_artifact_parallel.sbatch` gives each benchmark a GPU of its own.
- Results come back as a single `results_<device>_<arch>_<timestamp>.tar.gz` in
  the artifact root; `scp` that one file.
- CSCS Alps (GH200, aarch64), uenv `prgenv-gnu/25.11:v1`: the uenv supplies
  gcc, CMake, Ninja and CUDA 12.9, but no MPFR (the compiler needs it: build
  it against the uenv's GMP and pass `-DCMAKE_PREFIX_PATH=<mpfr>`) and a python
  without numpy and matplotlib (make a venv and pin `PY` in `local.env`). Build
  LLVM with `-DLLVM_TARGETS_TO_BUILD="AArch64;NVPTX"`. Configure the compiler
  without cuMpSGEMM there (omit the two `POSEIDON_*CUMPSGEMM*` options of
  Setup): at the pinned commit its library does not link on that toolchain, and
  no GH200 decision of the paper uses the error-corrected dispatch. There is
  no Herbie on such a machine unless you build Racket and Rust yourself: give
  the build `-DPOSEIDON_HERBIE_BINARY=<a path that does not exist>` and the
  GENGA steps an answer pack (`HERBIE_ANSWERS`, Choices).

## Determinism notes

- With the shipped cost model and the shipped profiles every solver decision
  is deterministic; `./run_artifact.sh --picks` reproduces all of them on any
  machine with the compiler, no GPU needed.
- The DP caches (`cache*/table.json`, `cache*/budgets.txt`) regenerate per
  run; `--picks` and `--explain` seed them from `devices/<device>/caches/`
  (`scripts/caches.sh` says when an entry is used and when it is re-solved).
  `budgets.txt` is written only if absent: the pipeline scripts delete it
  whenever the profile or the cost model changes.
- Profiles: value ranges and execution counts are deterministic; sums of
  products and condition numbers (`Kappa`) are accumulated with atomics and
  differ from run to run in the last digits.
- The reduction trip counts the R-ChFSI solve needs travel in the `.fpprofile`
  header (`RedTrip = <slot> <trips>`), so the profile directory is the only thing
  `make solve` needs from `make profile`.
- ozp's FP64/direct-TC trajectories use atomic reductions and show ~0.3%
  run-to-run accuracy noise. The Ozaki-II residue GEMM is exact in INT32, but
  the TRS4 glue around it also reduces with atomics, so the deployed RMSD of an
  Ozaki-II rung reproduces to ~0.1%, not bitwise.
