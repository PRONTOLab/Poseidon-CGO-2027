# Poseidon: CGO 2027 artifact

With this repository you check the evaluation of the paper. Poseidon is a
profile-guided compiler pass that rewrites GPU floating-point kernels under an
accuracy target you give it: algebraic rewrites, precision changes,
floating-point expansions, tensor-core raising and tensor-core emulation.

You do not need the authors' GPU. Every compiler decision of the paper is a
compile-time function of a profile, a cost model and the compiler, and the
repository ships all three for the three GPUs of the paper (RTX 5090, B300,
GH200). On any x86-64 machine with Docker you can check every decision and
read why the compiler made it, without a GPU. With an NVIDIA GPU you can then
rerun the benchmarks and measure.

## Pick your route

| Route | You need | Time | What you check |
|---|---|---|---|
| [Decisions, no GPU](#2-check-every-decision-without-a-gpu) | x86-64 machine with Docker | 14 min image build, then about 1 min (16 min without the shipped caches) | all 25 decisions of the paper (RTX 5090, B300, GH200), and why each was made |
| [Figures from the shipped data](#redraw-the-figures-without-a-gpu) | the same | seconds | every figure redraws from the recorded RTX 5090 run |
| [Kick the tires](#3-kick-the-tires-on-an-rtx-5090-about-16-minutes) | an RTX 5090 or another compute capability 12.0 card | 16 min | the toolchain end to end; five decisions applied and measured |
| [Full reproduction](#4-reproduce-the-paper-on-an-rtx-5090) | the same | about 2.5 h | every number and figure, claim by claim |
| [Another NVIDIA GPU](#another-nvidia-gpu) | any NVIDIA GPU of compute capability 7.5 or newer | as above, plus up to 30 min to measure the GPU | the same pipeline on that GPU's own cost model |
| [B300 at Modal](#a-b300-at-modal) | a Modal account with B300 access | about 50 min | the paper's B300 rows, measured |
| [Slurm system](#a-slurm-system) | a GPU node | 1 h 36 min on a four-GH200 node | the full pipeline in one batch job |
| [From scratch](#5-run-from-scratch) | a GPU and a Herbie build | hours (not timed end to end) | everything recomputed, nothing precomputed |

Every time above is from a recorded run unless the section says otherwise.

## 1. Setup

Clone the repository with its submodules:

```sh
git clone --recursive https://github.com/PRONTOLab/Poseidon-CGO-2027.git
cd Poseidon-CGO-2027
# in an existing clone: git submodule update --init --recursive
```

Archived copy: Zenodo DOI **TODO (owner)**.

The submodules hold the sources the toolchain is built from:

```text
sources/Enzyme      the compiler: Enzyme with Poseidon at 9e2e0572931248b4610940f2906e583914f9913e
sources/cuMpSGEMM   cuMpSGEMM 42dd92ea1c0e3d603bbb3f8f354f53e0967a9f4b and its submodules
sources/herbie      Herbie 73ba1fe76f97d4cdb852a602addd82e3a983b209
sources/herbie-num-taylor-terms.patch   adds the --num-taylor-terms option to it
```

Then choose the container (A) or a build from source (B).

### A. Container

You need Docker. For the GPU steps you also need the NVIDIA Container Toolkit
and a driver for CUDA 12.9 or newer. Build the image from the repository root:

```sh
scripts/docker.sh build          # image poseidon-cgo2027
```

Or load the prebuilt image of the deposit:

```sh
docker load -i <TODO (owner): image file of the deposit>
```

The build took about 14 minutes on a 192-core machine at 96 jobs, not
counting the pull of the 10.2 GB CUDA base image. It downloads the base image,
the LLVM source tarball, Ubuntu packages and two Python wheels. The image is
13.9 GB (3.1 GB of it LLVM). It holds CUDA 12.9.1, the pinned LLVM, cuMpSGEMM,
the compiler, Python with numpy and matplotlib, and this repository (without
`sources/`) at `/artifact`, with `POSEIDON_BUILD` and `PY` set.

The image is built for the RTX 5090 class (compute capability 12.0). For
another GPU, give its architecture:

```sh
scripts/docker.sh build --build-arg CUDA_ARCHS=90-real   # e.g. an H100 or a GH200
```

`scripts/docker.sh` runs every step of this README in the container:
`picks` and `explain` need no GPU; `run` and `shell` use the GPU (`--gpus all`,
or `GPU_ID=<n>` for one) and put the results bundle into `./out`. Containers
are capped at `MEMORY` (default `24g`).

### B. From source

You need CMake 3.24 or newer, Ninja, git, a C++17 compiler, the CUDA toolkit
(12.x or 13.x), MPFR and GMP (`libmpfr-dev libgmp-dev`), zlib, zstd, libxml2
and liblzma development files, and Python 3 with numpy and matplotlib. Run,
from the repository root:

```sh
scripts/build_toolchain.sh --prefix $PWD/toolchain
# no GPU visible on this machine: add --cuda-archs 120-real (or your GPU's)
# an LLVM 9d618fc already built: add --llvm <its install or build tree>
```

It builds LLVM, cuMpSGEMM and the compiler into the prefix, resumes after the
last finished step when you rerun it, and prints the line to use at the end:

```sh
export POSEIDON_BUILD=$PWD/toolchain/poseidon-build PY=<a python with numpy and matplotlib>
./run_artifact.sh --check
```

In the image build, at 96 jobs, LLVM took 9 minutes, cuMpSGEMM 3 minutes and
the compiler 20 seconds. Disk: LLVM's source tree 2.8 GB and build tree 4.4 GB
(deleted after the install unless `--keep-llvm-sources`), its installation
3.0 GB, the compiler's build tree 330 MB, cuMpSGEMM 50 MB. A GPU run
generates about 4 GB of inputs. [Build from source, by hand](#build-from-source-by-hand)
lists the same steps as commands.

Every command below that starts with `./run_artifact.sh` works the same in
both setups: inside the container, or on the host with `POSEIDON_BUILD`
exported. `scripts/docker.sh shell` opens a shell in the container, on the
GPU when the machine has one.

## 2. Check every decision without a GPU

### Check the decisions

```sh
scripts/docker.sh picks --device all
# from source: ./run_artifact.sh --picks --device all
```

This compiles each decision row of the paper for its device, from the shipped
profile and that device's saved cost model, and compares the configuration the
compiler applied with the expected one. It took 66 seconds. You should see
(the CUDA container banner and a "NVIDIA Driver was not detected" warning come
first; both are expected without a GPU):

```text
compiler: /opt/poseidon/build/Poseidon/bin/poseidon-clang++
plugin:   /opt/poseidon/build/Poseidon/bin/../lib/Poseidon-24.so (md5 c6f9bb0c)
CUDA:     /usr/local/cuda-12.9 (release 12.9, V12.9.86)
caches:   seeded from devices/<device>/caches/
== rtx5090 (sm_120): NVIDIA GeForce RTX 5090: the cost model the paper decided every decision row with
genga-force-fp32         SAME               5s  seeded; table hits 1
genga-force-expansion    SAME               6s  seeded; table hits 1
...
rchfsi-blind             SAME               2s  table hits 0
== rtx5090: 15 of 15 rows SAME
== b300 (sm_103): NVIDIA B300 SXM6 AC: the models of the paper's B300 column of Table 1
t1-force-b300            SAME               7s  seeded; table hits 1
...
t1-ozp7-b300             SAME               2s  table hits 0, table MISSES 1 (device)
...
== b300: 5 of 5 rows SAME
== gh200 (sm_90): NVIDIA GH200 120GB: the model of the paper's GH200 column of Table 1
...
== gh200: 5 of 5 rows SAME
wrote picks_check/picks.tsv
```

Read it like this:

- `SAME`: the compiler applied exactly the configuration of
  [`EXPECTED_RESULTS.md`](EXPECTED_RESULTS.md). That file names each row's
  configuration in words. A row that is not `SAME` prints `DIFFERENT` or
  `COMPILE-FAILED`, and the command exits with status 1.
- The last column says what the shipped solve caches did. They only save time.
  `table MISSES 1 (device)` means the compile solved that table itself; it
  cannot change a decision.
- `--device rtx5090`, `b300` or `gh200` checks one device. Without `--device`
  only the RTX 5090 rows are checked.

To confirm that the caches decide nothing, solve every row yourself:

```sh
scripts/docker.sh picks --device all --no-shipped-cache
```

All 25 rows came out `SAME` this way too; it took 16 minutes (946 seconds)
instead of 66 seconds.

### Read why the compiler made a decision

```sh
scripts/docker.sh explain ozp-tau1e-7
# from source: ./run_artifact.sh --explain ozp-tau1e-7
```

It compiles the one row as `--picks` does and prints, for each FP subgraph,
the candidates the compiler considered, the one it applied and why it rejected
the others, each with its modelled error and its cost change (a share of the
profiled cost). It took 2.5 seconds. For row `ozp-tau1e-7` (the density-matrix
GEMM of ozp under the accuracy target 1e-7) you should see:

```text
== ozp-tau1e-7: Ozaki-II, 8 moduli (host-dispatched); the paper's nm = 9 is the pin _Z10matmul_optPdPKdS1_:M0=ozaki-ii nm=9 wmma m16n16k16 s8/s32
   selected by tau=1e-7, cost model cm_sm_120_RTX5090.csv
   applied: the configuration of the row (SAME)
...
  applied M0_11 (matmul): ozaki-ii nm=8 wmma m16n16k16 s8/s32
      modelled domain error 2.3e-08; cost change -0.968923
  considered:
    id       status    kind           error  cost change  rewrite / reason
...
    M0_2     rejected  matmul        0.0024    -0.932118  wmma m16n16k16 bf16/f32
                                                            modelled domain error 0.0024 is above the target 1e-07
...
    ... 14 more (--all)
```

The FP16, BF16 and TF32 tensor-core raises are cheaper but their modelled
error is above the target, so the compiler takes the cheapest Ozaki-II
configuration that meets it. Add `--all` to see every candidate, for example
`M0_12 ... 1.5e-09 ... ozaki-ii nm=9 ... a cheaper candidate also meets the target`.

Now explain the same product on the GH200, whose FP64 tensor cores make a
native DGEMM the cheapest candidate that meets the target:

```sh
scripts/docker.sh explain t1-ozp7-gh200
```

```text
  applied M0_23 (matmul): native-dgemm cublas f64/f64
      modelled domain error 1.4e-15; cost change -0.99076
```

Rows selected by a cost budget end with the Pareto frontier and the flag that
selects each point; `*` marks the row's point. For `genga-force-expansion`:

```text
Pareto frontier (cost, modelled error; * = selected):
            cost  modelled error  rewrites                 select with
       0.0197887         1.7e-07  PT0_0                    -poseidon-cost-budget=-0.9802112539377893
...
  *     0.265879         5.4e-15  PT0_11                   -poseidon-cost-budget=-0.7341213131565124
        0.496649         2.4e-15  PT0_12                   -poseidon-cost-budget=-0.5033506060243876
```

`scripts/docker.sh explain --list` names every row, per device. Other
options: `--all`, `--max N`, `--function <name>`, `--format json`,
`--format remarks`.

### Keep the files

`scripts/docker.sh picks` and `explain` also leave their files in `./out` on
your machine: `out/picks_check/` (one compile log per row and `picks.tsv`) and
`out/explain/<row>/` (`report.txt`, the compile log, JSON). The files belong
to root, as is usual for Docker.

### Redraw the figures without a GPU

```sh
mkdir -p out
docker run --rm -v "$PWD/out:/out" poseidon-cgo2027 bash -c './scripts/plot_all.sh && cp figures/*.pdf /out/'
# from source: ./scripts/plot_all.sh       (or ./scripts/plot_all.sh ozp for one benchmark)
```

Each plotting script falls back to the recorded RTX 5090 run in
`benchmarks/*/golden/rtx5090/` when the benchmark has not run on this machine.
You get `intro_drift.pdf` (Fig. 1), `genga_force_drift.pdf` (Fig. 3),
`genga_drift_timeseries.pdf` (Fig. 4), `ozp_convergence.pdf` (Fig. 5),
`ozp_surrogate.pdf` and `rchfsi_residuals.pdf` (Fig. 6). `plot_all.sh`
overwrites `figures/`, and a redraw is not byte-identical to an earlier one
(matplotlib stamps a creation date).

With no GPU, this is as far as you can go: you have checked every decision of
the paper and can read the measured numbers behind them in
`benchmarks/*/golden/rtx5090/`, which [`EXPECTED_RESULTS.md`](EXPECTED_RESULTS.md)
cites row by row. Timing and accuracy need a GPU.

## 3. Kick the tires on an RTX 5090 (about 16 minutes)

Check your GPU first. The shipped RTX 5090 cost model applies when this prints
`12.0` (the GeForce RTX 50 series and the RTX PRO Blackwell workstation cards,
on which FP64 runs at 1/64 of the FP32 rate):

```sh
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

Any other GPU: go to [Another NVIDIA GPU](#another-nvidia-gpu).

```sh
scripts/docker.sh run --check    # preflight: toolchain and GPU, nothing runs
scripts/docker.sh run --quick    # cost-model check and one smoke test per benchmark
GPU_ID=1 scripts/docker.sh run --quick   # on GPU 1 instead of all GPUs
```

From source, the same is `./run_artifact.sh --check` and
`./run_artifact.sh --quick` (`GPU_ID=1 ./run_artifact.sh --quick` for GPU 1).

`--check` prints what it resolved and ends with `==== preflight OK ====`:

```text
  DRIVER        = /opt/poseidon/build/Poseidon/bin/poseidon-clang++
  PLUGIN        = /opt/poseidon/build/Poseidon/bin/../lib/Poseidon-24.so (md5 c6f9bb0c)
  CLANG         = /opt/llvm/bin/clang++ (clang version 24.0.0git)
  CUDA          = /usr/local/cuda-12.9 (release 12.9, V12.9.86)
  GPU           = id 0: NVIDIA GeForce RTX 5090, 12.0
```

`--quick` took 944 seconds in the container on an RTX 5090. It checks the
shipped cost model against your GPU (`[bootstrap] cost model: the shipped RTX
5090 model (cm_sm_120_RTX5090.csv)`), runs one smoke test per benchmark, and
ends with:

```text
==== configurations checked against EXPECTED_RESULTS.md ====
[expected] rchfsi-knee: the compiler applied the configuration of EXPECTED_RESULTS.md
[expected] genga-force-fp32: the compiler applied the configuration of EXPECTED_RESULTS.md
[expected] genga-drift-naive: the compiler applied the configuration of EXPECTED_RESULTS.md
[expected] genga-drift-algebraic: the compiler applied the configuration of EXPECTED_RESULTS.md
[expected] ozp-tau1e-7: the compiler applied the configuration of EXPECTED_RESULTS.md
==== DONE. Compare the bundle against EXPECTED_RESULTS.md. ====
```

The bundle `results_RTX5090_sm_120_<timestamp>.tar.gz` lands in `./out` (from
source: in the repository root). Its `logs/<bench>.log` hold the smoke tests'
numbers. The recorded run printed:

```text
# logs/fig2_drift.log
GPU FP64 (as written)          5.471 ms     1.00x
GPU 3xFP32 emulation           2.620 ms     2.09x
# logs/genga_nbody.log
  force FP32: baseline=186.023026ms opt=3.502465ms speedup=53.1x L2=5.056976e-06  (expect ~51x, 5.056976e-06)
  drift @2000 steps: naive-FP32=1.901524e-05  Poseidon=1.296310e-08  (Poseidon should be ~100-1000x smaller)
# logs/ozp.log
  tau      compile-time pick                        RMSD          time       x FP64    x cuBLAS
  1e-7     ozaki-ii nm=8 wmma m16n16k16 s8/s32      1.375121e-08  42.223392ms 27.295x   15.699x
# logs/chase_eigensolver.log
OPT eigval L2 vs planted = 1.230509e-11 (locked 64/64)
```

These are smoke tests, not paper measurements. What to check: every
`[expected]` line says "applied the configuration", every benchmark says `OK`,
the accuracy values match the ones above, and the speedups are near them. A
run on a GPU that is busy or throttled shows lower ratios (see
[Troubleshooting](#troubleshooting)).

You can also run one benchmark's smoke test by itself:

```sh
cd benchmarks/fig2_drift        && make all        # builds, runs, plots Fig. 1
cd ../genga_nbody               && ./quick.sh      # one force point and a short drift check
cd ../ozp                       && ./quick.sh      # the tau = 1e-7 decision, measured
cd ../chase_eigensolver         && ./quick.sh      # profile, the rchfsi-knee solve, verify
```

## 4. Reproduce the paper on an RTX 5090

```sh
scripts/docker.sh run            # from source: ./run_artifact.sh
```

This runs, in order: the preflight, the cost model of your GPU, the four
benchmarks one after another (each assumes the GPU to itself), the figures,
and the results bundle. A failing benchmark does not stop the others; the end
of the run lists the failures. Expect about 2.5 hours on an RTX 5090: that is
the sum of the recorded parts of a run, not one timed run. The longest step is
GENGA's drift frontier (about 120 points, each timed and integrated for 20k
steps), then ozp's surrogate sweep, whose `N = 8192` leg is FP64-bound.

To run part of it, set the variable in front of either form of the command
(`scripts/docker.sh run` hands these variables to the container):

```sh
BENCHES="ozp chase_eigensolver" ./run_artifact.sh     # some benchmarks
SURROGATE_SIZES="1024 2048 4096" ./run_artifact.sh    # skip ozp's N = 8192 leg
PARALLEL_GPUS="0 1 2 3" ./run_artifact.sh             # one benchmark per GPU
```

### Compare your numbers with the paper's, claim by claim

Unpack the bundle and print every number your run measured, keyed by the claim
it belongs to:

```sh
mkdir -p results && tar -xzf out/results_RTX5090_sm_120_<timestamp>.tar.gz -C results   # container route only
python3 scripts/expected_from_run.py results/RTX5090_sm_120_<timestamp> mine --dry
```

From source, the run already left the directory `results/<GPU>_<arch>_<timestamp>/`
next to the bundle; skip the `tar` line.

Each line is `<claim>[<metric>]`, your value, and the file it came from.
`--dry` writes nothing. The quick run's bundle printed:

```text
fig2-gpu[0]	2.09x faster (5.471 -> 2.620 ms)	benchmarks/fig2_drift/golden/mine/fig2_drift.log
fig2-gpu[1]	9.0x slower (31.022 -> 277.926 ms)	benchmarks/fig2_drift/golden/mine/fig2_drift.log
fig2-gpu[2]	1.611652e-20	benchmarks/fig2_drift/golden/mine/drift.csv
ozp-tau1e-7[0]	1.375121e-08	benchmarks/ozp/golden/mine/autopick.tsv:2
ozp-tau1e-7[1]	27.295x	benchmarks/ozp/golden/mine/autopick.tsv:2
ozp-tau1e-7[2]	15.699x	benchmarks/ozp/golden/mine/autopick.tsv:2
```

The third column names where a run without `--dry` would copy the data; the
value came from your bundle's `data/` directory. A full run prints every claim.

Then open [`EXPECTED_RESULTS.md`](EXPECTED_RESULTS.md) and compare each line
with the claim's `Expected` cell and `Tolerance`; metric `[0]` is the claim's
first row, `[1]` its second. A claim is reproduced when:

1. its decision is the expected one: the run's `[expected]` lines (collected
   in the bundle's `expected_picks.txt`) say "applied the configuration". A
   line that says `DIFFERENT configuration` prints the expected and the
   applied configuration below it; see [Troubleshooting](#troubleshooting);
2. its accuracy matches to the printed digit where the computation is
   deterministic (GENGA, R-ChFSI, the fig2 drift file), and to about 0.5% for
   ozp's RMSD (its purification reduces with atomics);
3. its speedup is within 10% of `Expected` on an idle RTX 5090 at stock clocks
   (R-ChFSI: about 4% run to run). Repeated runs of the same binaries landed
   within -8% to +5%.

`Paper` is what the paper printed. Where this compiler picks another
configuration than the paper reports, or measures a different number for the
same one, [`EXPECTED_RESULTS.md`](EXPECTED_RESULTS.md) explains each case
under "Where the configuration differs" and "Where the numbers differ".

### Where each paper result comes from

| Paper result | Command (in the benchmark directory) | Your data |
|---|---|---|
| Fig. 1, intro drift | `benchmarks/fig2_drift`: `make all` | `drift.csv`, `figures/intro_drift.pdf` |
| Fig. 3 left, GENGA force frontier and expansion-off ablation | `benchmarks/genga_nbody`: `make force force-mfoff` | `force_full_g0.tsv`, `force_mfoff.tsv` |
| Fig. 3 right, GENGA drift frontier and Herbie-off ablation | `make kepler kepler-herbieoff` | `kepler_full_g0.tsv`, `kepler_herbieoff_g0.tsv` |
| Fig. 4, GENGA drift time series | `make timeseries` | `kepler_timeseries.tsv` |
| GENGA unified force-then-drift solve | `make unified` | `unified.tsv` |
| Fig. 5, ozp convergence and frontier | `benchmarks/ozp`: `make pareto dispatch direct-tc-lib autopick` | `pareto.tsv`, `dispatch_rungs.tsv`, `direct_tc_lib.tsv`, `autopick.tsv`, `traj_*.tsv` |
| ozp accuracy-target decisions | `bash scripts/run_autopick.sh 1e-7` (or `1e-6`, `1e-12`) | `autopick.tsv` |
| ozp surrogate figure (profile at N = 512, deploy larger) | `make surrogate` | `surrogate.tsv` |
| Fig. 6, R-ChFSI residual spectrum and sensitivity-blind ablation | `benchmarks/chase_eigensolver`: `bash scripts/run_ladder.sh` | `scvh_probe_on.tsv`, `scvh_probe_off.tsv`, `ablation_knee.tsv`, `traces/` |
| R-ChFSI per-site condition numbers | `make profile`, then `make probe` prints them | `Kappa` lines in each `.fpprofile` |
| Table 1, B300 and GH200 columns | `./run_artifact.sh --picks --device b300` (`gh200`) | `picks_check/picks.tsv` |

Each benchmark writes its tables into its own directory as TSV files with one
header row. Frontier tables record each point's share of the profiled cost
(`budget`) and the configuration applied there (`pin`). The exception is
`benchmarks/ozp/traj_*.tsv`: per-iteration traces, two columns
`iteration<TAB>value`. To run one benchmark step by step, follow its
`expected_results.md`.

At the end, `run_artifact.sh` collects everything into a directory named for
your GPU and tars it as `results_<GPU>_<arch>_<timestamp>.tar.gz`:

```text
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

To redraw the figures from a bundle elsewhere, run a plotting script with
`--out` (GENGA: `--out-dir`) and `--data-dir` / `--trace-dir`. The scripts:
`benchmarks/fig2_drift/plot_drift.py` (Fig. 1),
`benchmarks/genga_nbody/genga_figures.py` (Fig. 3, Fig. 4),
`benchmarks/ozp/plot_ozp.py` (Fig. 5), `benchmarks/ozp/plot_surrogate.py`, and
`benchmarks/chase_eigensolver/plot_chase.py` (Fig. 6).

## If you do not have an RTX 5090

### No GPU at all

Do [section 2](#2-check-every-decision-without-a-gpu). It checks every
decision of the paper, for all three GPUs, and redraws the figures from the
recorded runs. The recorded measurements behind every number are in
`benchmarks/*/golden/rtx5090/`, cited line by line in
[`EXPECTED_RESULTS.md`](EXPECTED_RESULTS.md).

### Another NVIDIA GPU

Build the image for your GPU's architecture
(`scripts/docker.sh build --build-arg CUDA_ARCHS=<cc>-real`, e.g. `90-real`),
or build from source on the machine with the GPU. Then run the same commands
as on an RTX 5090:

```sh
scripts/docker.sh run --quick    # or the full run: scripts/docker.sh run
```

On a GPU other than compute capability 12.0 the run does this differently:

- It first measures your GPU's cost model with `poseidon-calibrate`, into
  `cost_models/measured/`, recorded in `cost_models/target.env` for every later
  build. This took 3 min 25 s on a GH200 and 4 minutes on a B300; the scripts
  allow 15 to 30 minutes. `./run_artifact.sh --bootstrap-only` does only this
  step.
- Every decision is then your GPU's own. The run says so before the benchmarks:

  ```text
  NOTE: this run prices with a cost model measured on this device (<model>.csv).
        Its decisions are this model's own; they are reported, not compared with the
        RTX 5090's. ...
  ```

  Each `[expected]` line reads `decided with this run's cost model (...)` and
  `not compared`.
- GENGA's drift frontier, drift time series and unified solve are skipped,
  with a `[skip] ...: SKIPPED, not failed.` note: they need Herbie answers for
  your GPU's cost model. To run them, search with Herbie (build the image with
  `scripts/docker.sh build --herbie`, then `GENGA_HERBIE=run` in
  `scripts/docker.sh shell`), or replay an answer pack made for your model
  (`HERBIE_ANSWERS=<dir>`, see [Choices](#choices)).
- After the benchmarks the run checks the shipped decisions of all three
  devices compile-only (`--picks --device all`; a quick run checks the RTX 5090
  rows) and prints `==== this GPU priced with its own model ====` with the
  result.

Your speedups are your GPU's; the 10% band of `EXPECTED_RESULTS.md` is for an
RTX 5090. Check the paper's qualitative claims instead:

- each accuracy-target decision of ozp meets its target;
- the sensitivity-guided R-ChFSI solve rewrites the filter before any other
  product, and the sensitivity-blind solve does not lock every eigenpair;
- with Herbie, the GENGA drift frontier holds algebraic rewrites that are more
  accurate than the naive FP32 downcast at equal cost;
- every accuracy value of a configuration that also appears in
  `EXPECTED_RESULTS.md` matches it.

A GPU with fast FP64 (an A100, H100 or GH200) decides differently from an RTX
5090: it keeps more in FP64 and gains less. The GH200 column of Table 1 is
such a GPU's result, and `scripts/docker.sh explain t1-rchfsi-gh200` shows its
reasoning.

On a GH200 the quick run took 17 minutes after the 3.5-minute measurement of
the cost model, in a recorded run.

### A B300 at Modal

[`modal/README.md`](modal/README.md) runs the paper's B300 rows on one B300
at Modal: build (26 minutes the first time), smoke test (3 minutes),
cost-model comparison (4 minutes) and the rows (17 minutes). It compiles the
decisions that `./run_artifact.sh --picks --device b300` checks, measures
them, and lists the numbers measured on 2026-11-12 to compare with.

### A Slurm system

Submit from the repository root, inside a GPU allocation. The wrapper carries
only site-neutral directives; give the queue, node feature, GPU count, core
count and software environment on the command line:

```sh
export POSEIDON_BUILD=<the compiler's build tree>
sbatch -A <account> -C gpu -q regular --gpus-per-node=1 -c 32 slurm/run_artifact.sbatch          # or: ... --quick
sbatch -A <account> <site options> slurm/run_artifact_parallel.sbatch    # one benchmark per GPU of the node
```

The header of `slurm/run_artifact.sbatch` has a worked example per site. The
full run through `run_artifact_parallel.sbatch` took 1 h 36 min on a node with
four GH200s. [Slurm notes](#slurm-notes) has the site details.

## 5. Run from scratch

By default the run reuses inputs the repository ships. Each one only saves
time or pins a decision to the paper's, and you can regenerate each:

| Shipped input | Where | Used by | Regenerate with |
|---|---|---|---|
| RTX 5090 cost model | `cost_models/cm_sm_120_RTX5090.csv` (+ `x1e6/`) | every run on a compute capability 12.0 GPU | `./run_artifact.sh --calibrate` (your GPU's own model) |
| B300 and GH200 cost models | `devices/{b300,gh200}/cost_models/` | `--picks --device b300` / `gh200` | `--calibrate` on that GPU |
| Profiles | `benchmarks/*/profiles/` | `--picks`, `--explain`, `PROFILES=shipped`, GENGA drift and unified in replay mode | nothing to do: a GPU run profiles on your GPU by default (`PROFILES=fresh`) |
| Herbie answers | `benchmarks/genga_nbody/herbie_answers/`, `devices/{b300,gh200}/herbie_answers/` | GENGA drift and unified solves (`GENGA_HERBIE=replay`) | `GENGA_HERBIE=run`, or `scripts/make_herbie_answers.sh --device <d>` (no GPU) |
| Solve caches (DP tables, matrix-product accuracy memos) | `devices/<device>/caches/` | `--picks`, `--explain` | `--no-shipped-cache`, or `scripts/make_caches.sh --device <d>` (no GPU) |

To use none of them:

```sh
./run_artifact.sh --from-scratch            # or: ./run_artifact.sh --quick --from-scratch
```

It measures your GPU's cost model, profiles every benchmark on your GPU,
searches every GENGA drift and unified subgraph with Herbie, and removes every
earlier solve cache. It needs a Herbie that accepts `--num-taylor-terms` and
stops with the instructions if there is none. Build one of these:

```sh
scripts/docker.sh build --herbie                              # image poseidon-cgo2027:herbie; docker.sh then uses it
scripts/build_toolchain.sh --with-herbie --prefix $PWD/toolchain   # needs Racket 8.15 and Rust 1.85.1
```

Both build Herbie from the `sources/herbie` submodule with
`sources/herbie-num-taylor-terms.patch` applied. In the container run
`./run_artifact.sh --from-scratch` from `scripts/docker.sh shell`. A Herbie
search takes about 10 minutes and up to 13 GB of memory per solve. A complete
from-scratch run has not been timed; its decisions can differ from the
shipped ones where Herbie's randomized search or your profile changes the
candidates, and [`EXPECTED_RESULTS.md`](EXPECTED_RESULTS.md) (under "If you
change a default") lists what is still promised.

To regenerate one input without a GPU:

```sh
scripts/make_herbie_answers.sh --device gh200    # search a device's GENGA answers again; about 5 min and 13 GB per row
scripts/make_caches.sh --device all              # recompute the shipped solve caches from scratch
```

`make_herbie_answers.sh` writes `herbie_answers_<device>/` in the repository
root. Compare it with the shipped pack (`devices/<device>/herbie_answers/`, or
`benchmarks/genga_nbody/herbie_answers/` for the RTX 5090) with `diff -r`: the
question records are deterministic, and an answer's text carries Herbie's run
date and time. To check the row's decision with your answers, copy your pack
over the shipped one and run `scripts/check_picks.sh t1-drift-gh200`.

## 6. Use Poseidon on a kernel of your own

Mark the kernel, profile it once, compile it with an accuracy target:

```c++
#include "poseidon/poseidon.h"

POSEIDON_OPTIMIZE __global__ void filter(double *W, const double *H, ...);
```

```sh
poseidon-clang++ $CUDA_FLAGS -poseidon-profile-generate app.cu -o app_profiled
./app_profiled                 # runs a small problem; writes ./poseidon.profile
poseidon-clang++ $CUDA_FLAGS -poseidon-profile-use -poseidon-tau=1e-7 app.cu -o app
```

`poseidon-clang++` is on `PATH` in the container. The annotated kernel's whole
body is the optimization unit, and profile generation supplies its own shadow
buffers. Call `poseidon_metric("<name>", value)` once at the end of the
profiling run, and the same run measures each site's condition number against
that quantity. The manual is
`sources/Enzyme/enzyme/Poseidon/README.md` (in the container:
`/opt/poseidon/src/enzyme/Poseidon/README.md`): its "Use" section has every way
to opt in, and "What a compile decided" the remarks and `poseidon-explain`.

The benchmarks pass explicit directories, cost models and budgets, so their
command lines are longer. A site with no `-poseidon-tau` is solved against
`-poseidon-cost-budget=<share>`: the share of the site's profiled cost a
solution may add (negative: has to save). Every solve writes the shares of its
Pareto frontier to `budgets.txt` in its `-poseidon-cache` directory, and
prints the configuration it applied as a `[poseidon] pin: <value>` line that
`-poseidon-apply-rewrites=<value>` applies again.

## Troubleshooting

**`--check` without a GPU fails** with `detect.sh: error: nvidia-smi not found`.
That is expected: `--check` checks the GPU. Without a GPU use `--picks` and
`--explain`, which need none.

**A row of `--picks` is `DIFFERENT` or `COMPILE-FAILED`.** Read its
`picks_check/<row>.log` ([Keep the files](#keep-the-files)). Check that the
driver and the plugin come from one build tree (next item). The shipped
decisions were solved on x86-64; on another host architecture the shipped
caches are not used, and the accuracy model's arithmetic can round
differently in the last digits.

**The driver, the plugin and the profiler runtimes must come from the same
build tree.** `poseidon-clang++` takes the plugin and the runtimes from the
tree it runs from, and `scripts/detect.sh` resolves the driver from
`POSEIDON_BUILD`, so the default path is consistent. Two symptoms appear when
you introduce a second compiler tree by hand (a second checkout, a plugin
copied in, `POSEIDON_PLUGIN` set in the environment):

- An empty solve: the compile succeeds, the solve enumerates one budget or
  none, and the log reports every site as `uncovered` and skipped under
  `-poseidon-loose-coverage`. The profile was written by a different plugin,
  whose site keys do not match. Profile again with this compiler.
- `nvlink: Prototype doesn't match` for a profiler entry point: the plugin
  emits calls against one revision of the profiler runtime while the link takes
  another tree's archive.

`./run_artifact.sh --check` prints the driver and the plugin (with its md5) it
resolved.

**Speedups well below `Expected`.** A GPU that cannot sustain its boost
clock slows the FP64 baselines disproportionately and moves the ratios. Rerun
on an idle, cool GPU at stock clocks. R-ChFSI also moves with the host CPU:
part of its time is host-driven library calls (about 4% run to run).

**`run_artifact: refusing to start`** with a list of pids: a benchmark of this
repository is already running in `benchmarks/`. Two copies overwrite each
other's outputs and share the GPU. Stop the listed processes, or pass `--force`
if they belong to another copy you know about.

**`[skip] ...: SKIPPED, not failed.`** in GENGA's log: your run prices with a
measured cost model, for which no Herbie answers ship. See
[Another NVIDIA GPU](#another-nvidia-gpu).

**`clang++: warning: CUDA version is newer than the latest partially supported
version`** on every compile: your CUDA toolkit is newer than the clang knows.
It is harmless.

**`env.sh: no cost model for this device`** from a direct `make` or pipeline
script on a GPU other than compute capability 12.0: measure it first with
`./run_artifact.sh --bootstrap-only`.

**cuMpSGEMM's first use is slow.** Its PTX is compiled for your GPU by the
driver on first use. `-v poseidon-cuda-cache:/root/.nv` on `docker run` keeps
that result across containers.

## Reference

### Requirements

Hardware:

- For the GPU routes: an NVIDIA GPU of compute capability 7.5 or newer. The
  shipped cost model applies to compute capability 12.0 (RTX 5090 class);
  other GPUs measure their own.
- About 4 GB of free disk for generated inputs (ozp synthesizes about 3 GB of
  matrices), plus the image (13.9 GB) or the toolchain.

Software (all of it is in the container):

- CUDA 12.x or 13.x (cuBLAS, cuSOLVER), found through `CUDA_HOME` (module
  systems), `/usr/local/cuda*`, or `nvcc` on `PATH`. The paper used 12.9.
- The LLVM build the Poseidon plugin is built against (auto-detected from the
  build tree's CMakeCache; override with `CLANG`).
- CMake and Ninja to build the compiler. Herbie only to search for algebraic
  rewrites yourself (`GENGA_HERBIE=run`, `--from-scratch`).
- Python with numpy and matplotlib for the plots (auto-detected; override
  with `PY`).

### Build from source, by hand

LLVM is llvm-project (https://github.com/llvm/llvm-project) at commit
`9d618fc0243103263ebf9bdcd08ef25e871d9c78`, built from GitHub's source tarball
of that commit (sha256
`57eb91d0f644d74671300ea3e154fc22c9070aa87656844f0795c637693b4bdb`). The
results come from a compiler whose error-corrected (TCEC) GEMM dispatch
delegates to cuMpSGEMM, built as its own CMakeLists says (sm_80 code and PTX);
the shipped RTX 5090 cost model was measured against that build.
`scripts/build_toolchain.sh` runs these steps, with `P` the installation
directory and `CUDA` the toolkit:

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
./run_artifact.sh --check
```

- `CMAKE_CUDA_ARCHITECTURES=native` needs the GPU visible at configure time;
  without one give it explicitly (`120-real` for the RTX 5090 class).
- The image builds LLVM with Ubuntu's clang 18 and lld
  (`-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_USE_LINKER=lld`);
  any C++17 compiler works.
- git is needed because Enzyme's configure step requires the program; nothing
  is cloned.
- `POSEIDON_HERBIE_BINARY` names where a Herbie is, or will be put. By default
  the artifact replays the shipped answers and starts no Herbie, so the path
  may not exist. Leaving the option out makes CMake clone and build upstream
  Herbie (git, cargo and raco needed), which lacks the `--num-taylor-terms`
  option the GENGA search uses. To search yourself, build Herbie from
  `sources/herbie` with `sources/herbie-num-taylor-terms.patch` and the
  commands of `docker/Dockerfile.herbie` (Racket 8.15 and Rust 1.85.1), as
  `scripts/build_toolchain.sh --with-herbie` does, and install it at that path.

### Configuration

You edit nothing. `scripts/detect.sh` resolves every setting, in this order:

1. your environment or the `make` command line (`GPU_ID=0 ./run_artifact.sh`,
   `make CLANG=...`);
2. `local.env` in the repository root: optional pins for a machine you use
   repeatedly, written as guards (`GPU_ID="${GPU_ID:-1}"`); not shipped;
3. `cost_models/target.env`: written by the bootstrap when it measured your
   GPU (the measured cost model and your GPU's architecture);
4. auto-detection from `POSEIDON_BUILD` (required): the source tree its
   CMakeCache names (for the public header and the GEMM runtime sources the
   reference arms compile), the driver and `poseidon-calibrate` of that build,
   the plugin-matching `clang++` from its CMakeCache; CUDA from
   `CUDA_HOME`, `/usr/local/cuda*` or `PATH`; the GPU's architecture and name
   from `nvidia-smi`; `GPU_ID` from Slurm's `CUDA_VISIBLE_DEVICES` if set,
   else 0; the cost model as described next.

Variables: `POSEIDON_ROOT CLANG POSEIDON_CLANGXX POSEIDON_CALIBRATE CUDA
GPU_ARCH GPU_ID PY POSEIDON_INC OZAKI_RT TCEC_RT DIRECT_RT COST_MODEL`. `make`
targets and directly invoked pipeline scripts (`bash scripts/run_force.sh`)
resolve them identically, through `config.mk` and `env.sh`, which share
`detect.sh`.

### The cost model

The solver's decisions come from a per-device cost model: reciprocal
throughputs measured by `poseidon-calibrate`. The artifact never prices one
device's code with another device's model in a GPU run.

- On a compute capability 12.0 GPU the run uses the shipped
  `cost_models/cm_sm_120_RTX5090.csv`, measured on the RTX 5090 the paper ran
  on. The bootstrap checks it against your GPU (`poseidon-calibrate --check`)
  and never rewrites it. With it, every decision of `EXPECTED_RESULTS.md` is
  reproduced exactly.
- `./run_artifact.sh --calibrate` measures your GPU's own model instead (into
  `cost_models/measured/`, recorded in `cost_models/target.env`; delete that
  file to return to the shipped model). The decisions are then your card's own.
- On any other GPU there is no default: `run_artifact.sh` measures it first.
  The bootstrap passes the two inputs that belong to the artifact: the ozp
  surrogate profile the in-kernel arm enumerates its candidates from, and
  R-ChFSI's rectangular product shapes. A direct `make` or pipeline script on
  such a GPU stops with the command to run.
- `--picks` and `--explain` compile for the device of the row, with that
  device's saved model (`devices/devices.tsv`), on any machine.

GENGA prices with the device's model scaled uniformly by 1e6 (for the shipped
model `cost_models/x1e6/`, otherwise derived into
`benchmarks/genga_nbody/measured/`). A uniform scale leaves every decision
unchanged; the shipped Herbie answers were searched against the Herbie
platform generated from the scaled model.

### Choices

| Setting | Default | Alternative |
|---|---|---|
| cost model | the shipped RTX 5090 model on compute capability 12.0, else your GPU's measured model | `./run_artifact.sh --calibrate` (your GPU's own on any GPU) |
| Herbie (GENGA drift and unified) | `GENGA_HERBIE=replay`: the shipped answers, no Herbie process | `GENGA_HERBIE=run`: Herbie searches (`HERBIE=<binary>` optional); `HERBIE_ANSWERS=<dir>`: replay another answer pack |
| profiles | `PROFILES=fresh`: profiled on your GPU (GENGA drift and unified in replay mode: the shipped profiles) | `PROFILES=shipped`: every solve reads `benchmarks/*/profiles/` |
| solve caches | `--picks` and `--explain` start from `devices/<device>/caches/` | `--no-shipped-cache` (any mode) |

An answer is keyed by the device family of the cost model and by its Herbie
platform, so the shipped answers replay only with the shipped models. On
another device the drift and unified solves search (`GENGA_HERBIE=run`) or
replay an answer pack made for that device (`HERBIE_ANSWERS=<dir>`, a
directory with `kepler/` and `unified/`). A Herbie question also carries the
profiled value ranges and sensitivities, so in replay mode the GENGA drift and
unified solves read the shipped profiles the answers were searched for, and
each profiling step prints whether your fresh profile has the same ranges.
`GENGA_HERBIE=run` searches from the fresh profile. It needs a Herbie that
accepts `--num-taylor-terms` (the drift kernel's Newton-step rewrite is a
4-term series, and Herbie's own cap is 2 terms); with a Herbie that does not,
the compile stops naming the option. `GENGA_TAYLOR_TERMS=` (empty) then
searches with Herbie's own cap, which reproduces every GENGA row except the
ones that need the 4-term series (`EXPECTED_RESULTS.md` names them).

### Slurm notes

- Put the repository and the compiler build tree on a filesystem the compute
  nodes see (a scratch filesystem; `$HOME` is slow). Build the toolchain once
  per system, in a batch job.
- The job finds the compiler through `POSEIDON_BUILD`: export it before
  `sbatch` (which hands your shell's environment to the job) or pin it in
  `local.env`. Export `HERBIE_ANSWERS` or `GENGA_HERBIE` the same way.
- `module load cudatoolkit python` (the wrapper does this); CUDA is found
  through `CUDA_HOME`. Where the site supplies its toolchain another way (a
  Spack view, a CSCS uenv), whatever is already on `PATH` is used.
- Everything, the cost-model measurement included, runs inside the one
  allocation; binaries run directly on the allocated GPU (no per-step `srun`).
- Command-line options override the wrapper's `#SBATCH` directives, `-t`
  included. The default walltime is 12 h.
- Do not set `GPU_ID`: the wrapper adopts Slurm's `CUDA_VISIBLE_DEVICES`,
  which outranks a `local.env` pin left over from a workstation.
- One GPU of one node (`--gpus-per-node=1`) is the intended shape for
  `run_artifact.sbatch`. On a node with several GPUs,
  `slurm/run_artifact_parallel.sbatch` gives each benchmark a GPU of its own.
- The results come back as one `results_<GPU>_<arch>_<timestamp>.tar.gz` in the
  repository root.
- CSCS Alps (GH200, aarch64), uenv `prgenv-gnu/25.11:v1`: the uenv supplies
  gcc, CMake, Ninja and CUDA 12.9, but no MPFR (build it against the uenv's GMP
  and pass `-DCMAKE_PREFIX_PATH=<mpfr>` to the compiler's configure) and a
  python without numpy and matplotlib (make a venv and pin `PY` in
  `local.env`). Build LLVM with `-DLLVM_TARGETS_TO_BUILD="AArch64;NVPTX"`.
  Configure the compiler without cuMpSGEMM there (omit the two
  `POSEIDON_*CUMPSGEMM*` options): at the pinned commit its library does not
  link on that toolchain, and no GH200 decision of the paper uses the
  error-corrected dispatch. Unless you build Racket and Rust there, there is no
  Herbie: configure with `-DPOSEIDON_HERBIE_BINARY=<a path that does not exist>`
  and give the GENGA steps an answer pack (`HERBIE_ANSWERS`).

### Determinism

- With the shipped cost model and the shipped profiles every decision is
  deterministic; `--picks` reproduces all of them on any x86-64 machine with
  the compiler, without a GPU.
- The solve caches (`cache*/table.json`, `cache*/budgets.txt`) regenerate per
  run. `--picks` and `--explain` seed them from `devices/<device>/caches/`;
  every entry stores the key of everything it was computed from, and a compile
  whose inputs differ says so and solves itself (`scripts/caches.sh`).
  `budgets.txt` is written only if absent; the pipeline scripts delete it
  whenever the profile or the cost model changes.
- Profiles: value ranges and execution counts are deterministic; sums of
  products and condition numbers (`Kappa`) are accumulated with atomics and
  differ from run to run in the last digits, which no decision is close to.
- The reduction trip counts the R-ChFSI solve needs travel in the `.fpprofile`
  header (`RedTrip = <slot> <trips>`), so the profile directory is the only
  thing `make solve` needs from `make profile`.
- ozp's FP64 and direct tensor-core trajectories use atomic reductions and
  show about 0.3% run-to-run accuracy noise. The Ozaki-II residue GEMM is exact
  in INT32, but the purification around it also reduces with atomics, so an
  Ozaki-II configuration's deployed RMSD reproduces to about 0.1%, not bitwise.

### Layout

```text
Poseidon-CGO-2027/
  run_artifact.sh      the driver: --check, --picks, --explain, --quick, full run (-h lists every mode)
  EXPECTED_RESULTS.md  every expected decision and number, with tolerances
  expected/            picks.tsv, picks_b300.tsv, picks_gh200.tsv (the decisions, by the
                       compiler's own names); claims.tsv (every checked number, its tolerance
                       and source); EXPECTED_RESULTS.md's tables are generated from them
  cost_models/         cm_sm_120_RTX5090.csv + .herbie.rkt: the RTX 5090 model; x1e6/: the same
                       in 1e6 units, as GENGA prices with it; measured/ + target.env: written
                       when your GPU is measured
  devices/             devices.tsv: the devices --picks checks; b300/, gh200/: their saved cost
                       models and Herbie answers; <device>/caches/: the shipped solve caches
  benchmarks/
    fig2_drift/        Fig. 1: FP64 vs FP32 vs 3xFP32 leapfrog drift
    genga_nbody/       GENGA force and drift kernels (elementwise)
    ozp/               density-matrix purification GEMM (Ozaki-II)
    chase_eigensolver/ R-ChFSI eigensolver (sensitivity-guided)
  scripts/
    detect.sh          configuration resolver (all auto-detection lives here)
    bootstrap_cost_model.sh  the cost model of your GPU (check or measure)
    build_toolchain.sh LLVM, cuMpSGEMM, the compiler and optionally Herbie, resumable
    build_cumpsgemm.sh its cuMpSGEMM step alone
    docker.sh          build|run|picks|explain|shell in the container
    check_picks.sh     compile-only check of the decisions (run_artifact.sh --picks)
    explain.sh         one row's candidates (run_artifact.sh --explain)
    caches.sh, make_caches.sh  the shipped solve caches: seeding, regeneration
    make_herbie_answers.sh     search a device's Herbie answers again
    genga_x1e6.sh      the x1e6 copy of a cost model GENGA prices with
    plot_all.sh        redraw every figure
    expected.sh, expected_results.py, expected_from_run.py  read expected/, regenerate
                       EXPECTED_RESULTS.md, read a run's numbers
  docker/              Dockerfile (the image), Dockerfile.herbie (adds Herbie)
  slurm/               run_artifact.sbatch, run_artifact_parallel.sbatch
  modal/               the paper's B300 rows on a B300 at Modal (modal/README.md)
  sources/             submodules: Enzyme (the compiler), cuMpSGEMM, herbie; the Herbie patch
  env.sh, config.mk    shell and make twins of the configuration; benchmarks include config.mk
  GEMM_RUNTIMES.md     the GEMM runtimes a rewritten program calls
  LICENSES.md          licences and provenance of every third-party part
  figures/             regenerated figure PDFs, named as in the paper
  local.env            optional pins (not shipped; see Configuration)
  results/             per-run result directories (created by run_artifact.sh)
```

Each benchmark directory holds the kernel sources, a `Makefile` that includes
`../../config.mk`, `scripts/` with the pipeline (`scripts/rows.sh` compiles one
decision row), `quick.sh` (smoke test) and `run_all.sh` (full reproduction),
`profiles/` (the profiles the expected decisions were made from),
`golden/rtx5090/` (the recorded RTX 5090 run `EXPECTED_RESULTS.md` cites),
`expected_results.md` (the pipeline step by step), and a plotting script that
writes into `../../figures/`. `genga_nbody/herbie_answers/` holds Herbie's
answers for the drift and unified kernels.

### Licences

The compiler is the Poseidon subproject of the Enzyme repository
(https://github.com/EnzymeAD/Enzyme, `enzyme/Poseidon`) at commit
`9e2e0572931248b4610940f2906e583914f9913e`. [`LICENSES.md`](LICENSES.md) lists
the licence and provenance of every third-party part. The licence of the
artifact's own files: **TODO (owner)**.
