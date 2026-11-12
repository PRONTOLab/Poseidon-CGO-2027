# Poseidon CGO 2027 artifact on a Modal B300

This directory measures the paper's B300 rows (Table 1: GENGA force and
drift, ozp at tau = 1e-7 and 1e-6, R-ChFSI) on one NVIDIA B300 at Modal. The
compiler's decisions for those rows were already made on a machine without a B300
with the saved B300 cost model (`bundle/replay_pack.tar.gz`); the B300 run
compiles each decided configuration by name, with no Herbie search and no
solver run on the B300 (the GENGA drift row reads Herbie answers searched on
that machine for the B300), and measures it against the artifact's
baselines. Six rows: the five Table 1 cells and `t1-ozp7-b300-paper`, the
paper's ozp configuration at tau = 1e-7 (Ozaki-II nm = 9) compiled by name
next to the compiler's current pick there (nm = 8).

Everything is in `poseidon_b300.py`, which runs the scripts in `remote/`
inside Modal containers. `make_bundle.sh` writes `bundle/`, what the app
uploads: the compiler source (`../sources/Enzyme`), the cuMpSGEMM source
(`../sources/cuMpSGEMM`), the artifact (the repository itself), the replay pack
(`replay_pack/`), and a `MANIFEST` with their checksums. The compiler is built
with the cuMpSGEMM error-corrected GEMM backend, as the paper's B300 runs
were. `solve_b300_locally.sh`, `b300_inputs/` (the saved B300 cost models) and
`test/` remake the replay pack on a machine without a B300; they are not
needed to run.

## Prerequisites

- Python 3.10 or newer with the Modal client: `pip install modal`
- A Modal account with access to B300 GPUs, set up once: `modal setup`
- This repository cloned with its submodules (`git clone --recursive`, or
  `git submodule update --init --recursive`), and the bundle made once:
  `./make_bundle.sh`. The app checks every tarball against `bundle/MANIFEST`
  and refuses to start if one is missing or differs.

## The commands, in order

Run them from this directory. Use one run id for all of them, so that every
result lands in one directory (any name; `b300-1` below).

```sh
# 0. Make the bundle from this clone (again after the submodules or the artifact change).
./make_bundle.sh

# 1. Build the compiler (CPU only, no GPU; compiles LLVM once, then cuMpSGEMM
#    for the B300 and Poseidon).
modal run --detach poseidon_b300.py --stage build

# 2. Check the toolchain on the B300 (minutes).
modal run poseidon_b300.py --stage smoke --run-id b300-1

# 3. Informational: measure this B300's cost model and compare it with the
#    saved one the decisions were made with (nothing uses the fresh model).
modal run poseidon_b300.py --stage calibrate --run-id b300-1

# 4. The B300 rows: compile the decided configurations and measure them.
#    No Herbie search and no solver run happen here; see "What FAILED means".
modal run --detach poseidon_b300.py --stage run --run-id b300-1

# 5. Fetch the results.
modal volume get poseidon-b300-results b300-1 .
```

`--stage all --run-id b300-1` runs 1 to 4 in one call (4 only if 2 passed).
`--rows "t1-ozp7-b300 t1-ozp6-b300"` limits step 4 to some rows.

## What each stage uses

| Stage | GPU | Container | Time limit |
|---|---|---|---|
| build | none | 32 CPU cores, 64 GiB | 8 h |
| smoke | 1 x B300 | 8 cores, 32 GiB | 30 min |
| calibrate | 1 x B300 | 8 cores, 32 GiB | 30 min |
| run | 1 x B300 | 16 cores, 64 GiB | 2 h |
| all | none itself (a coordinator that calls the four above) | | 24 h |

- Only smoke, calibrate and run hold a B300; build uses CPUs only.
- The build is stored in the Modal Volume `poseidon-b300-toolchain` and reused;
  a second build call finds it built. A new compiler bundle rebuilds only the
  Poseidon part. cuMpSGEMM compiles many kernel instances: 457 s with 32 jobs
  on a 192-core workstation.
- Results go to the Volume `poseidon-b300-results`. Delete both Volumes when
  done: `modal volume delete <name>`.
- Durations measured at Modal on 2026-11-12: build 26 min the first time
  (LLVM is most of it), 2 min when only the compiler bundle changed; smoke
  3 min; calibrate 4 min; run 17 min for the six rows.

To see or stop a running call: the Modal dashboard, or `modal app list` and
`modal app stop <app id>`.

## Results and how to send them back

```sh
modal volume get poseidon-b300-results b300-1 .
tar czf b300-1.tar.gz b300-1
```

and send `b300-1.tar.gz` back: the whole directory, also when a stage or a
row failed. It contains:

```
b300-1/
  smoke/                     the smoke stage's log, nvidia-smi output, a per-op cost model
  calibration_check/         fresh/cm_*.csv (this B300's model), compare_*.txt (per row family
                             new/saved, and every row that moved by more than 10%), calibrate.log
  replay/
    replay.tsv               per row: MEASURED / FAILED, the configuration, baseline and
                             optimized time, speedup, accuracy (RMSD, L2, energy drift, locked)
    replay.log               the whole stage's output
    rows/<row>/              compile.log, explanation files, run outputs, baseline outputs
    picks_b300.tsv           the decisions that were replayed
```

## How a row is timed

Every program is launched five times and the mean is reported; each launch is
kept in `replay/rows/<row>/runs.txt`. GENGA: the FP64 baseline before and
after the rewritten program. ozp: cuBLAS DGEMM, the FP64 baseline and the
row. R-ChFSI: one unrecorded launch of the baseline first (the first cuSOLVER
program of a job is slow), then two interleaved sets of five; the row is the
first set and the second shows the repeatability.

## Measured on 2026-11-12

One Modal B300 (NVIDIA B300 SXM6 AC, driver 610.57.04), this bundle:

| Row | Paper | Measured | Accuracy |
|---|---|---|---|
| `t1-force-b300` | 4.26x | 4.89x | L2 1.83e-13 |
| `t1-drift-b300` | 5.65x | 5.85x | energy drift 3.15e-8 |
| `t1-ozp7-b300` (nm = 8) | 16.94x (nm = 9) | 19.05x | RMSD 1.38e-8 |
| `t1-ozp7-b300-paper` (nm = 9) | 16.94x | 17.13x | RMSD 8.68e-10 |
| `t1-ozp6-b300` | 33.5x | 41.60x | RMSD 1.67e-7 |
| `t1-rchfsi-b300` | 2.18x | 2.14x to 2.33x | 64 of 64 locked, L2 3.01e-15 |

The R-ChFSI row depends on the host Modal assigns: two containers gave
314.7 -> 147.2 ms (2.14x) and 298.6 -> 128.4 ms (2.33x), each repeatable within
its container to 2%. The other rows repeated to 0.5% across containers.

## What FAILED means

A row is `FAILED` (and not measured) when its compile did not apply exactly
the configuration decided ahead of time, or when a solver or a
Herbie search would have run. The other rows are still measured, and the run
stage then ends with a non-zero exit status. Nothing needs fixing on your side:
send the whole results directory back; `replay/rows/<row>/compile.log` says
why.

## If the smoke stage fails

The smoke stage prints one `[smoke] OK` or `[smoke] FAILED` line per step and
saves its log under `b300-1/smoke/`. Do not start the run stage; send that
directory back. Steps: 1 nvidia-smi (no GPU, or not a B300), 2 the build's
CUDA architecture against the device (the build is for compute capability 10.3),
3 to 6 compile, profile, cost model and optimized compile of a small program
(a toolchain problem; the log names the failing command), 8 the artifact's
preflight. Step 7 (Herbie) is informational: the run stage does not use Herbie.

The build log is in the toolchain Volume:
`modal volume ls poseidon-b300-toolchain logs`, then
`modal volume get poseidon-b300-toolchain /logs/<file> .`.

## Not the default: the full on-device flow

`./make_bundle.sh --with-herbie`, `modal run poseidon_b300.py --stage build --with-herbie`, then
`modal run --detach poseidon_b300.py --stage full --run-id <id>` runs the
artifact's own pipeline on the B300 (measure the cost model, profile, solve,
Herbie search for GENGA), as `slurm/run_artifact.sbatch` would; results under
`<id>/full/`. It saves after each benchmark and resumes with the same id.
