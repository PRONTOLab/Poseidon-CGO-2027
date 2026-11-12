# The paper's B300 rows on a B300 at Modal

Use this route if you have a Modal account with access to NVIDIA B300 GPUs. It
measures the B300 column of the paper's Table 1 (GENGA force and drift, ozp at
tau = 1e-7 and 1e-6, R-ChFSI) on one B300, in about 50 minutes of stages.

The decisions are the ones `./run_artifact.sh --picks --device b300` checks
on any machine without a GPU: the compiler decided each row with the saved
B300 cost model, and `replay_pack/` holds those decisions. On the B300, each
row is compiled by its configuration's name, with no solver run and no Herbie
search (the GENGA drift row reads the Herbie answers searched for the saved
B300 model), and measured against the artifact's baselines. You get six rows:
the five Table 1 cells and `t1-ozp7-b300-paper`, the paper's ozp configuration
at tau = 1e-7 (Ozaki-II `nm = 9`) compiled by name next to this compiler's
decision there (`nm = 8`).

## Before you start

- Python 3.10 or newer with the Modal client: `pip install modal`.
- Log in to Modal once: `modal setup`.
- The repository cloned with its submodules (`git clone --recursive`, or
  `git submodule update --init --recursive` in a clone).

Run every command below from this directory.

## Run it

Use one run id for all stages, so that every result lands in one directory
(any name; `b300-1` below).

```sh
# 0. Pack what the app uploads into bundle/ (again whenever the submodules or the
#    repository change). The app checks every file against bundle/MANIFEST and
#    refuses to start if one is missing or differs.
./make_bundle.sh

# 1. Build the toolchain on CPUs (LLVM once, then cuMpSGEMM for the B300 and the compiler).
modal run --detach poseidon_b300.py --stage build

# 2. Check the toolchain on a B300.
modal run poseidon_b300.py --stage smoke --run-id b300-1

# 3. Optional: measure this B300's cost model and compare it with the saved one
#    the decisions were made with. Nothing uses the fresh model.
modal run poseidon_b300.py --stage calibrate --run-id b300-1

# 4. Compile the decided configurations and measure them.
modal run --detach poseidon_b300.py --stage run --run-id b300-1

# 5. Fetch the results.
modal volume get poseidon-b300-results b300-1 .
```

`--stage all --run-id b300-1` runs stages 1 to 4 in one call (stage 4 only if
stage 2 passed). `--rows "t1-ozp7-b300 t1-ozp6-b300"` limits stage 4 to some
rows.

Durations measured at Modal on 2026-11-12: build 26 minutes the first time
(LLVM is most of it), 2 minutes when only the compiler changed; smoke
3 minutes; calibrate 4 minutes; run 17 minutes for the six rows.

| Stage | GPU | Container | Time limit |
|---|---|---|---|
| build | none | 32 CPU cores, 64 GiB | 8 h |
| smoke | 1 x B300 | 8 cores, 32 GiB | 30 min |
| calibrate | 1 x B300 | 8 cores, 32 GiB | 30 min |
| run | 1 x B300 | 16 cores, 64 GiB | 2 h |
| all | none itself (a coordinator that calls the four above) | | 24 h |

- Only smoke, calibrate and run hold a B300; build uses CPUs only. The
  compiler is built with the cuMpSGEMM error-corrected GEMM backend, as the
  paper's B300 runs were, and with CUDA 13.1, which the B300 needs.
- The build is stored in the Modal Volume `poseidon-b300-toolchain` and
  reused; a second build call finds it built, and a new compiler bundle
  rebuilds only the compiler.
- Results go to the Volume `poseidon-b300-results`. Delete both Volumes when
  you are done: `modal volume delete <name>`.
- To see or stop a running call: the Modal dashboard, or `modal app list` and
  `modal app stop <app id>`.

## What you get

```text
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

## Compare with the measured values

Open `b300-1/replay/replay.tsv`. Every row should be `MEASURED`. One Modal
B300 (NVIDIA B300 SXM6 AC, driver 610.57.04) gave, on 2026-11-12:

| Row | Paper | Measured | Accuracy |
|---|---|---|---|
| `t1-force-b300` | 4.26x | 4.89x | L2 1.83e-13 |
| `t1-drift-b300` | 5.65x | 5.85x | energy drift 3.15e-8 |
| `t1-ozp7-b300` (nm = 8) | 16.94x (nm = 9) | 19.05x | RMSD 1.38e-8 |
| `t1-ozp7-b300-paper` (nm = 9) | 16.94x | 17.13x | RMSD 8.68e-10 |
| `t1-ozp6-b300` | 33.5x | 41.60x | RMSD 1.67e-7 |
| `t1-rchfsi-b300` | 2.18x | 2.14x to 2.33x | 64 of 64 locked, L2 3.01e-15 |

Expect the accuracy values to match and the speedups of the GENGA and ozp rows
to land within a few percent: they repeated to 0.5% across containers. The
R-ChFSI row depends on the host Modal assigns: part of its time is host-driven
library calls, and two containers gave 314.7 -> 147.2 ms (2.14x) and
298.6 -> 128.4 ms (2.33x), each repeatable within its container to 2%.

How a row is timed: every program is launched five times and the mean is
reported; each launch is kept in `replay/rows/<row>/runs.txt`. GENGA: the FP64
baseline before and after the rewritten program. ozp: cuBLAS DGEMM, the FP64
baseline and the row. R-ChFSI: one unrecorded launch of the baseline first
(the first cuSOLVER program of a job is slow), then two interleaved sets of
five; the row is the first set and the second shows the repeatability.

## If something fails

**A row is `FAILED`** (and not measured) when its compile did not apply
exactly the decided configuration, or when a solver run or a Herbie search
would have been needed. The other rows are still measured, and the run stage
ends with a non-zero exit status. `replay/rows/<row>/compile.log` says why.

**The smoke stage** prints one `[smoke] OK` or `[smoke] FAILED` line per step
and saves its log under `b300-1/smoke/`. Do not start the run stage after a
failure. Steps: 1 nvidia-smi (no GPU, or not a B300); 2 the build's CUDA
architecture against the device (the build is for compute capability 10.3);
3 to 6 compile, profile, cost model and optimized compile of a small program
(a toolchain problem; the log names the failing command); 8 the artifact's
preflight. Step 7 (Herbie) is informational: the run stage does not use
Herbie.

**The build's log** is in the toolchain Volume:
`modal volume ls poseidon-b300-toolchain logs`, then
`modal volume get poseidon-b300-toolchain /logs/<file> .`.

## The full pipeline on the B300

To measure the B300's own cost model, profile, solve and search with Herbie
on the device, as `slurm/run_artifact.sbatch` would on a cluster:

```sh
./make_bundle.sh --with-herbie
modal run poseidon_b300.py --stage build --with-herbie
modal run --detach poseidon_b300.py --stage full --run-id <id>
```

Its results land under `<id>/full/`. It saves after each benchmark and resumes
with the same id. Its decisions are this B300's own; compare them with the
table above qualitatively. There is no recorded run of this flow, so no
duration is given.

## Files here

- `poseidon_b300.py`: the Modal app; it runs the scripts in `remote/` inside
  Modal containers.
- `make_bundle.sh`: writes `bundle/` from this clone: the compiler source
  (`../sources/Enzyme`), the cuMpSGEMM source (`../sources/cuMpSGEMM`), the
  repository itself, the replay pack (`replay_pack/`) and a `MANIFEST` with
  their checksums.
- `solve_b300_locally.sh`, `b300_inputs/` (the saved B300 cost models) and
  `test/`: remake the replay pack on a machine without a B300. You do not need
  them to run.
