# The B300 column of Table 1, measured on a B300 at Modal

The scripts in `modal/` measure the B300 column of Table 1 of the paper (GENGA
force and drift, ozp at tau = 1e-7 and tau = 1e-6, R-ChFSI) on one NVIDIA B300
rented from [Modal](https://modal.com). For every row of the B300 column, the
run stage (`./run_artifact.sh --table1` of the artifact) compiles the program
with the configuration that the compiler decided for the row, and measures the run time of the compiled program and of
the FP64 baseline. The decided configurations are those of
`expected/picks_b300.tsv`; `./run_artifact.sh --picks --device b300` checks the
same decisions without a GPU.

## Prerequisites

- A Modal account with access to NVIDIA B300 GPUs.
- Python 3.10 or newer with the Modal client (`pip install modal`) and a Modal
  login (`modal setup`).
- The repository cloned with the submodules (`git clone --recursive`).

All commands run in the directory `modal/`.

## Commands

```sh
# 0. Pack the sources for the upload into bundle/ (repeat after any change of the repository).
./make_bundle.sh
# 1. Build the toolchain on CPUs.
modal run --detach poseidon_b300.py --stage build
# 2. Verify the toolchain on a B300.
modal run poseidon_b300.py --stage verify --run-id b300-1
# 3. Optional: measure the cost model of the rented B300 and compare the measured cost model with the shipped B300 cost model.
modal run poseidon_b300.py --stage calibrate --run-id b300-1
# 4. Compile the decided configurations and measure the compiled programs.
modal run --detach poseidon_b300.py --stage run --run-id b300-1
# 5. Download the results.
modal volume get poseidon-b300-results b300-1 .
```

`--stage all --run-id b300-1` runs the stages 1 to 4 in one call.
`--stage full` runs `./run_artifact.sh` on the B300 and writes the results to
`<run-id>/full/`; the stage `full` needs a bundle and a build made with
`--with-herbie`.

The first build takes about 30 minutes. The stages verify, calibrate and run
take about 25 minutes together.

## Expected values

The run stage writes `b300-1/table1/table1.tsv`. The status column of every
row is expected to read `MEASURED`. Table 1 of the paper reports the following
values, measured on one B300 at Modal (NVIDIA B300 SXM6 AC):

| Row | Speedup | Accuracy |
|---|---|---|
| `t1-force-b300` | 4.85x | L2 1.83e-13 |
| `t1-drift-b300` | 5.63x | energy drift 4.28e-8 |
| `t1-ozp7-b300` | 19.04x | RMSD 1.38e-8 |
| `t1-ozp6-b300` | 61.3x | RMSD 1.67e-7 |
| `t1-rchfsi-b300` | 2.21x | 64 of 64 locked, L2 3.01e-15 |

The speedups vary with the host machine that Modal assigns to the run.
