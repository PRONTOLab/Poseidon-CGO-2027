# fig2_drift: the paper's Fig. 1

Rows `fig2-*` of [`../../EXPECTED_RESULTS.md`](../../EXPECTED_RESULTS.md)
hold the expected numbers of this benchmark.

The introduction's figure (`intro_drift.pdf`). A long leapfrog integration of
a harmonic oscillator (k = 1, dt = 1/128) runs three ways (FP64, FP32, and a
three-component FP32 expansion, "3xFP32") and is compared with a 2xFP64
reference that uses the same dt, so all truncation error cancels and what
remains is accumulated rounding. Each variant is written by hand and timed on
the GPU and on one CPU thread; this benchmark does not run the Poseidon pass.
It shows why a floating-point expansion pays off on a GPU but not on a CPU.

## Run it

From this directory, with the configuration of the repository root
(`POSEIDON_BUILD` set, or in the container):

```sh
make          # build, run on GPU $(GPU_ID), redraw ../../figures/intro_drift.pdf
make run      # build and run only (writes drift.csv, run.log)
make plot     # redraw the figure from drift.csv only (no GPU)
```

The recorded RTX 5090 run printed (`golden/rtx5090/fig2_drift.log` has the
whole output):

```text
GPU FP64 (as written)          5.471 ms     1.00x
GPU 3xFP32 emulation           2.620 ms     2.09x
...
  final drift @  1048576 steps: fp32=1.07e-05  fp64=2.20e-14  3xfp32=1.61e-20
```

The CPU timings run on one thread and move with the load of the machine; the
row `fig2-gpu` gives them a 15% tolerance.

## Reference data

`drift.csv.reference` is the paper's `drift.csv`: the mean drift over 4096
oscillators at power-of-2 step checkpoints. A run's `drift.csv` must match its
drift columns bit for bit, because the integration is deterministic and
independent of the device:

```sh
cmp drift.csv drift.csv.reference
```

Checkpoints are stored as hi+lo double-double pairs so that the difference
between 3xFP32 and the reference, which is below one double ulp, survives.
`dt = 1/128` is exact in FP32, so `dt` stays a single float component; an
inexact `dt` would need a three-component multiply and add a systematic
frequency error.
