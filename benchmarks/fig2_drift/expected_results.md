# fig2_drift (paper Fig. 1)

> The expected numbers and tolerances of this benchmark are rows `fig2-*` of
> [`../../EXPECTED_RESULTS.md`](../../EXPECTED_RESULTS.md), the one place they
> are written. This file describes what the benchmark does.

Intro motivating figure (paper Fig. 2, `img/intro_drift.pdf`). A long leapfrog
integration of a harmonic oscillator (k = 1, dt = 1/128) run three ways and
compared against a 2xFP64 reference using the **same** dt, so all truncation
error cancels and the residual is pure accumulated rounding.

This benchmark does not invoke the Poseidon pass. It motivates the MultiFloat
emulation axis by measuring both the accuracy and the wall-clock cost of each
hand-written variant.

## Reference data

`drift.csv.reference` is the paper's `drift.csv` (mean drift over 4096
oscillators at power-of-2 step checkpoints). A fresh `make run` regenerates
`drift.csv`; the drift columns match it bit for bit: the integration is
deterministic and device independent. Checkpoints are stored as hi+lo
double-double pairs so the sub-double-ulp 3xFP32-vs-reference difference
survives. `dt = 1/128` is FP32-exact, so `dt` stays a single float component;
an inexact `dt` would force a three-component multiply and a systematic
frequency error.

## Reproduce

```bash
make          # build + run on GPU $(GPU_ID) + regenerate ../../figures/intro_drift.pdf
make run      # just build + run (writes drift.csv, run.log)
make plot     # just regenerate the figure from drift.csv
```
