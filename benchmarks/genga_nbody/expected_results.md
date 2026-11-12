# genga_nbody: GENGA force and drift kernels

Rows `genga-*` of [`../../EXPECTED_RESULTS.md`](../../EXPECTED_RESULTS.md)
hold the expected numbers of this benchmark.

GENGA is a production GPU N-body integrator. The benchmark drives two of its
kernels through Poseidon, one per elementwise rewrite technique:

- **Force kick** (`accEncFull`, dense O(N²) gravity, `genga_nbody.cu`):
  precision changes and floating-point expansions. An FMA reduction that
  precision rewrites accelerate.
- **Kepler drift** (`fastfg`, universal-variable step, `genga_kepler.cu`):
  Herbie's algebraic rewrites. Catastrophic cancellation (1 - cos ΔE,
  sin ΔE - ΔE) that algebra fixes more cheaply than extra bits.
- **Unified** force-then-drift (`genga_unified.cu`): one solve gives each half
  the technique it needs (a reported number; no figure).

Both kernels are ported from GENGA (Grimm and Stadel 2014,
bitbucket.org/sigrimm/genga, `directAcc.h` and `FG2.h`) as written, with one
change: `__dmul_rn` becomes `*` (Enzyme has no reverse-mode rule for it; the
difference is below one ulp, and the FP64 reference and the optimized kernels
get the same change).

## Run it

From this directory, with the configuration of the repository root
(`POSEIDON_BUILD` set, or in the container):

```sh
make quick        # smoke test: one force point and a 2000-step drift accuracy check
make run-all      # everything: both figures and every genga-* number
```

Or one step at a time:

```sh
make force force-mfoff                    # -> force_full_g0.tsv, force_mfoff.tsv
make kepler kepler-herbieoff timeseries   # -> kepler_full_g0.tsv, kepler_herbieoff_g0.tsv, kepler_timeseries.tsv
make unified                              # -> unified.tsv
make figures                              # -> ../../figures/genga_force_drift.pdf, genga_drift_timeseries.pdf
```

`make quick` ends with two lines to compare with the recorded RTX 5090 run:

```text
  force FP32: baseline=186.023026ms opt=3.502465ms speedup=53.1x L2=5.056976e-06  (expect ~51x, 5.056976e-06)
  drift @2000 steps: naive-FP32=1.901524e-05  Poseidon=1.296310e-08  (Poseidon should be ~100-1000x smaller)
```

Each frontier step (force, kepler, unified and both ablations) solves once at
budget 0, which writes the frontier's shares to `<cache>/budgets.txt`, then
compiles and measures one binary per frontier point. The unified step measures
a spanning subset, every point where the modelled error first drops tenfold,
and the point of row `genga-unified`. Every TSV row records the share and the
configuration the compiler applied (`pin`). The drift frontier is the longest
step of the whole artifact: about 120 points, each timed and integrated for
20k steps.

## Figures

- `../../figures/genga_force_drift.pdf` (Fig. 3), drawn by
  `genga_figures.py`: the force frontier and the expansion-off ablation (left),
  the drift frontier and the Herbie-off ablation (right).
- `../../figures/genga_drift_timeseries.pdf` (Fig. 4): orbital-energy drift
  against step count for FP64, the naive FP32 downcast and Poseidon.

`make figures` falls back to `golden/rtx5090/` for a step that has not run,
so it works without a GPU.

## Inputs it reads

- **Cost model:** the GPU's model scaled by 1e6 (`../../cost_models/x1e6/`
  for the shipped RTX 5090 model, else derived into `measured/`). A uniform
  scale leaves every decision unchanged.
- **Herbie:** the drift and unified solves replay `herbie_answers/<kernel>/`
  (`-poseidon-import-herbie-answers`; checksums in `herbie_answers/MD5SUMS`).
  These were searched with Herbie at the Taylor-term caps 2 and 4
  (`-poseidon-herbie-num-taylor-terms=2,4`) and solved exactly
  (`-poseidon-early-prune=false`). `GENGA_HERBIE=run` searches instead; the
  header of `scripts/common.sh` lists the settings.
- **Profiles:** `profiles/<kernel>/` are the profiles the answers were searched
  for and the expected decisions made from (`profiles/MD5SUMS`).
- **Recorded run:** `golden/rtx5090/` holds the RTX 5090 run that
  `EXPECTED_RESULTS.md` cites.

On a GPU priced with its own measured cost model, no shipped answers apply:
the drift frontier, the drift time series and the unified solve print
`[skip] ...: SKIPPED, not failed.` unless `GENGA_HERBIE=run` (with a Herbie
that accepts `--num-taylor-terms`) or `HERBIE_ANSWERS=<pack>` is set.
