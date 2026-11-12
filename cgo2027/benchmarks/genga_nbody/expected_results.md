# genga_nbody (eval §5.1)

> The expected numbers, decisions and tolerances of this benchmark are rows
> `genga-*` of [`../../EXPECTED_RESULTS.md`](../../EXPECTED_RESULTS.md), the one
> place they are written. This file describes what the benchmark does.

GENGA is a production GPU N-body integrator. We drive two of its verbatim
kernels through Poseidon, one per elementwise rewrite axis:

- **Force kick** (`accEncFull`, dense O(N²) gravity, `genga_nbody.cu`) — the
  **floating-point expansion / precision** axis. An FMA reduction that precision rewrites
  accelerate.
- **Kepler drift** (`fastfg`, universal-variable step, `genga_kepler.cu`) — the
  **Herbie algebraic-rewrite** axis. Catastrophic cancellation ($1-\cos\Delta E$,
  $\sin\Delta E-\Delta E$) that algebra fixes more cheaply than extra bits.
- **Unified** force-then-drift (`genga_unified.cu`) — one solve assigns each half
  the technique it needs (a reported number; no figure).

Both kernels are ported verbatim from GENGA (Grimm & Stadel 2014,
bitbucket.org/sigrimm/genga, `directAcc.h` + `FG2.h`). The one deviation is
`__dmul_rn -> *` (no Enzyme reverse-AD rule; sub-ulp, applied identically to the
FP64 gold and the optimized kernels).

## Figures produced

- `../../figures/genga_force_drift.pdf` — `fig_force_drift()`: force frontier +
  expansion-off ablation (left), drift frontier + Herbie-off ablation (right).
- `../../figures/genga_drift_timeseries.pdf` — `fig_drift_timeseries()`:
  orbital-energy drift vs step count for FP64 / naive-FP32 / Poseidon.

## Pipeline

```bash
make quick        # kick-the-tires: one force point + a 2k-step drift accuracy check
make run-all      # full §5.1 pipeline: both figures + all reported numbers
# or per phase:
make force force-mfoff                    # -> force_full_g0.tsv, force_mfoff.tsv
make kepler kepler-herbieoff timeseries   # -> kepler_*_g0.tsv, kepler_timeseries.tsv
make unified                              # -> unified.tsv
make figures                              # -> ../../figures/genga_{force_drift,drift_timeseries}.pdf
```

Each frontier step (force, kepler, unified and both ablations) solves once at
budget 0, which writes the frontier's shares to `<cache>/budgets.txt`, then
compiles and measures one binary per frontier point (unified: a spanning subset
plus every point where the modelled error first drops tenfold, plus the point of
row `genga-unified`). Every TSV row records the share and the configuration the
compiler applied (`pin`).

- Cost model: the device's model scaled x1e6 (`../../cost_models/x1e6/` for the
  shipped RTX 5090 model, else derived into `measured/`).
- Herbie: the drift and unified solves replay `herbie_answers/<kernel>/`
  (`-poseidon-import-herbie-answers`), searched with Herbie at the Taylor-term
  caps 2 and 4 (`-poseidon-herbie-num-taylor-terms=2,4`) and solved exactly
  (`-poseidon-early-prune=false`); `GENGA_HERBIE=run` searches instead
  (`scripts/common.sh`). The answers' checksums are in `herbie_answers/MD5SUMS`.
- Profiles: `profiles/<kernel>/` are the profiles the answers were searched
  for and the expected decisions made from (`profiles/MD5SUMS`).
- `golden/rtx5090/` holds the RTX 5090 run `EXPECTED_RESULTS.md` cites;
  `make figures` falls back to it, so the figures regenerate from the shipped
  artifact without a GPU.
