# figures/

Regenerated figures land here, named as in the paper. `scripts/plot_all.sh`,
run from the repository root, redraws all of them (each script falls back to
the recorded RTX 5090 run, so it works without a GPU); `make figures` in one
benchmark redraws that benchmark's.

- `intro_drift.pdf` (Fig. 1): `benchmarks/fig2_drift/plot_drift.py`
- `genga_force_drift.pdf` (Fig. 3), `genga_drift_timeseries.pdf` (Fig. 4): `benchmarks/genga_nbody/genga_figures.py`
- `ozp_convergence.pdf` (Fig. 5): `benchmarks/ozp/plot_ozp.py`
- `ozp_surrogate.pdf`: `benchmarks/ozp/plot_surrogate.py`
- `rchfsi_residuals.pdf` (Fig. 6): `benchmarks/chase_eigensolver/plot_chase.py`
