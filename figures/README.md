# figures/

Your regenerated figures land here, named as in the paper. Run
`scripts/plot_all.sh` from the repository root (no GPU needed: each script
falls back to the recorded RTX 5090 run), or `make figures` in one benchmark.

- `intro_drift.pdf` (Fig. 1): `benchmarks/fig2_drift/plot_drift.py`
- `genga_force_drift.pdf` (Fig. 3), `genga_drift_timeseries.pdf` (Fig. 4): `benchmarks/genga_nbody/genga_figures.py`
- `ozp_convergence.pdf` (Fig. 5): `benchmarks/ozp/plot_ozp.py`
- `ozp_surrogate.pdf`: `benchmarks/ozp/plot_surrogate.py`
- `rchfsi_residuals.pdf` (Fig. 6): `benchmarks/chase_eigensolver/plot_chase.py`
