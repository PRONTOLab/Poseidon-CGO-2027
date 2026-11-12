# chase_eigensolver: R-ChFSI

Judge your numbers against rows `rchfsi-*` of
[`../../EXPECTED_RESULTS.md`](../../EXPECTED_RESULTS.md).

R-ChFSI (arXiv:2503.22652) is a dense Hermitian eigensolver that performs a
residual-based Chebyshev-filtered subspace iteration. `chase_real.cu` ports it
and runs it to convergence on a dense matrix with a known planted spectrum
(the wanted `NEV` lowest eigenpairs in `[0.1, 1]`, a gap, the rest in
`[3, 30]`; deterministic `srand(12345)`), so the eigenvalue error is known
exactly.

Each matrix product of the solver is its own kernel, written as a scalar
reduction loop and marked `POSEIDON_OPTIMIZE`: five annotated sites.

| Site (kernel) | Product | Role | Precision it needs |
|---|---|---|---|
| `hemm_body` | Chebyshev filter `H·Z` | dominant (about 77 to 86% of the product time) | low (its error is washed out) |
| `resid_body` | residual `R = HV - VΛ` | cancellation `HV ≈ VΛ` near convergence | FP64 |
| `rrhv_body` | Rayleigh-Ritz `W = H·V` | feeds the projection | FP64 |
| `rrproj_body` | Rayleigh-Ritz `Vᵀ·W` | determines the computed eigenvalues | FP64 |
| `gram_body` | Gram `Vᵀ·V` | basis orthogonality | FP64 |

R-ChFSI's residual reformulation suppresses the filter's rounding error (the
filter is applied to a correction term `Z ∝ ‖R‖ → 0`), so the dominant filter
tolerates low precision while the other products need FP64. Poseidon finds
these opposite strategies from the profile: the profiling run measures each
site's condition number (`Kappa`), and the filter's comes out near 1e4
against near 1e12 for the others (recorded run: 9.0e3 for `hemm`, 6.7e11 to
8.3e12 for the rest).

The benchmark profiles at `N = 512`, `NEV = 32` and deploys at `N = 2048`,
`NEV = 64`. The solve is one joint DP over all five sites under a shared cost
budget (`-poseidon-joint-dp`).

## Run it

From this directory, with the configuration of the repository root
(`POSEIDON_BUILD` set, or in the container):

```sh
make quick                      # profile at N = 512 (with Kappa), the rchfsi-knee solve at N = 2048, verify
make run-all                    # the sensitivity-guided sweep, the sensitivity-blind ablation, the figure
```

Or one step at a time:

```sh
make profile                    # profile at N = 512, NEV = 32; writes each site's Kappa into fpprofile/
make probe                      # print the Kappa line of every site
make solve BUDGET=<share>       # joint DP at N = 2048, NEV = 64 (PROFDIR=profiles/surrogate512: the shipped profile)
make verify                     # run it: eigenvalue L2 against the planted spectrum, locked count
bash scripts/run_ladder.sh      # both arms of the sweep (average of 5 runs each) and the traces
make figures                    # -> ../../figures/rchfsi_residuals.pdf
```

`make quick` ends with the converged solve. The recorded RTX 5090 run printed:

```text
  applied: hemm=wmma m16n16k16 bf16/f32;rrhv=ozaki-ii nm=12 wmma m16n16k16 s8/s32
[expected] rchfsi-knee: the compiler applied the configuration of EXPECTED_RESULTS.md
...
  iter  8: locked=64/64  max-resid=1.207e-11 ritz[0]=0.100000
OPT eigval L2 vs planted = 1.230509e-11 (locked 64/64)
OPT solve_ms = 81.987 (iters=9)
```

What `scripts/run_ladder.sh` does:

- It sweeps a sample of the Pareto frontier (`cache/budgets.txt`) and, on the
  shipped model, the frontier points of the R-ChFSI rows of
  `EXPECTED_RESULTS.md` (`../../expected/picks.tsv`); `LADDER_BUDGETS`
  replaces the sample.
- The sensitivity-blind arm reads a copy of the same profile with its `Kappa`
  lines removed, so the two arms differ in nothing else. `ablation_knee.tsv`
  compares both arms at the share of row `rchfsi-knee`.
- Every row records the configuration the solver applied, one
  `<site>=<scheme>` per rewritten site, from the compile's pin lines.
- Outputs: `scvh_probe_on.tsv` (guided), `scvh_probe_off.tsv` (blind),
  `ablation_knee.tsv`, `chase_money_data.tsv` (the FP64 baseline) and
  `traces/eigresid_*.txt`.

The native DGEMM candidate of a product whose extents are compile-time
constants is priced from the cost-model row for exactly those extents
(`ozaki_dispatch_rect_rel,dgemm_m112n112k2048` and `dgemm_m2048n112k2048` for
the Gram and Rayleigh-Ritz products at the deploy size); the bootstrap
measures them on a GPU it calibrates.

## Figure

`../../figures/rchfsi_residuals.pdf` (Fig. 6), drawn by `plot_chase.py`: the
residual of every wanted eigenpair at solver exit (the locking criterion) for
scalar FP64, the four sensitivity-guided rows of `EXPECTED_RESULTS.md` and the
sensitivity-blind solve at the share of `rchfsi-knee`, against the locking
tolerance. Lower is better. It draws from fresh traces
(`traces/eigresid_<row>.txt`) when a run left all of them, else from
`golden/rtx5090/traces`, so it works without a GPU.

## Inputs it reads

- `profiles/surrogate512/`: the profile the expected decisions were made from.
- `golden/rtx5090/`: the RTX 5090 run that `EXPECTED_RESULTS.md` cites
  (sweeps, ablation and traces).
