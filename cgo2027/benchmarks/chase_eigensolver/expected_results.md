# chase_eigensolver: R-ChFSI (eval §5.3)

> The expected numbers, decisions and tolerances of this benchmark are rows
> `rchfsi-*` of [`../../EXPECTED_RESULTS.md`](../../EXPECTED_RESULTS.md), the one
> place they are written. This file describes what the benchmark does.

R-ChFSI (arXiv:2503.22652) is a dense Hermitian eigensolver that performs a
**residual-based Chebyshev-filtered subspace iteration**. `chase_real.cu` is a
faithful port run to convergence on a dense matrix with a **known planted
spectrum** (wanted `NEV` lowest eigenpairs in `[0.1, 1]`, a gap, the rest in
`[3, 30]`; deterministic `srand(12345)`), so the eigenvalue error is known exactly.

The solver's four matrix products, launched as separate kernels and each written
as a scalar reduction loop in a kernel carrying `POSEIDON_OPTIMIZE` (**five annotated sites**):

| site (kernel) | product | role | optimal precision |
|---|---|---|---|
| `hemm_body`  | Chebyshev filter `H·Z`  | dominant (~77–86% of matmul time) | **low** (washed out) |
| `resid_body` | residual `R = HV − VΛ`   | cancellation `HV ≈ VΛ` near convergence | FP64 |
| `rrhv_body`  | Rayleigh–Ritz `W = H·V`  | feeds the projection            | FP64 |
| `rrproj_body`| Rayleigh–Ritz `Vᵀ·W`     | determines the computed eigenvalues | FP64 |
| `gram_body`  | Gram `Vᵀ·V`              | basis orthogonality             | FP64 |

R-ChFSI's residual reformulation systematically suppresses the filter's rounding
error (it is applied to a correction term `Z ∝ ‖R‖ → 0`), so the **dominant** filter
tolerates low precision while the other three products need FP64. These are
**opposite** optimal strategies per site — the point the paper automates.

Following **SCVH**, we profile at `N = 512`, `NEV = 32` and deploy at `N = 2048`,
`NEV = 64`. The solve is **one joint DP** over all five sites under a shared cost
budget (`-poseidon-joint-dp`).

## Figure produced

- `../../figures/rchfsi_residuals.pdf` — `plot_chase.py`: the **per-eigenpair
  residual spectrum** at solver exit (the locking criterion; one value per wanted
  eigenpair) for scalar FP64, the four sensitivity-guided rows of
  `EXPECTED_RESULTS.md` and the sensitivity-blind solve at the knee's budget,
  against the locking tolerance. Lower is better. From fresh traces
  (`traces/eigresid_<row>.txt`) when a run left all of them, else from
  `golden/rtx5090/traces`.

## Pipeline

```bash
make quick                # profile @512 (with Kappa), the knee solve @2048, verify
make run-all              # ladder + blind ablation, SCVH sweep, Ozaki shape sweep, figure
# or per step:
make profile              # profile-gen @ N=512 NEV=32; writes each site's Kappa
make solve BUDGET=<share> # joint DP @ N=2048 NEV=64 (PROFDIR=profiles/surrogate512: the shipped profile)
make verify               # run it (eigenvalue L2 vs the planted spectrum, locked count)
bash scripts/run_ladder.sh      # guided ladder + blind ablation (avg-of-5) + traces
bash scripts/run_scvh.sh        # the configuration picked at N=2048 deployed to N=4096
bash scripts/ozaki_shape.sh     # Ozaki-II width crossover
make figures                    # -> ../../figures/rchfsi_residuals.pdf
```

- `BUDGET` is `-poseidon-cost-budget`, a share of the profiled cost. The ladder
  sweeps a sample of the frontier (`cache/budgets.txt`) and, on the shipped
  model, the frontier points of the rows of `EXPECTED_RESULTS.md`
  (`../../expected/picks.tsv`); `LADDER_BUDGETS` overrides it.
- The sensitivity-blind arm reads a copy of the same profile with its `Kappa`
  lines removed, so the two arms differ in nothing else. The ablation row
  (`ablation_knee.tsv`) compares both arms at the guided knee.
- Every row records the configuration the solver applied, one `<site>=<scheme>`
  per rewritten site, from the compile's pin lines.
- The native DGEMM candidate of a product whose extents are compile-time
  constants is priced from the row the cost model holds for exactly those
  extents (`ozaki_dispatch_rect_rel,dgemm_m112n112k2048` and
  `dgemm_m2048n112k2048` for the Gram and Rayleigh-Ritz products at the deploy
  size); the bootstrap measures them on a GPU it calibrates.
- `profiles/surrogate512/` is the profile the expected decisions were made from;
  `golden/rtx5090/` holds the RTX 5090 run `EXPECTED_RESULTS.md` cites (ladders,
  ablation and traces).
