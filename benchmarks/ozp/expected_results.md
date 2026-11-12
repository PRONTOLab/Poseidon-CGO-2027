# ozp (eval §5.2)

> The expected numbers, decisions and tolerances of this benchmark are rows
> `ozp-*` of [`../../EXPECTED_RESULTS.md`](../../EXPECTED_RESULTS.md), the one
> place they are written. This file describes what the benchmark does.

ozp (Dawson JCTC'24) is a GPU quantum-chemistry application that computes a
system's electronic structure by **density-matrix purification**, a fixed-point
iteration of dense matrix products. We drive its TRS4 iteration
(`purify_imp.h`) through Poseidon: the iteration's dense FP64 GEMM, invoked as
`X·X` and as `X²·G` for a trace-resetting polynomial `G(X)`, is extracted as **one
FP subgraph**. Poseidon proposes direct tensor-core raises (FP16/TF32/BF16 inputs,
FP32 accumulators), an FP32-class error-corrected product (FP16TCEC, in-kernel and
host-dispatched), and the Ozaki-II INT8 residue scheme at every modulus count
`nm = 8 … 14` (more moduli → more accuracy at proportional cost), and selects per
a **call-site accuracy target** `tau`.

Inputs are synthetic `N×N` gapped Hamiltonians `H = Q diag(D) Qᵀ` built from a
known spectral decomposition (`gen_mtx.py`, deterministic `default_rng(12345)`,
gap 0.10, half filling), so the exact projector `K_ref` is known by construction.
Accuracy is the density-matrix **RMSD** of the computed `X` vs `K_ref`. Following
Dawson JCTC'24: scientifically-meaningful threshold RMSD ≤ `1e-7`, near-FP64
threshold `2^-37 ≈ 7.28e-12`.

Following **SCVH**, we profile at `N = 512` and deploy at `N = 2048`.

## Figures produced

- `../../figures/ozp_convergence.pdf` — `plot_ozp.py`: **Left** per-iteration RMSD
  (direct TC raises stall above the chemistry bar; the Ozaki-II rungs `nm = 9`,
  `13`, `14` bracket both bars); **Right** speed/accuracy Pareto frontier
  (emulation ON solid / OFF dashed), y = speedup over naive scalar-FP64 GEMM,
  cuBLAS DGEMM as an FP64 reference, the `tau` auto-picks (`autopick.tsv`) circled.
- `../../figures/ozp_surrogate.pdf` — `plot_surrogate.py`: **Left** the Ozaki-II
  rungs `nm = 8, 9, 10` against the chemistry bar at every deploy `N`; **Right**
  speedup vs scalar FP64 grows with `N`.

## Pipeline

```bash
make quick        # kick-the-tires: profile @512, the tau = 1e-7 auto-pick @2048
make run-all      # full §5.2 pipeline (data, ladder, frontier, auto-picks, surrogate, figures)
# or per phase:
make data                 # synthetic inputs (deterministic; regenerated, not shipped)
make pareto calibrate     # -> pareto.tsv, traj_M0_*.tsv
make dispatch             # -> dispatch_rungs.tsv, traj_OZII_*.tsv
make direct-tc-lib        # -> direct_tc_lib.tsv, traj_LIB_*.tsv
make autopick             # -> autopick.tsv
make surrogate            # -> surrogate.tsv
make figures              # -> ../../figures/ozp_convergence.pdf, ozp_surrogate.pdf
```

- The in-kernel ladder (`run_pareto.sh`) applies each candidate by the
  compiler's name for it (`-poseidon-apply-rewrites=M0=<name>`); the tags `M0_1`
  ... `M0_6` only name the output files: FP16, BF16, TF32 direct raises and
  FP16TCEC with 2, 3, 5 correction terms.
- The frontier (`run_dispatch_rungs.sh`) solves once at budget 0 and compiles one
  binary per frontier share in `cache/budgets.txt`; on the RTX 5090 it holds the
  direct library dispatches (cuBLAS FP16/BF16/TF32 GEMM) and the Ozaki-II rungs
  `nm = 8 ... 14`.
- The auto-picks (`run_autopick.sh`) compile at `-poseidon-tau=1e-6, 1e-7,
  1e-12`; the solver compares tau against its modelled domain error
  (`Poseidon error-budget: matmul -> ...` in the compile log, with the error).
- `profiles/surrogate512/` is the `N = 512` profile the expected decisions were
  made from (`PROFILES=shipped` solves from it).
- `golden/rtx5090/` holds the RTX 5090 run `EXPECTED_RESULTS.md` cites;
  `make figures` falls back to it.
- `GEMM_RUNTIMES.md` (artifact root) documents the Ozaki-II, TCEC and direct GEMM runtimes.
