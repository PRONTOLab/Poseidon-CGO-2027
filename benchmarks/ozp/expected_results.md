# ozp: density-matrix purification

Judge your numbers against rows `ozp-*` of
[`../../EXPECTED_RESULTS.md`](../../EXPECTED_RESULTS.md).

ozp (Dawson, JCTC 2024) is a GPU quantum-chemistry application that computes
a system's electronic structure by density-matrix purification, a fixed-point
iteration of dense matrix products. The benchmark drives its TRS4 iteration
(`purify_imp.h`) through Poseidon: the iteration's dense FP64 GEMM, invoked as
`X·X` and as `X²·G` for a trace-resetting polynomial `G(X)`, is one FP
subgraph. Poseidon proposes direct tensor-core raises (FP16, TF32 and BF16
inputs, FP32 accumulators), an FP32-class error-corrected product (FP16TCEC,
in-kernel and host-dispatched), and the Ozaki-II INT8 residue scheme with
`nm = 8` to `14` moduli (more moduli: more accuracy at proportional cost), and
picks one per call-site accuracy target `tau`.

The inputs are synthetic `N×N` gapped Hamiltonians `H = Q diag(D) Qᵀ` built
from a known spectral decomposition (`gen_mtx.py`, deterministic
`default_rng(12345)`, gap 0.10, half filling), so the exact projector `K_ref`
is known by construction. Accuracy is the RMSD of the computed density matrix
`X` against `K_ref`. As in Dawson's paper, the scientifically meaningful
threshold is RMSD ≤ `1e-7` and the near-FP64 threshold `2^-37 ≈ 7.28e-12`.
The benchmark profiles at `N = 512` and deploys at `N = 2048`.

## Run it

From this directory, with the configuration of the repository root
(`POSEIDON_BUILD` set, or in the container):

```sh
make quick        # smoke test: profile at N = 512, the tau = 1e-7 decision at N = 2048, measured
make run-all      # everything: inputs, in-kernel sweep, frontier, decisions, surrogate, figures
```

Or one step at a time:

```sh
make data                 # synthetic inputs (deterministic; about 3 GB, generated, not shipped)
make pareto               # -> pareto.tsv, traj_M0_*.tsv
make dispatch             # -> dispatch_rungs.tsv, traj_OZII_*.tsv
make direct-tc-lib        # -> direct_tc_lib.tsv, traj_LIB_*.tsv
make autopick             # -> autopick.tsv
make surrogate            # -> surrogate.tsv  (SURROGATE_SIZES="1024 2048 4096" skips the FP64-bound N = 8192)
make figures              # -> ../../figures/ozp_convergence.pdf, ozp_surrogate.pdf
```

`make quick` ends with the decision for `tau = 1e-7`, measured. The recorded
RTX 5090 run printed:

```text
  cuBLAS DGEMM: 662.855 ms   scalar FP64: 1152.486694 ms
  tau      compile-time pick                        RMSD          time       x FP64    x cuBLAS
[expected] ozp-tau1e-7: the compiler applied the configuration of EXPECTED_RESULTS.md
  1e-7     ozaki-ii nm=8 wmma m16n16k16 s8/s32      1.375121e-08  42.223392ms 27.295x   15.699x
```

The quick test times a single tau; `make autopick` measures the three rows
`ozp-tau1e-6`, `ozp-tau1e-7` and `ozp-tau1e-12`.

What each step does:

- `make pareto` (`scripts/run_pareto.sh`) applies each in-kernel candidate by
  the compiler's name for it (`-poseidon-apply-rewrites=M0=<name>`). The tags
  `M0_1` to `M0_6` only name the output files: FP16, BF16 and TF32 direct
  raises, and FP16TCEC with 2, 3 and 5 correction terms.
- `make dispatch` (`scripts/run_dispatch_rungs.sh`) solves once at budget 0
  and compiles one binary per frontier share in `cache/budgets.txt`. On the
  RTX 5090 the frontier holds the direct library dispatches (cuBLAS FP16, BF16
  and TF32 GEMM) and the Ozaki-II configurations `nm = 8` to `14`.
- `make autopick` (`scripts/run_autopick.sh`) compiles at
  `-poseidon-tau=1e-6`, `1e-7` and `1e-12`. The solver compares tau with its
  modelled domain error; the compile log prints it on the line
  `Poseidon error-budget: matmul -> ...`, and
  `./run_artifact.sh --explain ozp-tau1e-7` shows it without a GPU.

## Figures

- `../../figures/ozp_convergence.pdf` (Fig. 5), drawn by `plot_ozp.py`. Left:
  the RMSD per iteration (the direct tensor-core raises stall above the
  chemistry threshold; Ozaki-II with `nm = 9`, `13` and `14` brackets both
  thresholds). Right: the speed/accuracy Pareto frontier (emulation on solid,
  off dashed), speedup over the scalar FP64 GEMM, cuBLAS DGEMM as an FP64
  reference, and the accuracy-target decisions (`autopick.tsv`) circled.
- `../../figures/ozp_surrogate.pdf`, drawn by `plot_surrogate.py`. Left:
  Ozaki-II with `nm = 8`, `9` and `10` against the chemistry threshold at every
  deploy `N`. Right: the speedup over scalar FP64 grows with `N`.

`make figures` falls back to `golden/rtx5090/` when you have not run a step,
so it works without a GPU.

## Inputs it reads

- `profiles/surrogate512/`: the `N = 512` profile the expected decisions were
  made from (`PROFILES=shipped` solves from it).
- `golden/rtx5090/`: the RTX 5090 run that `EXPECTED_RESULTS.md` cites.
- The Ozaki-II, TCEC and direct GEMM runtimes a rewritten program calls:
  [`../../GEMM_RUNTIMES.md`](../../GEMM_RUNTIMES.md).
