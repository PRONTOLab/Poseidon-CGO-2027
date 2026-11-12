# Expected results

This is the one place the artifact states what a reviewer should see. Every
number and every solver decision below comes from two data files,
`expected/picks.tsv` (the decisions) and `expected/claims.tsv` (the numbers,
their tolerances and the saved measurement behind each), and the tables are
generated from them by `python3 scripts/expected_results.py`. The per-benchmark
`expected_results.md` files describe what each benchmark does and point here.

## How to run, and what "reproduced" means

```sh
export POSEIDON_BUILD=<Enzyme build tree configured with -DENZYME_POSEIDON=ON>
./run_artifact.sh --picks     # every decision below, compile-only, no GPU (minutes)
./run_artifact.sh --quick     # smoke tests on the GPU (~15 min)
./run_artifact.sh             # everything (hours); GPU_ID=<n> selects the GPU
```

A row is reproduced when:

1. **Decision.** The compiler applies the configuration the decision table names
   for the row. A decision is a compile-time function of the profile, the cost
   model and the compiler: with the shipped RTX 5090 cost model and the shipped
   profiles it is exact on any machine, and `./run_artifact.sh --picks` checks
   every row without a GPU (`picks_check/picks.tsv`: `SAME` per row). During a
   GPU run each row prints `[expected] <row>: ...` saying whether the compile
   applied that configuration; the results bundle collects these lines in
   `expected_picks.txt`.
2. **Accuracy.** Within the tolerance of its row. The accuracy of an unchanged
   configuration reproduces to the printed digit where the computation is
   deterministic (GENGA L2 and energy drift, R-ChFSI eigenvalue L2 and locked
   count, the fig2 drift file bit for bit), and to about 0.5% where the
   application reduces with atomics (ozp's density-matrix RMSD: the Ozaki-II
   product is exact in INT32, the purification around it is not). Repeated runs
   of this artifact on the RTX 5090 reproduced every accuracy column this way.
3. **Speedup.** Within 10% of the expected value on an idle RTX 5090 at stock
   clocks, as the average of 5 runs (repeated runs of the same binaries landed
   within -8% to +5%). A GPU that cannot sustain its boost clock slows the FP64
   baselines disproportionately and inflates ratios: rerun on an idle, cool GPU.

Every `Expected` cell is one run of this artifact on an RTX 5090 (`Source`
names the row of its saved output under `benchmarks/*/golden/rtx5090/`); the
`Paper` column is what the paper printed.

## Required: RTX 5090 or a similar card

The required path is an NVIDIA RTX 5090 or a similar card: a GPU of compute
capability 12.0, i.e. `nvidia-smi --query-gpu=name,compute_cap --format=csv`
prints `12.0` (the GeForce RTX 50 series and the RTX PRO Blackwell workstation
cards), on which FP64 runs at 1/64 of the FP32 rate. On such a card the
artifact uses the shipped cost model measured on the paper's RTX 5090
(`cost_models/cm_sm_120_RTX5090.csv`; the bootstrap prints which card it found
and `poseidon-calibrate --check` validates the model against it), so:

- on an **RTX 5090** every row below applies as stated;
- on **another compute-capability-12.0 card** the decisions are the same
  (they are the RTX 5090 model's), the accuracy columns hold within the same
  tolerances, and the speedups are that card's own: the paper's qualitative
  claims (each row's `Claim`) are what to check, not the 10% band.

Nothing in this section needs a B300 or a GH200, and nothing in the optional
sections can fail it.

### Decisions

`Selected by` is how the pipeline reaches the row: an accuracy target, the
row's Pareto frontier point (its share of the profiled cost, as
`-poseidon-cost-budget` takes it; the frontier's shares are in the solve's
`budgets.txt`), or the cheapest frontier point. The configuration is the
compiler's own name for each applied candidate (`[poseidon] pin:` in the compile
log; `expected/picks.tsv` holds the full value, which
`-poseidon-apply-rewrites` takes). R-ChFSI sites: `hemm` is the Chebyshev
filter `H·Z`, `gram` the Gram product, `rrhv` and `rrproj` the two
Rayleigh-Ritz products, `resid` the residual.

<!-- BEGIN GENERATED decisions -->
| Row | Selected by | Configuration (shipped RTX 5090 model) | Compiler's names (`[poseidon] pin:`) |
|---|---|---|---|
| `genga-force-fp32` | cheapest frontier point | the whole force kernel in FP32 | `PT0=All FP64(0%) + FP32(100%)` |
| `genga-force-expansion` | `-poseidon-cost-budget=-0.7341213131565124` | the whole force kernel as a two-component FP32 expansion | `PT0=All FP64(0%) + Expansion2(100%)` |
| `genga-drift-naive` | cheapest frontier point | the naive downcast: both drift subgraphs (Newton step PT6, f/g block PT7) in FP32, no rewrite (cheapest point of the Herbie-off solve) | `PT6=All FP64(0%) + FP32(100%)`<br>`PT7=All FP64(0%) + FP32(100%)` |
| `genga-drift-algebraic` | `-poseidon-cost-budget=-0.7833673264258758` | the algebraic configuration: the Newton-step subgraph rewritten by Herbie into a binary32 expression (R13: numerator the 4-term series of the residual, denominator a series too) and the f/g block as a two-component FP32 expansion (PT7); R0, R2, R4, R6 are exact rewrites of fma terms in the small FP64 subgraphs (a zero operand dropped, operand order) | `PT7=All FP64(0%) + Expansion2(100%)`<br>`R0=#s(approx (+ (* v2 v3) (* v0 v1)) (*.f64 v0 v1))`<br>`R13=(/.f32 #s(approx (+ (* (+ (+ (* (sin v10) v12...`<br>`R2=#s(approx (+ (* v2 v5) (* v4 v1)) (fma.f64 v1...`<br>`R4=#s(approx (+ (* v2 v7) (* v6 v1)) (fma.f64 v1...`<br>`R6=#s(approx (+ (* v9 v3) (* v0 v8)) (*.f64 v0 v8))` |
| `genga-unified` | `-poseidon-cost-budget=-0.7046299397030764` | force as a two-component expansion (PT0) with a zero term dropped (R4); Newton step in binary64 with sin as a series (R6); f/g functions as a two-component expansion (PT2); zero drift terms dropped (R16, R18) | `PT0=All FP64(0%) + Expansion2(100%)`<br>`PT2=Funcs FP64(0%) + Expansion2(100%)`<br>`R16=#s(approx (+ (* v22 v9) (* (+ (* v25 v15) v26...`<br>`R18=#s(approx (+ (* v24 v9) (* (+ (* v25 v15) v26...`<br>`R4=#s(approx (+ (* (- v8 v9) (* v2 (* (/ 1 (sqrt...`<br>`R6=(-.f64 v10 (/.f64 (-.f64 (-.f64 v10 (-.f64 v1...` |
| `ozp-tau1e-6` | `-poseidon-tau=1e-6` | FP16TCEC: the error-corrected FP16 tensor-core GEMM (cuMpSGEMM), host-dispatched | `matmul: M0=tcec-dispatch wmma m16n16k16 f16/f32` |
| `ozp-tau1e-7` | `-poseidon-tau=1e-7` | Ozaki-II, 8 moduli (host-dispatched); the paper's nm = 9 is the pin _Z10matmul_optPdPKdS1_:M0=ozaki-ii nm=9 wmma m16n16k16 s8/s32 | `matmul: M0=ozaki-ii nm=8 wmma m16n16k16 s8/s32` |
| `ozp-tau1e-12` | `-poseidon-tau=1e-12` | Ozaki-II, 12 moduli (host-dispatched); the paper's nm = 13 is the pin _Z10matmul_optPdPKdS1_:M0=ozaki-ii nm=13 wmma m16n16k16 s8/s32 | `matmul: M0=ozaki-ii nm=12 wmma m16n16k16 s8/s32` |
| `ozp-direct-bf16` | `-poseidon-cost-budget=-0.9910149999960528` | cuBLAS GEMM, BF16 inputs, FP32 accumulation | `matmul: M0=direct-dispatch cublas bf16/f32` |
| `ozp-direct-tf32` | `-poseidon-cost-budget=-0.9859669999953171` | cuBLAS GEMM, TF32 inputs, FP32 accumulation | `matmul: M0=direct-dispatch cublas tf32/f32` |
| `rchfsi-tf32tcec` | `-poseidon-cost-budget=-0.6835037117561215` | TF32TCEC on the filter only | `hemm: M0=tcec n=2 wmma m16n16k8 tf32/f32` |
| `rchfsi-bf16tcec` | `-poseidon-cost-budget=-0.7593169255791722` | BF16TCEC on the filter only | `hemm: M0=tcec n=2 wmma m16n16k16 bf16/f32` |
| `rchfsi-bf16tcec-nm14` | `-poseidon-cost-budget=-0.8037852980785799` | BF16TCEC filter + Ozaki-II nm = 14 on rrhv | `hemm: M0=tcec n=2 wmma m16n16k16 bf16/f32`<br>`rrhv: M0=ozaki-ii nm=14 wmma m16n16k16 s8/s32` |
| `rchfsi-knee` | `-poseidon-cost-budget=-0.8196793023307888` | BF16 filter + Ozaki-II nm = 12 on rrhv | `hemm: M0=wmma m16n16k16 bf16/f32`<br>`rrhv: M0=ozaki-ii nm=12 wmma m16n16k16 s8/s32` |
| `rchfsi-blind` | `-poseidon-cost-budget=-0.8196793023307888` | Kappa removed: BF16TCEC filter, FP16TCEC dispatch on the Gram product, Ozaki-II nm = 10 on rrhv | `gram: M0=tcec-dispatch wmma m16n16k16 f16/f32`<br>`hemm: M0=tcec n=2 wmma m16n16k16 bf16/f32`<br>`rrhv: M0=ozaki-ii nm=10 wmma m16n16k16 s8/s32` |
<!-- END GENERATED decisions -->

### Numbers

<!-- BEGIN GENERATED required -->
| Claim | Metric | Paper | Expected | Tolerance | Source | Run |
|---|---|---|---|---|---|---|
| `fig2-gpu`: 3xFP32 emulation runs faster than FP64 on the GPU | GPU time, 3xFP32 vs FP64 | 2.1x faster | 2.09x faster (5.470 -> 2.620 ms) | 10% | `benchmarks/fig2_drift/golden/rtx5090/fig2_drift.log` | `cd benchmarks/fig2_drift && make all` |
|  | CPU time, 3xFP32 vs FP64 | 8.8x slower | 8.6x slower (22.530 -> 193.504 ms) | 15% (one CPU thread) | `benchmarks/fig2_drift/golden/rtx5090/fig2_drift.log` |  |
|  | 3xFP32 drift after 2^20 steps | below 1e-19, flat | 1.611652e-20 | drift.csv equal to the reference file | `benchmarks/fig2_drift/golden/rtx5090/drift.csv` |  |
| `genga-force-fp32`: force kernel in FP32: 51x at L2 5e-6 | speedup over FP64 | 51x | 53.085x | 10% | `benchmarks/genga_nbody/golden/rtx5090/force_full_g0.tsv:2` | `cd benchmarks/genga_nbody && make force` |
|  | L2 vs FP64 | 5e-6 | 5.056976e-06 | exact | `benchmarks/genga_nbody/golden/rtx5090/force_full_g0.tsv:2` |  |
| `genga-force-expansion`: whole-kernel two-component expansion: near-FP64 at 4.5x | speedup over FP64 | 4.5x (Table 1: 4.46x) | 5.338x | 10% | `benchmarks/genga_nbody/golden/rtx5090/force_full_g0.tsv:7` | `cd benchmarks/genga_nbody && make force` |
|  | L2 vs FP64 | 1.8e-13 | 1.828723e-13 | near-FP64 (below 1e-12); then exact | `benchmarks/genga_nbody/golden/rtx5090/force_full_g0.tsv:7` |  |
| `genga-force-expoff`: with expansion disabled, near-FP64 accuracy needs FP64 | most accurate point below FP64 cost, expansion off | none below 1e-12 (Fig. 3 left, dashed) | L2 4.091316e-08 at 1.329x | L2 exact, speedup 10% | `benchmarks/genga_nbody/golden/rtx5090/force_mfoff.tsv:5` | `cd benchmarks/genga_nbody && make force force-mfoff` |
| `genga-drift-naive`: the naive FP32 downcast of the drift kernel | energy drift (20k steps) | 1.4e-4 | 1.380877e-04 | exact | `benchmarks/genga_nbody/golden/rtx5090/kepler_herbieoff_g0.tsv:2` | `cd benchmarks/genga_nbody && make kepler kepler-herbieoff` |
|  | speedup over FP64 | (Fig. 3 right) | 9.731x | 10% | `benchmarks/genga_nbody/golden/rtx5090/kepler_herbieoff_g0.tsv:2` |  |
| `genga-drift-algebraic`: algebraic rewrites keep the FP32 drift kernel near-FP64: 3000x more accurate than the downcast at 5.4x | energy drift (20k steps) | 4.7e-8 (Table 1: 4.655009e-08) | 2.977160e-08 | printed digit | `benchmarks/genga_nbody/golden/rtx5090/kepler_full_g0.tsv:90` | `cd benchmarks/genga_nbody && make kepler` |
|  | speedup over FP64 | 5.4x (Table 1: 5.48x) | 5.613x | 10% | `benchmarks/genga_nbody/golden/rtx5090/kepler_full_g0.tsv:90` |  |
| `genga-drift-ablation`: the algebraic rewrite against the naive downcast: the rewritten FP32 kernel has far less energy drift | energy drift, naive downcast (genga-drift-naive) / algebraic configuration (genga-drift-algebraic), with the speedup of each | 800x (ablation paragraph); 3000x (1.4e-4 vs 4.7e-8, results paragraph) | 4638x (1.380877e-04 at 9.731x vs 2.977160e-08 at 5.613x) | printed digit | `benchmarks/genga_nbody/golden/rtx5090/kepler_herbieoff_g0.tsv, benchmarks/genga_nbody/golden/rtx5090/kepler_full_g0.tsv` | `cd benchmarks/genga_nbody && make kepler kepler-herbieoff` |
| `genga-drift-timeseries`: the downcast drifts away, the rewritten kernel stays near FP64 over 5e5 steps | naive FP32 drift at 5e5 steps | 3e-3 | 2.856903e-03 | exact | `benchmarks/genga_nbody/golden/rtx5090/kepler_timeseries.tsv:10` | `cd benchmarks/genga_nbody && make kepler kepler-herbieoff timeseries` |
|  | Poseidon drift at 5e5 steps | 6e-7 (4500x below naive) | 1.702847e-07 (16777x below naive) | printed digit | `benchmarks/genga_nbody/golden/rtx5090/kepler_timeseries.tsv:10` |  |
| `genga-unified`: one solve over force and drift: expansion on the force, algebraic rewrites on the drift, FP64-grade at 4.4x | L2 vs FP64 (position and velocity, one step) | 2.8e-14 | 2.777272e-14 | FP64-grade (below 1e-13); then printed digit | `benchmarks/genga_nbody/golden/rtx5090/unified.tsv:19` | `cd benchmarks/genga_nbody && make unified` |
|  | speedup over FP64 | 4.4x | 5.831x | 10% | `benchmarks/genga_nbody/golden/rtx5090/unified.tsv:19` |  |
| `ozp-tau1e-7`: tau = 1e-7 picks an Ozaki-II rung (paper: nm = 9, 24.3x over FP64, 14.0x over cuBLAS DGEMM; this compiler: nm = 8) | deployed RMSD vs the exact projector | 3.0e-8 | 1.375262e-08 | 0.5% (atomics) | `benchmarks/ozp/golden/rtx5090/autopick.tsv:3` | `cd benchmarks/ozp && bash scripts/run_autopick.sh 1e-7` |
|  | speedup over scalar FP64 | 24.3x | 27.396x | 10% | `benchmarks/ozp/golden/rtx5090/autopick.tsv:3` |  |
|  | speedup over cuBLAS DGEMM | 14.0x | 15.760x | 10% | `benchmarks/ozp/golden/rtx5090/autopick.tsv:3` |  |
| `ozp-tau1e-6`: tau = 1e-6 picks FP16TCEC, a cheaper FP32-class product (26.7x at RMSD 1.7e-7) | deployed RMSD vs the exact projector | 1.7e-7 | 1.672398e-07 | 0.5% (atomics) | `benchmarks/ozp/golden/rtx5090/autopick.tsv:2` | `cd benchmarks/ozp && bash scripts/run_autopick.sh 1e-6` |
|  | speedup over scalar FP64 | 26.7x | 30.795x | 10% | `benchmarks/ozp/golden/rtx5090/autopick.tsv:2` |  |
| `ozp-tau1e-12`: tau = 1e-12 picks an Ozaki-II rung near FP64 (paper: nm = 13; this compiler: nm = 12) | deployed RMSD vs the exact projector | 9.4e-13 | 3.802463e-13 | 0.5% (atomics) | `benchmarks/ozp/golden/rtx5090/autopick.tsv:4` | `cd benchmarks/ozp && bash scripts/run_autopick.sh 1e-12` |
|  | speedup over scalar FP64 | 18.6x | 20.172x | 10% | `benchmarks/ozp/golden/rtx5090/autopick.tsv:4` |  |
| `ozp-tau-interval`: the same strategy is picked for a whole interval of tau (paper: nm = 9 for every tau in [9.2e-9, 1.3e-7]) | modelled domain error of nm = 8 and of nm = 9 (nm = 8 is picked for tau >= the first, nm = 9 down to the second) | 9.2e-9, 1.3e-7 | 2.275334e-08, 1.513766e-09 | exact (compile-time) | `the candidate table of picks_check/ozp-tau1e-7.log` | `./run_artifact.sh --picks` |
| `ozp-direct-bf16`: direct reduced-precision library GEMMs are fast but stall above the threshold (up to 71.0x, RMSD 1.6e-4) | speedup over scalar FP64 / RMSD, cuBLAS BF16 GEMM | 70.9x / 4.8e-4 (Fig. 5) | 71.224x / 4.760054e-04 | speedup 10%, RMSD 0.5% (atomics) | `benchmarks/ozp/golden/rtx5090/dispatch_rungs.tsv:3` | `cd benchmarks/ozp && make dispatch` |
| `ozp-direct-tf32`:  | speedup over scalar FP64 / RMSD, cuBLAS TF32 GEMM | 54.4x / 1.6e-4 (Fig. 5) | 54.225x / 1.601559e-04 | speedup 10%, RMSD 0.5% (atomics) | `benchmarks/ozp/golden/rtx5090/dispatch_rungs.tsv:4` | `cd benchmarks/ozp && make dispatch` |
| `ozp-direct-f16`:  | speedup over scalar FP64 / RMSD, cuBLAS FP16 GEMM (the reference arm: the compiler lists it on the frontier but refuses FP16 for these operands) | 71.0x / 1.6e-4 | 73.199x / 1.604838e-04 | speedup 10%, RMSD 0.5% (atomics) | `benchmarks/ozp/golden/rtx5090/direct_tc_lib.tsv:3` | `cd benchmarks/ozp && make direct-tc-lib` |
| `ozp-inkernel-ladder`: in-kernel tensor-core raises (FP16, BF16, TF32 inputs) stall above the threshold | speedup / RMSD of FP16, BF16, TF32 direct raises | 8.9x to 14.5x, above 1e-7 | 14.064x / 1.596695e-04, 14.114x / 4.760054e-04, 9.815x / 1.589214e-04 | speedup 10%, RMSD 0.5% | `benchmarks/ozp/golden/rtx5090/pareto.tsv:3-5` | `cd benchmarks/ozp && make pareto` |
|  | speedup / RMSD of in-kernel FP16TCEC (2 terms) | (Fig. 5) | 11.179x / 1.664176e-07 | speedup 10%, RMSD 0.5% | `benchmarks/ozp/golden/rtx5090/pareto.tsv:6` |  |
| `ozp-profile-transfer`: the Ozaki-II rung picked from the N = 512 profile meets 1e-7 at the deploy sizes; speedup grows with N | nm = 9 speedup over scalar FP64, N = 1024 / 8192 | 12.4x / 47.0x | 12.14x / 49.49x | 10% | `benchmarks/ozp/golden/rtx5090/surrogate.tsv` | `cd benchmarks/ozp && make surrogate` |
|  | nm = 9 RMSD at N = 8192 | 1.16e-7 (nm = 10 needed) | 1.566000e-09 | 0.5% | `benchmarks/ozp/golden/rtx5090/surrogate.tsv:20` |  |
| `rchfsi-sensitivity`: the profile tells the washed-out filter (S ~ 1e4) from the residual, Gram and projection products (S ~ 1e12) | Kappa per site: hemm / resid / rrhv / rrproj / gram | ~1e4 vs ~1e12 | 9.0e3 / 8.3e12 / 2.2e12 / 8.9e11 / 6.7e11 | order of magnitude (atomics) | `benchmarks/chase_eigensolver/profiles/surrogate512/*.fpprofile (Kappa lines)` | `cd benchmarks/chase_eigensolver && make profile` |
| `rchfsi-tf32tcec`: TF32TCEC on the filter only | speedup over FP64 / eigenvalue L2 / locked | 1.88x / 3.2e-15 / 64 of 64 | 1.832x / 3.152291e-15 / 64 of 64 | speedup about 4% run to run (observed, shared machine); L2, locked exact | `benchmarks/chase_eigensolver/golden/rtx5090/scvh_probe_on.tsv:13` | `cd benchmarks/chase_eigensolver && bash scripts/run_ladder.sh` |
| `rchfsi-bf16tcec`: BF16TCEC on the filter only, indistinguishable from FP64 | speedup over FP64 / eigenvalue L2 / locked | 2.08x / 2.7e-15 / 64 of 64 | 2.084x / 2.724089e-15 / 64 of 64 | speedup about 4% run to run (observed, shared machine); L2, locked exact | `benchmarks/chase_eigensolver/golden/rtx5090/scvh_probe_on.tsv:10` | `cd benchmarks/chase_eigensolver && bash scripts/run_ladder.sh` |
| `rchfsi-bf16tcec-nm14`: BF16TCEC filter + Ozaki-II nm = 14 on a Rayleigh-Ritz product | speedup over FP64 / eigenvalue L2 / locked | 2.22x / 4.7e-14 / 64 of 64 | 2.160x / 4.633054e-14 / 64 of 64 | speedup about 4% run to run (observed, shared machine); L2 printed digit; locked 64 of 64 | `benchmarks/chase_eigensolver/golden/rtx5090/scvh_probe_on.tsv:9` | `cd benchmarks/chase_eigensolver && bash scripts/run_ladder.sh` |
| `rchfsi-knee`: the fastest strategy that locks every eigenpair: BF16 filter + Ozaki-II nm = 12, within 3% of the ~2.5x cuSOLVER-bound limit | speedup over FP64 / eigenvalue L2 / locked | 2.43x / 1.28e-11 / 64 of 64 | 2.270x / 1.230509e-11 / 64 of 64 | speedup about 4% run to run (observed, shared machine); L2 printed digit; locked 64 of 64 | `benchmarks/chase_eigensolver/golden/rtx5090/scvh_probe_on.tsv:8` | `cd benchmarks/chase_eigensolver && bash scripts/run_ladder.sh` |
| `rchfsi-blind`: sensitivity-blind solve at the same cost: no eigenpair locks | locked / speedup over FP64 | 0 of 64 / 0.28x | 0 of 64 / 0.277x | locked exact; speedup about 4% run to run (observed, shared machine) | `benchmarks/chase_eigensolver/golden/rtx5090/ablation_knee.tsv:3` | `cd benchmarks/chase_eigensolver && bash scripts/run_ladder.sh` |
<!-- END GENERATED required -->

### Where the configuration differs from the paper's

The decisions table is what this artifact's compiler picks. Where that is not
the configuration the paper reports, the difference and its effect:

- **`ozp-tau1e-7`, `ozp-tau1e-12`.** The Ozaki-II runtime the compiler links
  returns GEMMul8's fast-mode result bit for bit, which is more accurate at
  equal `nm` than the runtime the paper measured, and the compiler's accuracy
  model follows it: tau = 1e-7 picks `nm = 8` (paper: `nm = 9`) and tau = 1e-12
  picks `nm = 12` (paper: `nm = 13`), one modulus fewer, so faster. The paper's
  configurations are frontier rungs of `make dispatch` (`dispatch_rungs.tsv`)
  and can be forced by name:
  `-poseidon-apply-rewrites=_Z10matmul_optPdPKdS1_:M0=ozaki-ii nm=9 wmma m16n16k16 s8/s32`
  (`nm=13` likewise).
- **`genga-drift-algebraic`.** The algebraic configuration has two
  ingredients: Herbie's rewrite of the Newton-step subgraph into a binary32
  expression (`R13`; its numerator is the paper's 4-term series, its
  denominator a series as well, where the paper's kept `sin` and `cos`), and
  the f/g block computed as a two-component FP32 expansion (`PT7`), as in the
  paper; `R0`, `R2`, `R4`, `R6` are exact rewrites of fma terms in the small
  FP64 subgraphs. `genga-drift-ablation` compares it with the naive FP32
  downcast (`genga-drift-naive`).
- **`genga-unified`.** The FP64-grade point (the cheapest frontier point whose
  modelled error is within 10x of the FP64 code as written) combines the
  force kernel as a two-component expansion with an algebraic rewrite of the
  drift (the Newton step's `sin` as a series) and terms dropped that are zero
  for this input; the paper's configuration expanded parts of the drift instead.
- **`ozp-direct-*`.** The frontier lists the cuBLAS FP16 dispatch as its
  cheapest point, but the model refuses FP16 for ozp's operands (as above), so
  no budget selects it and the frontier sweep records it as `NOT-SELECTABLE`;
  the BF16 and TF32 dispatches are selected at their shares. The FP16 point of
  Fig. 5 comes from the plain cuBLAS reference arm (`make direct-tc-lib`).
- **`rchfsi-blind`.** Without the condition numbers the solve at the knee's
  share lowers three products, as the paper's blind solve did, but not the same
  three: BF16TCEC on the filter, the FP16TCEC dispatch on the Gram product and
  Ozaki-II `nm = 10` on a Rayleigh-Ritz product, where the paper's used TF32 on
  the filter and the FP16TCEC dispatch on both Rayleigh-Ritz products.

### Where the numbers differ from the paper's

Same configuration, different number (more than 1% from the paper):

- **R-ChFSI.** The same configurations lock 64 of 64 in the paper's 9
  iterations at the paper's eigenvalue error; their speedups (`rchfsi-knee`
  2.27x against the paper's 2.43x, `rchfsi-bf16tcec-nm14` 2.16x against 2.22x,
  `rchfsi-tf32tcec` 1.83x against 1.88x) are within the run-to-run spread of
  this solver on the RTX 5090, about 4% on a shared machine. Back to back on
  one GPU, the binary the paper measured averages 84.47 ms today (the paper
  recorded 81.39 ms) and this artifact's knee configuration 83.42 ms, with
  equal FP64 baselines.
- **`ozp-profile-transfer`.** `nm = 9` at `N = 1024` runs 12.1x (paper 12.4x);
  at `N = 8192` it runs 49.5x (paper 47.0x) and its RMSD is 1.6e-9, so it meets
  1e-7 at every deploy size where the paper's needed `nm = 10` at `N = 8192`
  (the Ozaki-II runtime's accuracy, as for `ozp-tau1e-7`).

Better than the paper, same claim: the GENGA expansion and algebraic rows
(`genga-force-expansion` 5.3x, `genga-drift-algebraic` 5.6x at drift 3.0e-8,
`genga-unified` 5.8x at 2.8e-14), the Ozaki-II tau rows (one modulus fewer) and
`ozp-tau1e-6` (FP16TCEC, 30.8x at the paper's RMSD 1.67e-7; paper 26.7x).

## Choices, and what each one guarantees

**Cost model.** By default the shipped RTX 5090 model (`cost_models/`), which
makes every decision above. `./run_artifact.sh --calibrate` measures this card
instead (`poseidon-calibrate`, ~15-30 min; delete `cost_models/target.env` to
return). With an own model the decisions are that model's: frontier sweeps and
ladders enumerate its own frontier, tau rows pick against its prices, and each
row's `[expected]` line says whether the configuration differs from the table.
What is then promised is the paper's qualitative claims, not the table's cells:
the tau picks are accepted by the accuracy model at their target, the
sensitivity-guided R-ChFSI picks rewrite the filter before any other product,
the GENGA drift frontier holds algebraic rewrites more accurate than the naive
downcast at equal cost, and every accuracy column of a configuration that also
appears in the table matches it.

**Herbie.** By default (`GENGA_HERBIE=replay`) the GENGA drift and unified
solves replay the shipped Herbie answers (`benchmarks/genga_nbody/herbie_answers/`)
whichever Herbie the compiler was built with, without starting one; a question
the shipped answers do not hold stops the compile. The shipped answers are data
searched for the shipped RTX 5090 model (an answer is keyed by the device family
of the cost model and by its Herbie platform); with it this reproduces the GENGA
decisions exactly. Another device searches (`GENGA_HERBIE=run`) or replays an
answer pack made for it (`HERBIE_ANSWERS=<dir>`). `GENGA_HERBIE=run` searches with Herbie (`HERBIE=<binary>`
optional) from this run's profile: about 10 minutes and up to 13 GB per solve.
Its answers can differ from the shipped ones (Herbie's search is randomized
by seed and sample, and a fresh profile changes the questions' ranges); what is
promised is the drift frontier's shape (rewrites beat the downcast at equal
cost). The search runs at Herbie's Taylor-term caps 2 and 4, which needs a
Herbie that accepts `--num-taylor-terms`; with `GENGA_TAYLOR_TERMS=` (empty)
any Herbie searches at its own cap of 2 terms, and every GENGA row reproduces
except `genga-drift-algebraic`, `genga-drift-ablation`, the Poseidon
column of `genga-drift-timeseries` and `genga-unified`, which need the 4-term
series of the Newton step.

**Profiles.** By default (`PROFILES=fresh`) every benchmark profiles on this
GPU and solves from that profile, except the GENGA drift and unified solves in
replay mode, which read the shipped profiles the answers were searched for (the
profiling step prints whether the fresh profile has the same ranges).
`PROFILES=shipped` makes every solve read `benchmarks/*/profiles/`, the
profiles the decisions above were made from. Value ranges and counts in a
profile are deterministic; its sums and condition numbers are accumulated with
atomics and move in the last digits, which no decision above is close to.

## Optional: B300 and GH200

These sections are not needed for the required path and cannot fail it. On any
GPU other than compute capability 12.0 the artifact measures that GPU's cost
model first (`./run_artifact.sh` does it), and every decision and number is
then that GPU's own. The paper's Table 1 reports the decisions three GPUs made;
the B300 and GH200 columns are below as the paper reports them. They were made
with those GPUs' own measured models and profiles, which this artifact does not
ship; a run on such a GPU measures its own, so its decisions can differ where
two candidates are priced close together, and its numbers are compared with
these qualitatively (which product is rewritten, and that the accuracy target
is met).

### B300 (sm_103)

<!-- BEGIN GENERATED b300 -->
| Claim | Metric | Paper | Expected | Tolerance | Source | Run |
|---|---|---|---|---|---|---|
| `t1-force-b300`: force kernel: two-component expansion | speedup / L2 | 4.26x / 1.765e-13 | (that GPU's run) | same rewrite class; L2 near-FP64 | paper Table 1 | `./run_artifact.sh (on the B300: measures its cost model first)` |
| `t1-drift-b300`: drift kernel: expansion + algebraic | speedup / energy drift | 5.65x / 4.674074e-08 | (that GPU's run) | same rewrite classes; drift near-FP64 | paper Table 1 |  |
| `t1-ozp7-b300`: ozp, tau = 1e-7: Ozaki-II nm = 9 | speedup / RMSD | 16.94x / 3.04e-8 | (that GPU's run) | an Ozaki-II rung; RMSD below 1e-7 | paper Table 1 |  |
| `t1-ozp6-b300`: ozp, tau = 1e-6: FP16TCEC | speedup / RMSD | 33.5x / 1.7e-7 | (that GPU's run) | an FP32-class product; RMSD near 1e-7 | paper Table 1 |  |
| `t1-rchfsi-b300`: R-ChFSI: BF16TCEC on the filter | speedup / locked | 2.18x / 64 of 64 | (that GPU's run) | the filter rewritten; 64 of 64 locked | paper Table 1 |  |
<!-- END GENERATED b300 -->

### GH200 (sm_90)

<!-- BEGIN GENERATED gh200 -->
| Claim | Metric | Paper | Expected | Tolerance | Source | Run |
|---|---|---|---|---|---|---|
| `t1-force-gh200`: force kernel: no rewrite (FP64 is fast) | speedup | 1.00x | (that GPU's run) | no profitable rewrite | paper Table 1 | `./run_artifact.sh (on the GH200: measures its cost model first)` |
| `t1-drift-gh200`: drift kernel: FP32 + algebraic | speedup / energy drift | 2.09x / 4.474077e-08 | (that GPU's run) | algebraic rewrites stay profitable | paper Table 1 |  |
| `t1-ozp7-gh200`: ozp, tau = 1e-7: FP64 tensor cores (native DGEMM) | speedup / RMSD | 75.2x / 9.3e-16 | (that GPU's run) | the FP64 tensor-core product | paper Table 1 |  |
| `t1-ozp6-gh200`: ozp, tau = 1e-6: FP64 tensor cores (native DGEMM) | speedup / RMSD | 76.1x / 9.3e-16 | (that GPU's run) | the FP64 tensor-core product | paper Table 1 |  |
| `t1-rchfsi-gh200`: R-ChFSI: BF16 filter + FP64 tensor cores on the sensitive products | speedup / locked | 5.80x / 64 of 64 | (that GPU's run) | reduced precision only on the filter; 64 of 64 locked | paper Table 1 |  |
<!-- END GENERATED gh200 -->
