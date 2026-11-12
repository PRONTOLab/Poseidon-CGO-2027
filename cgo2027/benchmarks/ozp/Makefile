# ozp — density-matrix purification (TRS4), eval §5.2. The iteration's dense FP64
# GEMM (X·X and X²·G) is extracted as one FP subgraph and driven through Poseidon,
# which proposes direct tensor-core raises, FP16TCEC, and Ozaki-II INT8 rungs and
# selects per a call-site accuracy target. Matmul-only (no Herbie).
#
# Config (compiler, CUDA, GPU_ID, python, cost model, Ozaki-II runtime) comes from
# ../../config.mk; every variable is overridable on the command line. The phase
# scripts live in scripts/ and inherit these via the environment.

include ../../config.mk

# Export the resolved config so scripts/*.sh see identical values.
export POSEIDON_ROOT CLANG POSEIDON_CLANGXX CUDA GPU_ARCH GPU_ID PY \
       POSEIDON_INC \
       OZAKI_RT TCEC_RT DIRECT_RT COST_MODEL

.PHONY: all figures data pareto dispatch direct-tc-lib autopick surrogate quick run-all clean

## Regenerate both paper figures from whatever TSVs are present (fresh or golden).
figures:
	mkdir -p ../../figures
	$(PY) plot_ozp.py --out ../../figures/ozp_convergence.pdf
	$(PY) plot_surrogate.py

## Generate the synthetic H / K_ref inputs (deterministic; regenerated, not shipped).
data:
	bash scripts/gen_data.sh

## --- Convergence ladder: in-kernel direct TC raises + FP16TCEC -> pareto.tsv, traj_*.tsv
pareto:
	bash scripts/run_pareto.sh

## --- The frontier (Ozaki-II and direct library dispatches) -> dispatch_rungs.tsv, traj_OZII_nm*.tsv
dispatch:
	bash scripts/run_dispatch_rungs.sh

## --- cuBLAS reference for the direct tensor-core raises (same operand precision
## as the in-kernel WMMA raises, one library GEMM) -> direct_tc_lib.tsv, traj_LIB_*.tsv
direct-tc-lib:
	bash scripts/run_direct_tc_lib.sh

## --- Call-site accuracy-target auto-pick (tau -> rewrite) -> autopick.tsv
autopick:
	bash scripts/run_autopick.sh

## --- SCVH surrogate N-sweep -> surrogate.tsv
surrogate:
	bash scripts/run_surrogate.sh

## Kick-the-tires: profile @512, then the tau=1e-7 auto-pick at N=2048 (row
## ozp-tau1e-7 of ../../EXPECTED_RESULTS.md). A few minutes; not a paper measurement.
quick:
	bash quick.sh

## Full paper pipeline for §5.2 (both figures + all reported numbers).
run-all:
	bash run_all.sh

clean:
	rm -f *.exe pareto.tsv dispatch_rungs.tsv direct_tc_lib.tsv autopick.tsv surrogate.tsv traj_*.tsv
	rm -rf fpprofile cache tmp __pycache__
	# NOTE: keeps golden/, profiles/ and the regenerable data/. `make clean` drops only the
	# derived DP tables / profiles / generated TSVs. To reclaim disk, also
	# `rm -rf data` (regenerate with `make data`).
