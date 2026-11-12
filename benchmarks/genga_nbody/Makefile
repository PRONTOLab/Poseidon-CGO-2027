# genga_nbody — GENGA N-body: force kick (MultiFloat/precision axis) + Kepler
# drift (Herbie algebraic-rewrite axis), eval §5.1. Verbatim GENGA kernels
# (accEncFull force, fastfg drift) driven through Poseidon.
#
# Config (compiler, CUDA, GPU_ID, python) comes from ../../config.mk; every
# variable is overridable on the command line. The Poseidon phase scripts live in
# scripts/ and inherit these via the environment (single source of truth).
#
# COST MODEL: the device's cost model uniformly scaled x1e6 (for the shipped
# RTX 5090 model, ../../cost_models/x1e6/; otherwise derived by
# scripts/common.sh): the shipped Herbie answers (herbie_answers/) were searched
# against the Herbie platform of that scaled copy, and a uniform scale leaves
# every solver decision unchanged. GENGA_HERBIE=run searches with Herbie
# instead of replaying them (scripts/common.sh).

include ../../config.mk

# Export the resolved config so scripts/*.sh see identical values.
export POSEIDON_ROOT CLANG POSEIDON_CLANGXX CUDA GPU_ARCH GPU_ID PY \
       POSEIDON_INC

.PHONY: all figures force force-mfoff kepler kepler-herbieoff timeseries unified \
        quick run-all clean

## Regenerate both paper figures from whatever TSVs are present (fresh or golden).
figures:
	mkdir -p ../../figures
	$(PY) genga_figures.py --out-dir ../../figures

## --- Force kick (expansion / FP32 precision axis) -> force_full_g0.tsv ---
force:
	bash scripts/run_force.sh

## Expansion-off ablation (needs `force` first) -> force_mfoff.tsv
force-mfoff:
	bash scripts/run_force_mfoff.sh

## --- Kepler drift (Herbie axis) -> kepler_full_g0.tsv ---
kepler:
	bash scripts/run_kepler.sh

## Herbie-off ablation (needs `kepler` first) -> kepler_herbieoff_g0.tsv
kepler-herbieoff:
	bash scripts/run_kepler_herbieoff.sh

## Energy-drift time series (needs `kepler` and `kepler-herbieoff` first) -> kepler_timeseries.tsv
timeseries:
	bash scripts/run_kepler_timeseries.sh

## Unified force-then-drift single solve -> unified.tsv (reported number, no figure)
unified:
	bash scripts/run_unified.sh

## Kick-the-tires: one force budget + a short (2k-step) drift accuracy check.
quick:
	bash quick.sh

## Full paper pipeline for §5.1 (both figures + all reported numbers).
run-all:
	bash run_all.sh

clean:
	rm -f *.exe *.tsv
	rm -rf fpprofile fpprofile_kep_rolled fpprofile_unified measured \
	       tmp tmp_kep tmp_kep_ho tmp_uni tmp_mfoff tmp_quick logs logs_kep logs_uni \
	       logs_kep_ho logs_mfoff logs_ts reports reports_kep_rolled reports_kep_ho \
	       reports_mfoff reports_unified cache cache_mfoff cache_kep_herbieoff \
	       solve_kepler.log solve_unified.log __pycache__
	# NOTE: keeps the shipped Herbie answers (herbie_answers/), the shipped
	# profiles (profiles/) and golden/. The Herbie caches of GENGA_HERBIE=run
	# searches stay in cache_kep_rolled/ and cache_unified/; only their derived
	# DP tables are dropped.
	rm -f cache_kep_rolled/table.json cache_kep_rolled/budgets.txt \
	      cache_unified/table.json cache_unified/budgets.txt
