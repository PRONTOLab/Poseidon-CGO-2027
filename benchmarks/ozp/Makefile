# ozp density-matrix purification (Fig. 5); each target runs one script in scripts/.

include ../../config.mk

export POSEIDON_ROOT CLANG POSEIDON_CLANGXX CUDA GPU_ARCH GPU_ID PY \
       POSEIDON_INC \
       OZAKI_RT TCEC_RT DIRECT_RT COST_MODEL

.PHONY: all figures pareto dispatch direct-tc-lib autopick surrogate clean

figures:
	mkdir -p ../../figures
	$(PY) plot_ozp.py --out ../../figures/ozp_convergence.pdf

pareto:
	bash scripts/run_pareto.sh

dispatch:
	bash scripts/run_dispatch.sh

direct-tc-lib:
	bash scripts/run_direct_tc_lib.sh

autopick:
	bash scripts/run_autopick.sh

surrogate:
	bash scripts/run_surrogate.sh

clean:
	rm -f *.exe pareto.tsv dispatch.tsv direct_tc_lib.tsv autopick.tsv surrogate.tsv traj_*.tsv
	rm -rf fpprofile cache tmp __pycache__
