# GENGA force and drift kernels (Fig. 3, Fig. 4); each target runs one script in scripts/.

include ../../config.mk

export POSEIDON_ROOT CLANG POSEIDON_CLANGXX CUDA GPU_ARCH GPU_ID PY \
       POSEIDON_INC

.PHONY: all figures force force-mfoff kepler kepler-herbieoff timeseries unified \
        clean

figures:
	mkdir -p ../../figures
	$(PY) genga_figures.py --out-dir ../../figures

force:
	bash scripts/run_force.sh

force-mfoff:
	bash scripts/run_force_mfoff.sh

kepler:
	bash scripts/run_kepler.sh

kepler-herbieoff:
	bash scripts/run_kepler_herbieoff.sh

timeseries:
	bash scripts/run_kepler_timeseries.sh

unified:
	bash scripts/run_unified.sh

clean:
	rm -f *.exe *.tsv
	rm -rf fpprofile fpprofile_kep_rolled fpprofile_unified measured \
	       tmp tmp_kep tmp_kep_ho tmp_uni tmp_mfoff tmp_quick logs logs_kep logs_uni \
	       logs_kep_ho logs_mfoff logs_ts reports reports_kep_rolled reports_kep_ho \
	       reports_mfoff reports_unified cache cache_mfoff cache_kep_herbieoff \
	       solve_kepler.log solve_unified.log __pycache__
	# Herbie results cached by GENGA_HERBIE=run stay; only their DP tables go.
	rm -f cache_kep_rolled/table.json cache_kep_rolled/budgets.txt \
	      cache_unified/table.json cache_unified/budgets.txt
