# R-ChFSI eigensolver (Fig. 6): make profile | probe | solve BUDGET=<share> [PROFDIR=<dir>] | baseline | figures.

include ../../config.mk

PROF_DEFS  ?= -DCHASE_N=512 -DCHASE_NEV=32
BENCH_DEFS ?= -DCHASE_N=2048 -DCHASE_NEV=64
BUDGET     ?= 0
# No trailing spaces: make would keep them in the value.
PROFDIR    ?= ./fpprofile
SLOGBITS   ?= 0
SRC        ?= chase_real.cu
PROBE_TOOL ?= $(dir $(POSEIDON_CLANGXX))poseidon-probe

CUDA_FLAGS = -x cuda --cuda-gpu-arch=$(GPU_ARCH) --cuda-path=$(CUDA) -fcuda-rdc \
    -O3 -I$(CUDA)/include -L$(CUDA)/lib64 -lcublas -lcusolver -lstdc++ -lm \
    -Wno-deprecated-declarations -Wno-unknown-cuda-version

# One joint DP over the five sites; 128 accuracy samples per candidate.
JOINT_FLAGS ?= -poseidon-joint-dp -poseidon-num-samples=128

.PHONY: profile probe solve baseline figures clean

# poseidon-probe adds each site's condition number (the Kappa line) to the profile.
profile: $(SRC)
	@echo "=== Profile-gen ($(PROF_DEFS)) ==="
	bash $(ARTIFACT_ROOT)/scripts/pt.sh profile_compile chase -- $(POSEIDON_CLANGXX) $(CUDA_FLAGS) $(PROF_DEFS) -poseidon-profile-generate $(SRC) -o chase_profiled.exe
	rm -rf fpprofile
	PT_PROBE_SPLIT=1 bash $(ARTIFACT_ROOT)/scripts/pt.sh profile_run chase -- env CUDA_VISIBLE_DEVICES=$(GPU_ID) POSEIDON_PROFILE_DIR=./fpprofile $(PROBE_TOOL) -- ./chase_profiled.exe --profiling
	@ls fpprofile
	@echo "--- Kappa per site ---"; grep -H "^Kappa" fpprofile/*.fpprofile
	@grep -q "^Kappa =" fpprofile/*.fpprofile || { echo "*** make profile: no profile has a measured Kappa"; exit 1; }

probe: fpprofile
	@echo "--- Kappa per site ---"; grep -H "^Kappa" fpprofile/*.fpprofile

# Writes the frontier's shares to cache/budgets.txt once; delete it when the profile changes.
solve: $(SRC) $(PROFDIR)
	@echo "=== Solve (BUDGET=$(BUDGET), $(BENCH_DEFS)) ==="
	@# A leftover dispatch descriptor would redirect a site this solve did not raise.
	@rm -f cache/*.ozdispatch
	@rm -f chase_opt.exe
	bash $(ARTIFACT_ROOT)/scripts/pt.sh solve_compile "chase b=$(BUDGET)" -- $(POSEIDON_CLANGXX) $(CUDA_FLAGS) $(BENCH_DEFS) \
	    -poseidon-profile-use=$(PROFDIR) \
	    -poseidon-enable-herbie=0 \
	    -poseidon-enable-pt=0 \
	    -poseidon-cost-budget=$(BUDGET) \
	    -poseidon-cache=cache \
	    -poseidon-report-path=reports \
	    -poseidon-cost-model=$(COST_MODEL) \
	    -poseidon-sample-log-bits=$(SLOGBITS) \
	    -poseidon-loose-coverage \
	    -poseidon-raise-wmma \
	    -poseidon-print \
	    $(JOINT_FLAGS) $(EXTRA_SOLVE) \
	    $(SRC) -o chase_opt.exe 2>&1 | tee solve.log \
	    | grep -iE "applying solution|pareto|matmul|wmma|ozaki|budget|error|joint" || true
	@# The pipe hides the compiler's status: a fresh chase_opt.exe is the proof of success.
	@test -s chase_opt.exe || { \
	  echo ""; \
	  echo "*** make solve FAILED: the compiler produced no chase_opt.exe."; \
	  echo "*** Full compiler output is in ./solve.log; first errors:"; \
	  grep -i -m5 error solve.log 2>/dev/null || echo "    (nothing matching 'error' -- read solve.log)"; \
	  exit 1; \
	}
	@if [ -f cache/budgets.txt ]; then echo "Budgets: $$(cat cache/budgets.txt)"; fi

baseline: $(SRC)
	bash $(ARTIFACT_ROOT)/scripts/pt.sh baseline_compile chase -- $(POSEIDON_CLANGXX) $(CUDA_FLAGS) $(BENCH_DEFS) $(SRC) -o chase_baseline.exe

figures:
	mkdir -p ../../figures
	$(PY) plot_chase.py --out ../../figures/rchfsi_residuals.pdf

clean:
	rm -f *.exe solve.log strategies.tsv measure_pareto_*.tsv \
	      guided.tsv blind.tsv ablation.tsv
	rm -rf fpprofile fpprofile_blind cache reports tmp logs traces __pycache__ *.nsys-rep *.sqlite
	@echo "kept: recorded/, profiles/"
