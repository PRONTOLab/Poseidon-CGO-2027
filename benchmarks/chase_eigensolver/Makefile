# chase_eigensolver — R-ChFSI dense Hermitian eigensolver, eval §5.3. A faithful
# port of the residual-based Chebyshev-filtered subspace iteration (chase_real.cu).
# Its four matrix products (filter H·Z, residual R=HV−VΛ, Gram VᵀV, Rayleigh–Ritz
# VᵀHV — five annotated launches: hemm/resid/gram/rrhv/rrproj) are each written as
# a scalar reduction loop and carries POSEIDON_OPTIMIZE. Poseidon extracts each
# as an FP subgraph, proposes direct tensor-core raises / TF32TCEC / Ozaki-II, and
# solves ONE joint DP over all five sites under a shared cost budget. The
# profiling run measures per-site how far the declared quantity of interest moves
# when a site is computed at a given relative precision, and the solver weighs
# that against each rewrite's local error, so only the washed-out filter gets
# raised.
#
# Config (compiler, CUDA, GPU_ID, python, cost model, Ozaki-II runtime) comes from
# ../../config.mk; every variable is overridable on the command line. Profile at a
# small surrogate (N=512) and deploy large (N=2048), per SCVH.
#
#   make profile                       # profile-gen @ N=512 NEV=32 (writes Kappa too)
#   make solve  BUDGET=<share>         # profile-use + joint DP @ N=2048 NEV=64
#   make verify                        # run the optimized binary
#   make baseline                      # all-FP64 reference binary
#
# BUDGET is -poseidon-cost-budget, a share of the profiled cost (a frontier
# point's share is in cache/budgets.txt after any solve; the rows of
# ../../EXPECTED_RESULTS.md name theirs in ../../expected/picks.tsv).
# PROFDIR=profiles/surrogate512 solves from the shipped profile.

include ../../config.mk

PROF_DEFS  ?= -DCHASE_N=512 -DCHASE_NEV=32     # SCVH surrogate (small; Enzyme AD heap)
BENCH_DEFS ?= -DCHASE_N=2048 -DCHASE_NEV=64    # deploy size (paper figure)
BUDGET     ?= 0
# The sensitivity-blind arm points this at a copy of the profile with no Kappa
# lines; no trailing spaces, make would keep them in the value.
PROFDIR    ?= ./fpprofile
SLOGBITS   ?= 0
SRC        ?= chase_real.cu

# poseidon-clang++ supplies the two pass plugins, Poseidon's include directory,
# -ffp-contract=on, the profiler under -poseidon-profile-generate, and under
# -poseidon-profile-use the three GEMM runtimes plus the cuMpSGEMM backend the
# error-corrected family delegates to (the solver may pick any of the three on a
# chase site; they export disjoint symbols and only a picked path is called).
#  -fcuda-rdc : the annotated kernels are compiled as relocatable device code.
CUDA_FLAGS = -x cuda --cuda-gpu-arch=$(GPU_ARCH) --cuda-path=$(CUDA) -fcuda-rdc \
    -O3 -I$(CUDA)/include -L$(CUDA)/lib64 -lcublas -lcusolver -lstdc++ -lm \
    -Wno-deprecated-declarations -Wno-unknown-cuda-version

# chase_real has five wrapped sites -> the solve is ONE joint DP over all of them.
# 128 MPFR samples (halved from 256) keeps the large-margin picks and ~2x the
# accuracy phase per budget.
JOINT_FLAGS ?= -poseidon-joint-dp -poseidon-num-samples=128

.PHONY: profile probe solve baseline verify figures quick run-all clean

# Step 1: profile-gen. The run supplies its own shadow buffers and, before it
# returns, measures each site's condition number against the quantity of
# interest the source declares with poseidon_metric. The WMMA reduction trip
# counts the solve reads back travel in the .fpprofile header the run writes.
profile: $(SRC)
	@echo "=== Profile-gen ($(PROF_DEFS)) ==="
	bash $(ARTIFACT_ROOT)/scripts/pt.sh profile_compile chase -- $(POSEIDON_CLANGXX) $(CUDA_FLAGS) $(PROF_DEFS) -poseidon-profile-generate $(SRC) -o chase_profiled.exe
	rm -rf fpprofile
	bash $(ARTIFACT_ROOT)/scripts/pt.sh profile_run chase -- env CUDA_VISIBLE_DEVICES=$(GPU_ID) POSEIDON_PROFILE_DIR=./fpprofile ./chase_profiled.exe --profiling
	@ls fpprofile
	@echo "--- Kappa per site ---"; grep -H "^Kappa" fpprofile/*.fpprofile

# The condition numbers are part of what `make profile` writes; kept as a target
# so the ladder and SCVH scripts keep running.
probe: fpprofile
	@echo "--- Kappa per site ---"; grep -H "^Kappa" fpprofile/*.fpprofile

# Step 2: profile-use + joint DP solve at BUDGET (writes the frontier's shares to
# cache/budgets.txt, which is write-once: delete it whenever the profile changes).
solve: $(SRC) $(PROFDIR)
	@echo "=== Solve (BUDGET=$(BUDGET), $(BENCH_DEFS)) ==="
	@# Stale host-dispatch descriptors from a PREVIOUS solve (dims baked at that
	@# solve's N) would silently redirect wrappers this config never raised --
	@# each solve must start with only its own dispatch state.
	@rm -f cache/*.ozdispatch
	@# Deleted BEFORE the compile so that its presence afterwards is proof
	@# this compile ran to completion, not a leftover from another budget.
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
	@# The pipeline above ends in `| tee | grep ... || true`, so its exit status
	@# is grep's, not the compiler's. Without this check a failed compile left
	@# make reporting SUCCESS, and the harness went on to record rows for a
	@# program that did not exist (all-FP64 results from a missing binary).
	@# This assertion is what makes `make solve` exiting 0 without a binary
	@# impossible; it costs nothing when the compile succeeds.
	@test -s chase_opt.exe || { \
	  echo ""; \
	  echo "*** make solve FAILED: the compiler produced no chase_opt.exe."; \
	  echo "*** Full compiler output is in ./solve.log; first errors:"; \
	  grep -i -m5 error solve.log 2>/dev/null || echo "    (nothing matching 'error' -- read solve.log)"; \
	  exit 1; \
	}
	@if [ -f cache/budgets.txt ]; then echo "Budgets: $$(cat cache/budgets.txt)"; fi

# All-FP64 reference (poseidon-clang++ without a Poseidon action: every site runs as written).
baseline: $(SRC)
	bash $(ARTIFACT_ROOT)/scripts/pt.sh baseline_compile chase -- $(POSEIDON_CLANGXX) $(CUDA_FLAGS) $(BENCH_DEFS) $(SRC) -o chase_baseline.exe

verify: chase_opt.exe
	CUDA_VISIBLE_DEVICES=$(GPU_ID) ./chase_opt.exe

## Regenerate the paper figure from whatever traces are present (fresh or golden).
figures:
	mkdir -p ../../figures
	$(PY) plot_chase.py --out ../../figures/rchfsi_residuals.pdf

## Kick-the-tires: profile @512, auto-probe, one deploy solve + verify.
quick:
	bash quick.sh

## Full paper pipeline (guided ladder + blind ablation + figure).
run-all:
	bash run_all.sh

# clean intentionally LEAVES golden/. Delete cache/budgets.txt so a changed
# profile re-enumerates DP breakpoints.
clean:
	rm -f *.exe solve.log chase_money_data.tsv measure_pareto_*.tsv \
	      scvh_probe_*.tsv ablation_knee.tsv
	rm -rf fpprofile fpprofile_blind cache reports tmp logs traces __pycache__ *.nsys-rep *.sqlite
	@echo "kept: golden/, profiles/"
