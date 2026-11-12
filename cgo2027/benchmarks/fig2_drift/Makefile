# fig2_drift — intro motivating figure (paper Fig. 2 / img/intro_drift).
#
# Hand-written precision comparison: a long leapfrog integration run in FP64,
# naive FP32, and 3xFP32 MultiFloat emulation, against a 2xFP64 reference with
# the SAME dt (identical truncation, so divergence is pure accumulated rounding).
# This benchmark does NOT invoke the Poseidon pass; it motivates the emulation
# axis by measuring the accuracy AND wall-clock cost of each hand-coded variant.
#
# Requires only CLANG + CUDA from config.mk (no Enzyme plugin, no profiler, no
# cost model).

NO_COST_MODEL := 1
include ../../config.mk

SRC      := fig2_drift.cu
EXE      := fig2_drift.exe
FIGURES  := ../../figures
PLOT     := plot_drift.py

# -ffp-contract=on (NOT the CUDA default 'fast'): MultiFloat's double-single
# error-free transforms rely on an un-fused a*b then subtract; 'fast' would
# contract them into an FMA and silently collapse 3xFP32 to plain float.
CUDA_FLAGS := -x cuda --cuda-gpu-arch=$(GPU_ARCH) --cuda-path=$(CUDA) \
              -O3 -std=c++20 -ffp-contract=on \
              -L$(CUDA)/lib64 -lcudart -lstdc++ -lm

.PHONY: all build run plot clean

all: plot

build: $(EXE)

$(EXE): $(SRC) mX_real/mX_real.hpp
	$(CLANG) $(CUDA_FLAGS) $(SRC) -o $(EXE)

# Runs the timing + drift sweep on GPU $(GPU_ID); writes drift.csv and run.log.
# GPU timings are the average of 5 kernel launches (CUDA events); CPU timings
# average 3 single-thread runs (see fig2_drift.cu).
run: $(EXE)
	@# drift.csv is deleted first so its presence below is proof THIS run wrote
	@# it. The run is piped into tee, so the pipeline's exit status is tee's, not
	@# the benchmark's: without this check an aborted run (CUDA error, OOM, a
	@# killed process) reported success and `plot` then regenerated the paper
	@# figure from the PREVIOUS run's drift.csv, silently.
	@rm -f drift.csv
	CUDA_VISIBLE_DEVICES=$(GPU_ID) ./$(EXE) 2>&1 | tee run.log
	@test -s drift.csv || { \
	  echo ""; \
	  echo "*** make run FAILED: $(EXE) wrote no drift.csv."; \
	  echo "*** Full output is in ./run.log; last lines:"; \
	  tail -5 run.log 2>/dev/null; \
	  exit 1; \
	}

# Regenerates the paper figure into ../../figures/intro_drift.pdf from drift.csv.
plot: run
	mkdir -p $(FIGURES)
	$(PY) $(PLOT)

clean:
	rm -f $(EXE) run.log drift.csv
