# config.mk — shared configuration for all Poseidon-GPU artifact benchmarks.
#
# Nothing here is machine-specific. Every value is resolved by
# scripts/detect.sh with this precedence:
#   command line / environment > local.env (user pins) > cost_models/target.env
#   (written by scripts/bootstrap_cost_model.sh) > auto-detection
# so `make GPU_ID=0`, exported variables, and a local.env pin all behave the
# same for make targets and for the pipeline scripts (which source env.sh, the
# shell twin of this file).
#
# detect.sh finds: the Poseidon source tree of the build (POSEIDON_BUILD), the
# driver and calibration tool of that build, the plugin-ABI-matching clang++
# (from the build's CMakeCache), the CUDA toolkit (CUDA_HOME-aware for module
# systems), the GPU arch/id (nvidia-smi; Slurm's CUDA_VISIBLE_DEVICES is
# respected), a python with numpy+matplotlib, and the cost model for this
# device (the shipped RTX 5090 model on an sm_120 GPU, else the one
# scripts/bootstrap_cost_model.sh measured; never another device's).

ARTIFACT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
DETECTED_MK   := $(ARTIFACT_ROOT)/.detected.mk

# Regenerate the fragment on every make; on detection failure the previous
# fragment (if any) is kept so already-configured setups keep working.
$(shell bash $(ARTIFACT_ROOT)/scripts/detect.sh mkfrag > $(DETECTED_MK).$$$$ \
        && mv $(DETECTED_MK).$$$$ $(DETECTED_MK) \
        || rm -f $(DETECTED_MK).$$$$)
-include $(DETECTED_MK)

ifeq ($(strip $(CLANG)),)
$(error artifact configuration failed - see the detect.sh message above; set the missing variable in the environment or in local.env (README: Configuration))
endif
# A benchmark that never invokes the pass (fig2_drift) sets NO_COST_MODEL := 1.
ifeq ($(strip $(COST_MODEL))$(NO_COST_MODEL),)
$(error no cost model for this device ($(GPU_ARCH)); measure it first with ./run_artifact.sh --bootstrap-only, or set COST_MODEL=<csv measured on this device>)
endif

export POSEIDON_ROOT CLANG CUDA GPU_ARCH GPU_ID PY POSEIDON_CLANGXX \
       POSEIDON_CALIBRATE POSEIDON_INC OZAKI_RT TCEC_RT DIRECT_RT COST_MODEL
