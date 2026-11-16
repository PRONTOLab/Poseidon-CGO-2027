# config.mk: included by every benchmark Makefile; values come from scripts/detect.sh.

ARTIFACT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
DETECTED_MK   := $(ARTIFACT_ROOT)/.detected.mk

# On a detection failure the previous fragment is kept.
$(shell bash $(ARTIFACT_ROOT)/scripts/detect.sh mkfrag > $(DETECTED_MK).$$$$ \
        && mv $(DETECTED_MK).$$$$ $(DETECTED_MK) \
        || rm -f $(DETECTED_MK).$$$$)
-include $(DETECTED_MK)

ifeq ($(strip $(CLANG)),)
$(error artifact configuration failed - see the detect.sh message above; set the missing variable in the environment or in local.env (./run_artifact.sh -h))
endif
ifeq ($(strip $(COST_MODEL)),)
$(error no cost model for this device ($(GPU_ARCH)); measure it first with ./run_artifact.sh --bootstrap-only, or set COST_MODEL=<csv measured on this device>)
endif

export POSEIDON_ROOT CLANG CUDA GPU_ARCH GPU_ID PY POSEIDON_CLANGXX \
       POSEIDON_CALIBRATE POSEIDON_INC OZAKI_RT TCEC_RT DIRECT_RT COST_MODEL
