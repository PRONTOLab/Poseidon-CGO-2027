#!/bin/bash
# profile_only.sh [PT_TAG]: the N = PROF_N profile in fpprofile/ (also run by the cost-model bootstrap).
# PT_TAG records the phases in overhead.tsv.
set -e
source "$(dirname "$0")/common.sh"
PT_TAG="${1:-}"
run_phase () {  # <phase> -- <cmd...>
  local phase="$1"; shift; [ "$1" = "--" ] && shift
  if [ -n "$PT_TAG" ]; then bash ../../scripts/pt.sh "$phase" "$PT_TAG" -- "$@"; else "$@"; fi
}

bash "$(dirname "$0")/gen_data.sh" "$PROF_N"

echo "### profile-gen @N=$PROF_N ###"
run_phase profile_compile -- "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${POSE[@]}" -DOZP_N=$PROF_N \
  -poseidon-profile-generate ozp_poseidon.cu -o "$TMPD/ozp_prof.exe"
rm -rf fpprofile
run_phase profile_run -- env CUDA_VISIBLE_DEVICES=$GPU_ID POSEIDON_PROFILE_DIR=./fpprofile \
  "$TMPD/ozp_prof.exe" data/synth${PROF_N}_h.mtx data/synth${PROF_N}_k.mtx --profiling

[ -f fpprofile/preprocess__Z10matmul_optPdPKdS1__poseidon_body.fpprofile ] || {
  echo "profile_only: fpprofile/ has no preprocess__Z10matmul_optPdPKdS1__poseidon_body.fpprofile;" >&2
  echo "profile_only: the in-kernel calibration cannot source a profile from it." >&2
  exit 1; }
echo "### profile in $(pwd)/fpprofile ###"
