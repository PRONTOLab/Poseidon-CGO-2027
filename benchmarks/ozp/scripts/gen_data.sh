#!/bin/bash
# Generate the synthetic gapped Hamiltonians H and their exact projectors K_ref
# (the answer key) that the ozp pipeline consumes. gen_mtx.py is fully
# deterministic (numpy default_rng(12345), fixed gap 0.10), so data/synth<N>_*.mtx
# is byte-reproducible; the giant matrices are therefore regenerated here rather
# than checked into the artifact. Each N is skipped if both files already exist.
#   Usage: bash scripts/gen_data.sh [N ...]   (default: 512 1024 2048 4096 8192)
#          OZP_DATA_SIZES="512 1024 2048 4096" bash scripts/gen_data.sh
# The environment form exists because run_all.sh calls this script with no
# arguments, so the positional list is unreachable from the artifact driver's
# full mode; a site that cannot spend the wall clock (or the 4.4 GB) on N=8192
# would otherwise have no way to run full mode at all.
# Sizes and their on-disk footprint: 512~7.7M 1024~31M 2048~130M 4096~0.53G 8192~2.2G.
source "$(dirname "$0")/common.sh"
mkdir -p data
SIZES="${*:-${OZP_DATA_SIZES:-512 1024 2048 4096 8192}}"
for N in $SIZES; do
  if [ -s "data/synth${N}_h.mtx" ] && [ -s "data/synth${N}_k.mtx" ]; then
    echo "[gen_data] data/synth${N}_{h,k}.mtx present, skipping"
    continue
  fi
  echo "[gen_data] generating N=$N ..."
  "$PY" gen_mtx.py "$N" "data/synth${N}"
done
