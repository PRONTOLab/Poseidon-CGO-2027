#!/bin/bash
# gen_data.sh [N ...]: deterministic H and exact projector K_ref inputs in data/ (gen_mtx.py).
# Default sizes: OZP_DATA_SIZES, else 512 1024 2048 4096 8192 (N = 8192 takes 2.2 GB).
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
