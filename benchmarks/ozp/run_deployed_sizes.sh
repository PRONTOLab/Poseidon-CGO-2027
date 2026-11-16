#!/usr/bin/env bash
# Sec. 4.3: the strategy selected from the N = 512 profile at the deployed sizes -> surrogate.tsv.
set -e
source "$(dirname "$0")/scripts/common.sh"
bash scripts/gen_data.sh
bash scripts/run_surrogate.sh
