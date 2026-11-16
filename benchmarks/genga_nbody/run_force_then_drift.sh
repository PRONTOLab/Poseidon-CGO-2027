#!/usr/bin/env bash
# Sec. 4.2: the force-then-drift program in one solve -> unified.tsv.
set -e
source "$(dirname "$0")/scripts/common.sh"
[ "$GENGA_HERBIE_USABLE" = 1 ] || { herbie_skip_note "the force-then-drift solve"; exit 0; }
bash scripts/run_unified.sh
