#!/usr/bin/env bash
# env.sh: source it to export the artifact configuration that scripts/detect.sh resolves.
# POSEIDON_ARTIFACT_NO_MODEL=1 accepts a device with no cost model yet (setup check, bootstrap).
_POSEIDON_ART="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_POSEIDON_ENV_OUT="$(bash "$_POSEIDON_ART/scripts/detect.sh" export)" || {
  echo "env.sh: artifact configuration failed (see detect.sh message above)" >&2
  unset _POSEIDON_ART _POSEIDON_ENV_OUT
  return 1 2>/dev/null || exit 1
}
eval "$_POSEIDON_ENV_OUT"
if [ -z "$COST_MODEL" ] && [ "${POSEIDON_ARTIFACT_NO_MODEL:-0}" != 1 ]; then
  echo "env.sh: no cost model for this device ($GPU_ARCH). The shipped model is the" >&2
  echo "env.sh: RTX 5090's; measure this device first: ./run_artifact.sh --bootstrap-only" >&2
  echo "env.sh: (or set COST_MODEL=<csv> to a model measured on this device)." >&2
  unset _POSEIDON_ART _POSEIDON_ENV_OUT
  return 1 2>/dev/null || exit 1
fi
unset _POSEIDON_ART _POSEIDON_ENV_OUT
