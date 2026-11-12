#!/usr/bin/env bash
# smoke.sh <out dir>: minutes-long checks on the GPU before the long run.
#
#   1. nvidia-smi: the device, its compute capability and the driver
#   2. the build's CUDA architectures cover this device
#   3. plain compile + run of one annotated kernel through poseidon-clang++
#   4. profile generation: compile + run (writes a profile)
#   5. a per-op cost model of this device (poseidon-calibrate --only microbm)
#   6. profile use against that model: compile + run, same printed result
#   7. Herbie runs (improve --help) and has --num-taylor-terms
#   8. the artifact's own preflight (run_artifact.sh --check)
#
# SMOKE_NO_GPU=1 does the compile steps only (3, 4 compile, 7, and the
# artifact's detection without nvidia-smi): for testing the container without
# a GPU. Exit status: the number of failed steps.
set -uo pipefail
OUT="${1:?usage: smoke.sh <out dir>}"
TC="${TC:-/toolchain}"
BUNDLE="${BUNDLE:-/bundle}"
CUDA="${CUDA:-/usr/local/cuda}"
NO_GPU="${SMOKE_NO_GPU:-0}"
BUILD="$TC/poseidon-build"
PCXX="$BUILD/Poseidon/bin/poseidon-clang++"
PCAL="$BUILD/Poseidon/bin/poseidon-calibrate"
mkdir -p "$OUT"
W="$(mktemp -d /tmp/smoke.XXXXXX)"
cd "$W"
FAILS=0
step() { echo; echo "==== smoke: $* ===="; }
ok()   { echo "[smoke] OK: $*"; }
bad()  { echo "[smoke] FAILED: $*"; FAILS=$((FAILS + 1)); }

for f in "$TC/poseidon.stamp" "$PCXX" "$PCAL"; do
  [ -e "$f" ] || { echo "[smoke] $f is missing: run the build stage first"; exit 99; }
done
case "$(cat "$TC/poseidon.stamp")" in
  "src=$(sha256sum "$BUNDLE/poseidon-src.tar.gz" | cut -d' ' -f1) "*) ;;
  *) echo "[smoke] the toolchain was built from another poseidon-src.tar.gz: run the build stage first"; exit 99 ;;
esac
echo "[smoke] toolchain: llvm [$(cat "$TC/llvm.stamp" 2>/dev/null || echo external)]"
echo "[smoke] poseidon: $(cat "$TC/poseidon.stamp")"
echo "[smoke] herbie:   $(cat "$TC/herbie.stamp" 2>/dev/null || echo 'not built')"

step "1. device"
ARCH=""
if [ "$NO_GPU" = 1 ]; then
  ARCH="sm_$(sed -n 's/^CMAKE_CUDA_ARCHITECTURES:[A-Z]*=\([0-9]*\).*/\1/p' "$BUILD/CMakeCache.txt")"
  echo "[smoke] SMOKE_NO_GPU=1: compiling for $ARCH, nothing runs"
else
  if nvidia-smi | tee "$OUT/nvidia-smi.txt"; then
    nvidia-smi -q > "$OUT/nvidia-smi-q.txt" 2>&1 || true
    CC="$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
    ARCH="sm_${CC/./}"
    ok "$(nvidia-smi -i 0 --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader)"
  else
    bad "nvidia-smi"; echo "[smoke] no GPU visible: nothing else can be checked"; exit 99
  fi
fi

step "2. CUDA architectures of the build"
BUILT="$(sed -n 's/^CMAKE_CUDA_ARCHITECTURES:[A-Z]*=//p' "$BUILD/CMakeCache.txt")"
case ";$BUILT;" in
  *";${ARCH#sm_}-real;"*|*";${ARCH#sm_};"*|*";${ARCH#sm_}-virtual;"*) ok "runtimes and calibrators built for $BUILT, device $ARCH" ;;
  *) bad "runtimes and calibrators built for '$BUILT' but the device is $ARCH: rebuild with CUDA_ARCHS=${ARCH#sm_}-real" ;;
esac

cat > smoke.cu <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
#include "poseidon/poseidon.h"

// The kernel of enzyme/test/Poseidon/Integration/annotated_kernel_profile.cu
// with a little more arithmetic per element.
POSEIDON_OPTIMIZE __global__ void axpb(double *o, const double *a,
                                       const double *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double x = a[i], y = b[i];
    o[i] = (x * y + x) / (1.0 + y * y) + x * x * 0.5;
  }
}

int main() {
  const int n = 1 << 16;
  double *o, *a, *b;
  cudaMalloc(&o, n * sizeof(double));
  cudaMalloc(&a, n * sizeof(double));
  cudaMalloc(&b, n * sizeof(double));
  static double h[n];
  for (int i = 0; i < n; ++i) h[i] = 1.0 + i * 1e-4;
  cudaMemcpy(a, h, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(b, h, n * sizeof(double), cudaMemcpyHostToDevice);
  axpb<<<(n + 255) / 256, 256>>>(o, a, b, n);
  cudaError_t e = cudaDeviceSynchronize();
  cudaMemcpy(h, o, n * sizeof(double), cudaMemcpyDeviceToHost);
  double s = 0;
  for (int i = 0; i < n; ++i) s += h[i];
  printf("status %d sum %.6e\n", (int)e, s);
  return e == cudaSuccess ? 0 : 1;
}
EOF
CFLAGS=( -x cuda --cuda-gpu-arch="$ARCH" --cuda-path="$CUDA" -O2 )

step "3. plain compile (+ run)"
if "$PCXX" "${CFLAGS[@]}" smoke.cu -o plain.exe 2>&1 | tee plain.log; [ -x plain.exe ]; then
  ok "plain compile"
  if [ "$NO_GPU" != 1 ]; then
    if ./plain.exe | tee plain.out; then ok "plain run: $(cat plain.out)"; else bad "plain run"; fi
  fi
else
  bad "plain compile"
fi

step "4. profile generation: compile (+ run)"
if "$PCXX" "${CFLAGS[@]}" -poseidon-profile-generate smoke.cu -o prof.exe 2>&1 | tee prof.log; [ -x prof.exe ]; then
  ok "profile-generate compile"
  if [ "$NO_GPU" != 1 ]; then
    if POSEIDON_PROFILE_DIR="$W/profile" ./prof.exe | tee prof.out && ls "$W"/profile/*.fpprofile >/dev/null 2>&1; then
      ok "profiling run: $(cat prof.out); $(ls "$W"/profile | tr '\n' ' ')"
      cp -r "$W/profile" "$OUT/smoke-profile"
    else
      bad "profiling run (no .fpprofile written)"
    fi
  fi
else
  bad "profile-generate compile"
fi

if [ "$NO_GPU" != 1 ]; then
  step "5. per-op cost model of this device"
  CM="$W/cm/cm_${ARCH}_smoke.csv"
  mkdir -p "$W/cm"
  T0=$(date +%s)
  if "$PCAL" --gpu 0 --only microbm --out "$CM" 2>&1 | tail -20 && [ -s "$CM" ]; then
    ok "poseidon-calibrate --only microbm in $(( $(date +%s) - T0 )) s: $(grep -c , "$CM") rows"
    cp "$CM" "$OUT/"
  else
    bad "poseidon-calibrate --only microbm"
  fi

  step "6. profile use: compile + run"
  if [ -s "$CM" ] && [ -d "$W/profile" ] &&
     "$PCXX" "${CFLAGS[@]}" -poseidon-profile-use="$W/profile" -poseidon-tau=1e-6 \
        -poseidon-cost-model="$CM" -poseidon-enable-herbie=0 -poseidon-cache="$W/cache" \
        smoke.cu -o opt.exe 2>&1 | tee opt.log | grep -E '\[poseidon\]' | tail -20; [ -x opt.exe ]; then
    ok "profile-use compile"
    if ./opt.exe | tee opt.out; then ok "optimized run: $(cat opt.out) (plain: $(cat plain.out 2>/dev/null))"; else bad "optimized run"; fi
  else
    bad "profile-use compile (see opt.log)"
  fi
fi

step "7. Herbie"
if [ -x "$TC/herbie/bin/herbie" ]; then
  if "$TC/herbie/bin/herbie" improve --help > herbie.help 2>&1; then
    if grep -q -- '--num-taylor-terms' herbie.help; then ok "herbie runs, has --num-taylor-terms"
    else bad "herbie has no --num-taylor-terms"; fi
  else
    bad "herbie improve --help"; tail -5 herbie.help
  fi
else
  echo "[smoke] note: no Herbie built (not needed by the replay run; the full on-device flow needs it for GENGA)"
fi

step "8. the artifact's preflight"
mkdir -p "$W/art" && tar -C "$W/art" -xzf "$BUNDLE/artifact.tar.gz"
export POSEIDON_BUILD="$BUILD"
if [ "$NO_GPU" = 1 ]; then
  if ( cd "$W/art/cgo2027" && POSEIDON_ARTIFACT_NO_GPU=1 GPU_ARCH="$ARCH" bash scripts/detect.sh export ) | tee preflight.out; then
    ok "detect.sh export (no GPU)"
  else
    bad "detect.sh export"
  fi
else
  if ( cd "$W/art/cgo2027" && GPU_ID=0 ./run_artifact.sh --check ) 2>&1 | tee preflight.out; then
    ok "run_artifact.sh --check"
  else
    bad "run_artifact.sh --check"
  fi
fi

cp -f "$W"/*.log "$W"/*.out "$OUT/" 2>/dev/null || true
echo
if [ "$FAILS" = 0 ]; then echo "==== smoke: ALL OK ===="; else echo "==== smoke: $FAILS step(s) FAILED ===="; fi
exit "$FAILS"
