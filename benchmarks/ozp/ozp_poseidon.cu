// ozp_poseidon.cu — ozp density-matrix purification (TRS4, the recurrence ozp's
// `ref`/`ozaki_gpu` actually use) with the naive F64 GEMM wrapped through
// Poseidon (POSEIDON_OPTIMIZE), so Poseidon can auto-rewrite the matmul to a
// tensor-core Ozaki scheme.
//
// Faithful to ozp's trs4 (include/purify_imp.h): Gershgorin init
// X = (h_max I - H)/(h_max - h_min), then per iteration
//   X2 = X@X ;  fx = 4X-3X2 ;  gx = X2-2X+I ;
//   sigma = (nel - tr(X2·fx))/tr(X2·gx)   [trace resetting]
//   X <- {2X-X2 | X2 | X2·(fx+sigma·gx)}  by sigma branch ;  symmetrize.
// The TWO GEMMs (X@X and X2·g) go through matmul_opt (Poseidon target); the
// trace-resetting glue is cheap device kernels so the loop stays device-resident
// (no per-iter host copies to dilute the GEMM speedup). Reuses ozp's read_mm +
// get_eig_circle. Compile-time OZP_N must equal the loaded matrix size.
//
// argv: <H.mtx> <K.mtx> [nel] [--profiling]

#include "include/matrix_arithmetic.h" // get_eig_circle (clean host headers)
#include "include/matrix_helpers.h"    // read_mm
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

#include "poseidon/poseidon.h"

#ifndef OZP_N
#define OZP_N 2048
#endif
#ifndef OZP_MAXIT
#define OZP_MAXIT 30
#endif
#define N OZP_N
#define NN ((size_t)OZP_N * OZP_N)
static const int N_PROFILE_ITERS = 2;
static const int WARMUP = 2;
static const int BENCH_ITERS = 5;
static const double TRS_SMIN = 0.0, TRS_SMAX = 6.0, TRS_CONV = 1e-8;
typedef double dfloat;

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,                     \
             cudaGetErrorString(e));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ── Naive F64 GEMM body: C = A*B, row-major, K-strided B (the Poseidon site).
__device__ void
matmul_body(dfloat *__restrict__ C, const dfloat *__restrict__ A,
            const dfloat *__restrict__ B) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= N || col >= N)
    return;
  const dfloat *arow = A + (size_t)row * N;
  dfloat val = 0.0;
#pragma unroll 1
  for (int k = 0; k < N; ++k)
    val += arow[k] * B[(size_t)k * N + col];
  C[(size_t)row * N + col] = val;
}
// Naive F64 GEMM kernel — baseline and accuracy reference. Always built.
__global__ void matmul_gold(dfloat *C, const dfloat *A, const dfloat *B) {
  matmul_body(C, const_cast<dfloat *>(A), const_cast<dfloat *>(B));
}

// Compile-time GEMM selection, mirroring the ~/Poseidon #ifdef convention:
//   default build    → matmul_gold (naive F64; baseline + reference)
//   -DPOSEIDON build → matmul_opt  (the annotated Poseidon site)
// The binary is one mode, chosen at compile time — no runtime branch. The
// baseline-vs-Poseidon wall-clock comparison is done across the two binaries by
// the harness (run_poseidon.sh), not inside one binary.
#ifdef POSEIDON
POSEIDON_OPTIMIZE __global__ void matmul_opt(dfloat *C, const dfloat *A,
                                             const dfloat *B) {
  matmul_body(C, A, B);
}
#define GEMM(C, A, B) matmul_opt<<<GG, GB>>>(C, A, B)
#elif defined(OZP_CUBLAS_TC)
// Library-dispatch reference for the direct tensor-core raises: the same
// operand precision as the in-kernel WMMA raise (FP64 operands rounded to
// FP16 / BF16 / TF32, FP32 accumulation, FP64 result), but the product runs as
// one cuBLAS GEMM instead of inside ozp's own loop nest. Not a Poseidon
// candidate; it bounds what a direct raise can gain from a library. The
// narrow / widen passes are part of the timed region, as they are in-kernel.
//   OZP_CUBLAS_TC = 1: FP16 inputs      (CUDA_R_16F,  CUBLAS_COMPUTE_32F)
//   OZP_CUBLAS_TC = 2: BF16 inputs      (CUDA_R_16BF, CUBLAS_COMPUTE_32F)
//   OZP_CUBLAS_TC = 3: TF32 (FP32 inputs, CUBLAS_COMPUTE_32F_FAST_TF32)
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if OZP_CUBLAS_TC == 1
typedef __half tc_in_t;
#define TC_IN_DT CUDA_R_16F
#define TC_COMPUTE CUBLAS_COMPUTE_32F
__device__ inline tc_in_t tc_narrow(dfloat x) { return __double2half(x); }
#elif OZP_CUBLAS_TC == 2
typedef __nv_bfloat16 tc_in_t;
#define TC_IN_DT CUDA_R_16BF
#define TC_COMPUTE CUBLAS_COMPUTE_32F
__device__ inline tc_in_t tc_narrow(dfloat x) { return __double2bfloat16(x); }
#elif OZP_CUBLAS_TC == 3
typedef float tc_in_t;
#define TC_IN_DT CUDA_R_32F
#define TC_COMPUTE CUBLAS_COMPUTE_32F_FAST_TF32
__device__ inline tc_in_t tc_narrow(dfloat x) { return (float)x; }
#else
#error "OZP_CUBLAS_TC must be 1 (FP16), 2 (BF16) or 3 (TF32)"
#endif
#define CB(x)                                                                  \
  do {                                                                         \
    cublasStatus_t st = (x);                                                   \
    if (st != CUBLAS_STATUS_SUCCESS) {                                         \
      printf("cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)st);         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
__global__ void tc_narrow_k(tc_in_t *dst, const dfloat *src, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = tc_narrow(src[i]);
}
__global__ void tc_widen_k(dfloat *dst, const float *src, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = (dfloat)src[i];
}
static cublasHandle_t g_tc_h = nullptr;
static tc_in_t *g_tc_a = nullptr, *g_tc_b = nullptr;
static float *g_tc_c = nullptr;
static void gemm_cublas_tc(dfloat *C, const dfloat *A, const dfloat *B) {
  if (!g_tc_h) {
    CB(cublasCreate(&g_tc_h));
    CK(cudaMalloc(&g_tc_a, NN * sizeof(tc_in_t)));
    CK(cudaMalloc(&g_tc_b, NN * sizeof(tc_in_t)));
    CK(cudaMalloc(&g_tc_c, NN * sizeof(float)));
  }
  const dim3 g((NN + 255) / 256), blk(256);
  tc_narrow_k<<<g, blk>>>(g_tc_a, A, NN);
  tc_narrow_k<<<g, blk>>>(g_tc_b, B, NN);
  const float alpha = 1.0f, beta = 0.0f;
  // Row-major C = A*B is column-major C^T = B^T * A^T: a plain NN GEMM with the
  // operands swapped.
  CB(cublasGemmEx(g_tc_h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, g_tc_b,
                  TC_IN_DT, N, g_tc_a, TC_IN_DT, N, &beta, g_tc_c, CUDA_R_32F,
                  N, TC_COMPUTE, CUBLAS_GEMM_DEFAULT));
  tc_widen_k<<<g, blk>>>(C, g_tc_c, NN);
}
#define GEMM(C, A, B) gemm_cublas_tc(C, A, B)
#else
#define GEMM(C, A, B) matmul_gold<<<GG, GB>>>(C, A, B)
#endif

// ── Cheap device glue for the TRS4 recurrence. ──
__global__ void combine2(dfloat *out, const dfloat *A, const dfloat *B,
                         dfloat a, dfloat b) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NN)
    return;
  out[i] = a * A[i] + b * B[i];
}
__global__ void add_diag(dfloat *M, dfloat c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;
  M[(size_t)i * N + i] += c;
}
__global__ void symmetrize(dfloat *P) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= N || j >= N || i >= j)
    return;
  dfloat a = 0.5 * (P[(size_t)i * N + j] + P[(size_t)j * N + i]);
  P[(size_t)i * N + j] = a;
  P[(size_t)j * N + i] = a;
}
__global__ void pdot_kernel(const dfloat *A, const dfloat *B, dfloat *res,
                            size_t nn) {
  __shared__ dfloat sh[256];
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  sh[threadIdx.x] = (i < nn) ? A[i] * B[i] : 0.0;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < (unsigned)s)
      sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    atomicAdd(res, sh[0]);
}

static dim3 G2((NN + 255) / 256), B2(256); // flat NN launch
static dim3 GG((N + 15) / 16, (N + 15) / 16), GB(16, 16); // N×N launch

static double dpdot(const dfloat *dA, const dfloat *dB, dfloat *dScalar) {
  CK(cudaMemset(dScalar, 0, sizeof(dfloat)));
  pdot_kernel<<<G2, B2>>>(dA, dB, dScalar, NN);
  dfloat h;
  CK(cudaMemcpy(&h, dScalar, sizeof(dfloat), cudaMemcpyDeviceToHost));
  return h;
}

struct Bufs {
  dfloat *X, *X2, *fx, *gx, *g2, *tmp;
  dfloat *scal;
};

// ── Shared benchmark accuracy utility (mode-independent). ──
// L2 relative error of a result matrix vs the analytic reference projector Kref
// ("did the chemistry converge to the true projector"). One definition used by
// BOTH the final report and the per-iteration --trace trajectory, and identical
// in the baseline (matmul_gold) and Poseidon (matmul_opt) builds — so every
// optimized variant is scored by the exact same metric. Optionally also returns
// the max absolute error.
static double l2_rel_error(const double *X, const double *ref, size_t n,
                           double *max_abs_out = nullptr) {
  double sse = 0, ssg = 0, max_abs = 0;
  for (size_t i = 0; i < n; ++i) {
    double d = X[i] - ref[i];
    if (fabs(d) > max_abs)
      max_abs = fabs(d);
    sse += d * d;
    ssg += ref[i] * ref[i];
  }
  if (max_abs_out)
    *max_abs_out = max_abs;
  return ssg > 0 ? sqrt(sse / ssg) : 0.0;
}

// Density-matrix RMSD vs reference — the Dawson JCTC'24 domain metric:
// sqrt( mean_ij (X_ij - ref_ij)^2 ) over all N^2 elements. Unlike the relative
// L2 above, this is the absolute per-element deviation; the projector elements
// are O(1), so it is the system/unit-independent number their chemistry-
// sufficiency target (~1e-7, two orders below NWChem SCF defaults) is stated in.
static double rmsd_error(const double *X, const double *ref, size_t n) {
  double sse = 0;
  for (size_t i = 0; i < n; ++i) {
    double d = X[i] - ref[i];
    sse += d * d;
  }
  return sqrt(sse / (double)n); // n = N*N  ->  ||X-ref||_F / N
}

// One TRS4 purification, in place into b.X (already initialized to X0).
static void trs4(Bufs &b, const dfloat *dH, double target_trace, int iters) {
  for (int it = 0; it < iters; ++it) {
    // X2 = X @ X
    GEMM(b.X2, b.X, b.X);
    // fx = 4X - 3X2 ; gx = X2 - 2X + I
    combine2<<<G2, B2>>>(b.fx, b.X, b.X2, 4.0, -3.0);
    combine2<<<G2, B2>>>(b.gx, b.X2, b.X, 1.0, -2.0);
    add_diag<<<(N + 255) / 256, 256>>>(b.gx, 1.0);
    // trace-resetting coefficient
    double tfx = dpdot(b.X2, b.fx, b.scal);
    double tgx = dpdot(b.X2, b.gx, b.scal);
    double sigma = (fabs(tgx) < TRS_CONV / 10) ? 0.5 * (TRS_SMAX - TRS_SMIN)
                                               : (target_trace - tfx) / tgx;
    if (sigma > TRS_SMAX) {
      combine2<<<G2, B2>>>(b.tmp, b.X, b.X2, 2.0, -1.0); // X <- 2X - X2
    } else if (sigma < TRS_SMIN) {
      CK(cudaMemcpy(b.tmp, b.X2, NN * sizeof(dfloat), cudaMemcpyDeviceToDevice));
    } else {
      combine2<<<G2, B2>>>(b.g2, b.fx, b.gx, 1.0, sigma); // g2 = fx + sigma*gx
      GEMM(b.tmp, b.X2, b.g2);
    }
    CK(cudaMemcpy(b.X, b.tmp, NN * sizeof(dfloat), cudaMemcpyDeviceToDevice));
    symmetrize<<<GG, GB>>>(b.X);
  }
}

int main(int argc, char **argv) {
  bool profiling = false, trace = false;
  const char *hfile = nullptr, *kfile = nullptr;
  double nel = -1;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--profiling") == 0)
      profiling = true;
    else if (std::strcmp(argv[i], "--trace") == 0)
      trace = true;
    else if (!hfile)
      hfile = argv[i];
    else if (!kfile)
      kfile = argv[i];
    else
      nel = std::atof(argv[i]);
  }
  if (!hfile || !kfile) {
    printf("Usage: %s <H.mtx> <K.mtx> [nel] [--profiling] [--trace]\n", argv[0]);
    return 1;
  }
  if (nel < 0)
    nel = N / 2; // half filling default (synth data)

  CK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 4ULL << 30));

  unsigned int sz = 0, szk = 0;
  double *H = read_mm<double>(std::string(hfile), sz);
  double *Kref = read_mm<double>(std::string(kfile), szk);
  if (sz != (unsigned)N || szk != (unsigned)N) {
    printf("ERROR: matrix size %u != OZP_N=%d; recompile -DOZP_N=%u\n", sz, N, sz);
    return 1;
  }
  printf("ozp_poseidon (TRS4): N=%d MAXIT=%d nel=%.0f  H=%s\n", N, OZP_MAXIT, nel,
         hfile);

  // TRS4 init: X0 = (h_max I - H)/(h_max - h_min)  (ozp get_eig_circle).
  double h_min, h_max;
  get_eig_circle(H, N, h_min, h_max);
  std::vector<double> X0(NN);
  double inv = 1.0 / (h_max - h_min);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      X0[(size_t)i * N + j] =
          ((i == j ? h_max : 0.0) - H[(size_t)i * N + j]) * inv;

  Bufs b;
  dfloat *dX0, *dH;
  auto al = [&](dfloat **p) { CK(cudaMalloc(p, NN * sizeof(dfloat))); };
  al(&b.X); al(&b.X2); al(&b.fx); al(&b.gx); al(&b.g2); al(&b.tmp);
  al(&dX0); al(&dH);
  CK(cudaMalloc(&b.scal, sizeof(dfloat)));
  CK(cudaMemcpy(dX0, X0.data(), NN * sizeof(dfloat), cudaMemcpyHostToDevice));

  if (profiling) {
    CK(cudaMemcpy(b.X, dX0, NN * sizeof(dfloat), cudaMemcpyDeviceToDevice));
    trs4(b, dH, nel, N_PROFILE_ITERS);
    CK(cudaDeviceSynchronize());
    printf("Profiling done.\n");
    return 0;
  }

  // --trace: untimed convergence trajectory — residual (L2 vs analytic Kref) per
  // TRS4 iteration, for the thesis-figure convergence-trajectory plot. Kept off
  // the timed path so the per-iteration host copies don't dilute the GEMM
  // speedup. trs4 iterates in place, so stepping it one iteration at a time is
  // numerically identical to a single iters=OZP_MAXIT call. Same residual metric
  // (l2_rel_error) as the final report, so every variant's curve is comparable.
  if (trace) {
    std::vector<dfloat> snap(NN);
    CK(cudaMemcpy(b.X, dX0, NN * sizeof(dfloat), cudaMemcpyDeviceToDevice));
    printf("# iter  L2_rel_vs_Kref\n");
    for (int it = 1; it <= OZP_MAXIT; ++it) {
      trs4(b, dH, nel, 1);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(snap.data(), b.X, NN * sizeof(dfloat),
                    cudaMemcpyDeviceToHost));
      printf("iter %d residual %.6e\n", it, l2_rel_error(snap.data(), Kref, NN));
    }
    return 0;
  }

  // Single measured purification — the GEMM is matmul_gold (default build) or
  // matmul_opt (-DPOSEIDON), fixed at compile time. Accuracy is reported against
  // the analytic reference projector Kref (the "did the chemistry converge to the
  // true projector" metric); the baseline-vs-Poseidon speedup is computed across
  // the two binaries by run_poseidon.sh.
  std::vector<dfloat> h_result(NN);
  CK(cudaMemcpy(b.X, dX0, NN * sizeof(dfloat), cudaMemcpyDeviceToDevice));
  trs4(b, dH, nel, OZP_MAXIT);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h_result.data(), b.X, NN * sizeof(dfloat), cudaMemcpyDeviceToHost));

  double max_abs = 0;
  double l2 = l2_rel_error(h_result.data(), Kref, NN, &max_abs);
  printf("L2 relative error = %.6e\n", l2);
  printf("Max absolute error = %.6e\n", max_abs);
  // Domain-scientific accuracy metrics (Dawson JCTC'24): density-matrix RMSD
  // (sufficiency target ~1e-7) and band-structure energy error
  // |Tr((X - Kref) H)| with H symmetric (target ~1e-8 Hartree on real systems;
  // the synthetic H sets the energy scale here).
  printf("RMSD = %.6e\n", rmsd_error(h_result.data(), Kref, NN));
  double eX = 0, eR = 0;
  for (size_t i = 0; i < NN; ++i) {
    eX += h_result[i] * H[i];
    eR += Kref[i] * H[i];
  }
  printf("Energy abs error = %.6e\n", fabs(eX - eR));

  cudaEvent_t evA, evB;
  CK(cudaEventCreate(&evA));
  CK(cudaEventCreate(&evB));
  auto timeit = [&]() -> double {
    for (int w = 0; w < WARMUP; ++w) {
      CK(cudaMemcpy(b.X, dX0, NN * sizeof(dfloat), cudaMemcpyDeviceToDevice));
      trs4(b, dH, nel, OZP_MAXIT);
    }
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(evA));
    for (int it = 0; it < BENCH_ITERS; ++it) {
      CK(cudaMemcpy(b.X, dX0, NN * sizeof(dfloat), cudaMemcpyDeviceToDevice));
      trs4(b, dH, nel, OZP_MAXIT);
    }
    CK(cudaEventRecord(evB));
    CK(cudaEventSynchronize(evB));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, evA, evB));
    return ms / BENCH_ITERS;
  };
  printf("Elapsed time = %.6f\n", timeit());
  return 0;
}
