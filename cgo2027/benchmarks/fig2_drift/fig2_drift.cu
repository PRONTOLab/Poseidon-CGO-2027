// fig2_drift.cu — REAL wall-clock + accuracy for the intro figure's
// floating-point EMULATION example, drift edition (nobody "dies"):
//   v1: double   leapfrog oscillator      (FP64, as written)
//   v2: 3xFP32   triple-float emulation   (mX_real tf_Real_sloppy)
//   v3: float                             (naive FP32 downcast)
// Long symplectic integration: truncation is identical across precisions
// (reference = dd_Real 2xFP64 leapfrog with the SAME dt), so trajectory
// divergence from the reference is PURE accumulated rounding. It grows
// steadily (no chaotic blow-up): FP32 drifts to ~1e-2, FP64 to ~1e-12,
// 3xFP32 stays ~5 orders below FP64. Checkpoints at powers of 2.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include "mX_real/mX_real.hpp"
using namespace mX_real;

#define CK(x)                                                            \
  do {                                                                   \
    cudaError_t e = (x);                                                 \
    if (e != cudaSuccess) {                                              \
      fprintf(stderr, "cuda %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(e));                                    \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

using TF = tf_Real_sloppy;  // 3xFP32, FP32 ops only

__host__ __device__ static inline double to_d(float a) { return (double)a; }
__host__ __device__ static inline double to_d(double a) { return a; }
template <typename T> __host__ __device__ static inline double to_d(const T &v) {
  double s = 0;
  for (int i = T::L - 1; i >= 0; --i) s += (double)v.x[i];
  return s;
}
__host__ __device__ static inline float fromD(double d, float) { return (float)d; }
__host__ __device__ static inline double fromD(double d, double) { return d; }
template <typename MX> __host__ __device__ static inline MX fromD(double d, MX) {
  MX r;
  double rem = d;
  for (int i = 0; i < MX::L; ++i) {
    typename MX::base_T li = (typename MX::base_T)rem;
    r.x[i] = li;
    rem -= (double)li;
  }
  return r;
}

#define N_ENS 4096
#define LOG2_STEPS 20            // 2^20 ~ 1.05e6 steps
#define DT 0.0078125             // 1/128: EXACT in FP32, so the emulated
                                  // version can keep dt as a single float
                                  // limb (mixed tf*float ops); ~804 steps/period
#ifndef N_TIME
#define N_TIME (1 << 20)
#endif
#ifndef K_TIME
#define K_TIME 2048
#endif

// dt as the cheapest exact scalar for each type: double for double/float
// paths, a SINGLE float limb for MultiFloat (1/128 is FP32-exact).
static inline __host__ __device__ double dt_of(double) { return DT; }
static inline __host__ __device__ float dt_of(float) { return (float)DT; }
template <typename MX>
static inline __host__ __device__ typename MX::base_T dt_of(MX) {
  return (typename MX::base_T)DT;
}

// ---- trajectory kernel: leapfrog, dump x at power-of-2 checkpoints ----
template <typename T>
__global__ void krn_traj(const double *__restrict__ x0, double *__restrict__ chk) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N_ENS) return;
  T x = fromD(x0[i], T());
  T v = fromD(0.0, T());
  auto dt = dt_of(T());
  long next = 1; int c = 0;
  for (long n = 1; n <= (1L << LOG2_STEPS); ++n) {
    v = v - x * dt;   // k = 1
    x = x + v * dt;
    if (n == next) {
      // store hi+lo so sub-double-ulp differences vs the reference survive
      double hi = to_d(x);
      chk[(size_t)(2 * c) * N_ENS + i] = hi;
      chk[(size_t)(2 * c + 1) * N_ENS + i] = to_d(x - fromD(hi, T()));
      ++c; next <<= 1;
    }
  }
}

// ---- timing kernel: K steps per thread ----
template <typename T>
__global__ void krn_time(const double *__restrict__ x0, double *__restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N_TIME) return;
  T x = fromD(x0[i], T());
  T v = fromD(0.0, T());
  auto dt = dt_of(T());
  for (int n = 0; n < K_TIME; ++n) {
    v = v - x * dt;
    x = x + v * dt;
  }
  out[i] = to_d(x) + to_d(v);
}

template <typename T>
static void cpu_time(const double *x0, double *out, size_t n) {
  auto dt = dt_of(T());
  for (size_t i = 0; i < n; ++i) {
    T x = fromD(x0[i], T());
    T v = fromD(0.0, T());
    for (int k = 0; k < K_TIME; ++k) {
      v = v - x * dt;
      x = x + v * dt;
    }
    out[i] = to_d(x) + to_d(v);
  }
}

volatile double g_sink;

template <typename F> static double time_ms(F f, int reps) {
  f();
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < reps; ++r) f();
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

int main() {
  std::srand(42);
  std::vector<double> hx0(N_TIME);
  for (auto &v : hx0) v = 0.5 + 1.0 * (std::rand() / (double)RAND_MAX);

  double *dx0, *dout, *dchk;
  CK(cudaMalloc(&dx0, (size_t)N_TIME * 8));
  CK(cudaMalloc(&dout, (size_t)N_TIME * 8));
  CK(cudaMalloc(&dchk, (size_t)N_ENS * (LOG2_STEPS + 1) * 16));
  CK(cudaMemcpy(dx0, hx0.data(), (size_t)N_TIME * 8, cudaMemcpyHostToDevice));
  int blk = 256;

  // ---- GPU timing ----
  auto gtime = [&](auto tag, const char *name, double base) {
    using T = decltype(tag);
    int grd = (N_TIME + blk - 1) / blk;
    krn_time<T><<<grd, blk>>>(dx0, dout);
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int r = 0; r < 5; ++r) krn_time<T><<<grd, blk>>>(dx0, dout);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 5;
    printf("%-26s %9.3f ms   %6.2fx\n", name, ms, base > 0 ? base / ms : 1.0);
    return (double)ms;
  };
  double gb = gtime(double(), "GPU FP64 (as written)", -1);
  gtime(TF(), "GPU 3xFP32 emulation", gb);
  gtime(float(), "GPU naive FP32", gb);

  // ---- CPU timing (single thread) ----
  const size_t nc = 1 << 12;
  std::vector<double> hout(N_TIME);
  double c64 = time_ms(
      [&] { cpu_time<double>(hx0.data(), hout.data(), nc); g_sink = hout[nc - 1]; }, 3);
  printf("%-26s %9.3f ms   %6.2fx\n", "CPU FP64 (as written)", c64, 1.0);
  double ctf = time_ms(
      [&] { cpu_time<TF>(hx0.data(), hout.data(), nc); g_sink = hout[nc - 1]; }, 3);
  printf("%-26s %9.3f ms   %6.2fx (i.e., %.1fx SLOWER)\n",
         "CPU 3xFP32 emulation", ctf, c64 / ctf, ctf / c64);

  // ---- drift trajectories vs dd_Real (same dt => same truncation) ----
  const int NC = LOG2_STEPS + 1;
  std::vector<double> t32(2 * N_ENS * NC), t64(2 * N_ENS * NC),
      ttf(2 * N_ENS * NC), tdd(2 * N_ENS * NC);
  int grd = (N_ENS + blk - 1) / blk;
  auto traj = [&](auto tag, std::vector<double> &h) {
    using T = decltype(tag);
    krn_traj<T><<<grd, blk>>>(dx0, dchk);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h.data(), dchk, (size_t)N_ENS * NC * 16,
                  cudaMemcpyDeviceToHost));
  };
  traj(float(), t32);
  traj(double(), t64);
  traj(TF(), ttf);
  traj(dd_Real(), tdd);

  FILE *f = fopen("drift.csv", "w");
  fprintf(f, "steps,fp32,fp64,tf32x3\n");
  for (int c = 0; c < NC; ++c) {
    double e32 = 0, e64 = 0, etf = 0;
    for (int i = 0; i < N_ENS; ++i) {
      size_t hi = (size_t)(2 * c) * N_ENS + i, lo = hi + (size_t)N_ENS;
      e32 += std::fabs((t32[hi] - tdd[hi]) + (t32[lo] - tdd[lo]));
      e64 += std::fabs((t64[hi] - tdd[hi]) + (t64[lo] - tdd[lo]));
      etf += std::fabs((ttf[hi] - tdd[hi]) + (ttf[lo] - tdd[lo]));
    }
    long steps = 1L << c;
    fprintf(f, "%ld,%.6e,%.6e,%.6e\n", steps, e32 / N_ENS, e64 / N_ENS,
            etf / N_ENS);
    printf("  steps=%8ld  fp32=%9.2e  fp64=%9.2e  3xf32=%9.2e\n", steps,
           e32 / N_ENS, e64 / N_ENS, etf / N_ENS);
  }
  fclose(f);
  printf("wrote drift.csv\n");
  return 0;
}
