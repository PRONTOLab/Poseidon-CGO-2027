// GENGA Kepler drift (symplectic orbit propagation) — Poseidon GPU Herbie bench.
//
// The numerically-relevant code (orbital-element extraction + f-and-g
// propagation) is ported from GENGA's `fastfg` (Grimm & Stadel 2014):
//   Ported from: https://bitbucket.org/sigrimm/genga/src/master/source/FG2.h
// GENGA is cite-ware ("free to use, must reference Grimm & Stadel 2014").
//
// Deviation from verbatim: GENGA accelerates the Kepler solve with a
// precomputed sin/cos lookup table (S_c[]/C_c[], a perf optimization). We use a
// direct Newton iteration with `sincos` instead — identical numerics (same
// residual f0, derivative f1=1-f3, same f-and-g), only the initial-guess
// machinery differs. This is the one deviation, analogous to genga_nbody's
// `__dmul_rn -> *`.
//
// The drift carries genuine catastrophic cancellations in VERBATIM GENGA:
//   ec = 1 - ria          (e*cos E; cancels for near-circular orbits, ria->1)
//   t1 = 1 - cos(dEj)      (cancels for small steps)
//   g  = dt + (sin(dEj) - dEj)*ien   (sin(x)-x; cancels for small steps)
// Herbie rewrites these (1-cos -> 2 sin^2(x/2); sin(x)-x -> Taylor series),
// recovering 5-13 bits at FP32 — "algebra instead of bits".
//
// Modes:  --profiling  Poseidon profile-gen (reverse-mode AD, small N)
//         (default)     one drift step, df64/FP32 vs F64-gold accuracy + timing

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#include "poseidon/poseidon.h"

#define N_BODIES    1048576   // 1<<20; per-body, embarrassingly parallel
#define N_PROFILE   256       // small N for profile-gen (reverse-mode tape)
#define BLOCK_SIZE  256
#define WARMUP      10
#define BENCH_ITERS 50
#define N_INTEG_STEPS 20000   // --integrate: drift steps; cancellation error accumulates

// GENGA units: G = def_ksq = 1, central mass Msun = 1  => mu = 1.
#define def_ksq 1.0
#define KEPLER_ITERS 8        // fixed Newton iterations (AD-friendly unroll)

// ── GENGA Kepler drift, per body — the annotated kernel's body ──
// Ported from: https://bitbucket.org/sigrimm/genga/src/master/source/FG2.h (fastfg)
__device__
void keplerDrift(const double4* __restrict__ x_in,
                 const double4* __restrict__ v_in,
                 double4* __restrict__ x_out,
                 double4* __restrict__ v_out,
                 int N, double dt, double mu) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N) return;

    double4 x4 = x_in[id];
    double4 v4 = v_in[id];

    double rsq = x4.x*x4.x + x4.y*x4.y + x4.z*x4.z;
    double vsq = v4.x*v4.x + v4.y*v4.y + v4.z*v4.z;
    double u   = x4.x*v4.x + x4.y*v4.y + x4.z*v4.z;
    double ir  = 1.0 / sqrt(rsq);
    double ia  = 2.0*ir - vsq/mu;          // 1/a (inverse semi-major axis)

    double f = 1.0, g = 0.0, fd = 0.0, gd = 1.0;   // identity if unbound

    if (ia > 0.0) {                        // bound orbit
        double ia2 = ia*ia;
        double ria = rsq*ir*ia;            // r/a
        double en  = sqrt(mu*ia2*ia);      // mean motion n = sqrt(mu/a^3)
        double ien = 1.0/en;
        double ec  = 1.0 - ria;            // e cos E   <-- cancellation
        double es  = u*ia2*ien;            // e sin E

        double dM  = en*dt;
        double dEj = dM;                   // initial guess
        dM -= es;                          // GENGA: pre-subtract es

        // Direct-Newton Kepler solve (GENGA residual; sin/cos replace the table.
        // Separate sin/cos rather than sincos(&s,&c): the pointer-output form
        // breaks Enzyme reverse-mode profiling; codegen re-fuses to sincos, so
        // no runtime cost).
        double s, c;
#ifdef KEPLER_ROLLED
        #pragma clang loop unroll(disable)   // keep trip count known (AD-friendly tape)
#else                                        // but stop the full-unroller, so Poseidon
        #pragma unroll                       // sees dEj as a loop-carried PHI (a leaf) ->
#endif                                       // f-and-g exprs stay small & Herbie-tractable
        for (int i = 0; i < KEPLER_ITERS; ++i) {
            s = sin(dEj); c = cos(dEj);
            double f2 = es*c + ec*s;
            double f3 = ec*c - es*s;
            double f0 = dEj - dM - f2;
            double f1 = 1.0 - f3;          // <-- cancellation
            dEj -= f0/f1;
        }

        s = sin(dEj); c = cos(dEj);
        double ria_ = rsq*ir*ia;
        double air_ = -1.0/ria_;           // -a/r
        double t1   = 1.0 - c;             // 1 - cos(dEj)  <-- cancellation
        double wp   = 1.0 - ec*c + es*s;
        double iwp  = 1.0/wp;

        f  = 1.0 + air_*t1;
        g  = dt + (s - dEj)*ien;           // sin(dEj)-dEj  <-- cancellation
        fd = air_*iwp*en*s;
        gd = 1.0 - t1*iwp;
    }

    // f-and-g propagation
    double4 xn, vn;
    xn.x = f*x4.x + g*v4.x;  xn.y = f*x4.y + g*v4.y;  xn.z = f*x4.z + g*v4.z;  xn.w = x4.w;
    vn.x = fd*x4.x + gd*v4.x; vn.y = fd*x4.y + gd*v4.y; vn.z = fd*x4.z + gd*v4.z; vn.w = v4.w;
    x_out[id] = xn;
    v_out[id] = vn;
}

// ── Poseidon site: the per-body drift ──
POSEIDON_OPTIMIZE __global__
void kepler_kernel(const double4* __restrict__ x_in, const double4* __restrict__ v_in,
                   double4* __restrict__ x_out, double4* __restrict__ v_out,
                   int N, double dt, double mu) {
    keplerDrift(x_in, v_in, x_out, v_out, N, dt, mu);
}

// ── F64 gold kernel (bypasses Poseidon) ──
__global__
void kepler_gold(const double4* __restrict__ x_in, const double4* __restrict__ v_in,
                 double4* __restrict__ x_out, double4* __restrict__ v_out,
                 int N, double dt, double mu) {
    keplerDrift(x_in, v_in, x_out, v_out, N, dt, mu);
}

// ── Keplerian orbit ICs (2D in xy-plane around central mu=1) ──
// Spread of semi-major axes and eccentricities; include near-circular (small e
// -> ec=1-ria cancels) and a small dt -> small dEj (t1=1-cos, g=sin-x cancel).
static double urand() { return (double)rand() / RAND_MAX; }

static void kepler_ic(double4 *x, double4 *v, int N, double mu) {
    const double PI = 3.14159265358979323846;
    for (int i = 0; i < N; i++) {
        double a = 0.5 + 4.5 * urand();          // semi-major axis [0.5,5]
        double e = 0.001 + 0.5 * urand();        // eccentricity [0.001,0.5]
        double fa = 2.0*PI*urand();              // true anomaly
        double p = a*(1.0 - e*e);
        double r = p/(1.0 + e*cos(fa));
        double m = 1.0e-6;                        // tiny mass; drift uses mu only
        x[i] = make_double4(r*cos(fa), r*sin(fa), 0.0, m);
        double vmag = sqrt(mu/p);
        v[i] = make_double4(-vmag*sin(fa), vmag*(e + cos(fa)), 0.0, 0.0);
    }
}

// Per-body Kepler orbital energy E = 1/2 |v|^2 - mu/|x|  (= -mu/2a, conserved
// EXACTLY by an exact drift). Numerical drift error makes E wander; the
// accumulated drift over many steps is the symplectic-integrator accuracy
// diagnostic — and where the 1-cos / sin-x cancellation error compounds.
static double body_energy(double4 x, double4 v, double mu) {
    double r = sqrt(x.x*x.x + x.y*x.y + x.z*x.z);
    double v2 = v.x*v.x + v.y*v.y + v.z*v.z;
    return 0.5*v2 - mu/r;
}

int main(int argc, char **argv) {
    bool profiling = false, integrate = false, use_gold = false;
    int Narg = 0, steps = N_INTEG_STEPS;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--profiling") == 0) profiling = true;
        else if (strcmp(argv[i], "--integrate") == 0) integrate = true;
        else if (strcmp(argv[i], "--gold") == 0) use_gold = true;   // integrate with pure-F64 kepler_gold (energy floor)
        else if (strncmp(argv[i], "--steps=", 8) == 0) steps = atoi(argv[i] + 8);
        else { int vv = atoi(argv[i]); if (vv > 0) Narg = vv; }
    }
    int N = profiling ? N_PROFILE : (Narg > 0 ? Narg : N_BODIES);
    if (N % BLOCK_SIZE != 0) N = (N / BLOCK_SIZE) * BLOCK_SIZE;

    const double mu = def_ksq * 1.0;     // Msun = 1
    const double dt = 0.01;              // small step -> small dEj (cancellations)

    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1ULL << 30);

    double4 *h_x = new double4[N];
    double4 *h_v = new double4[N];
    double4 *h_xo = new double4[N];
    double4 *h_vo = new double4[N];
    double4 *h_xg = new double4[N];
    double4 *h_vg = new double4[N];
    srand(42);
    kepler_ic(h_x, h_v, N, mu);

    double4 *d_x, *d_v, *d_xo, *d_vo, *d_xg, *d_vg;
    cudaMalloc(&d_x,  N*sizeof(double4)); cudaMalloc(&d_v,  N*sizeof(double4));
    cudaMalloc(&d_xo, N*sizeof(double4)); cudaMalloc(&d_vo, N*sizeof(double4));
    cudaMalloc(&d_xg, N*sizeof(double4)); cudaMalloc(&d_vg, N*sizeof(double4));
    cudaMemcpy(d_x, h_x, N*sizeof(double4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, N*sizeof(double4), cudaMemcpyHostToDevice);

    int blocks = N / BLOCK_SIZE;

    if (profiling) {
        kepler_kernel<<<blocks, BLOCK_SIZE>>>(d_x, d_v, d_xo, d_vo, N, dt, mu);
        cudaError_t err = cudaDeviceSynchronize();
        printf("GPU profiling complete. N=%d status=%s\n", N, cudaGetErrorString(err));
    } else if (integrate) {
        // Multi-step orbit integration: drift `steps` times (ping-pong buffers),
        // then measure mean/max relative drift of each body's orbital energy vs
        // its initial (analytic) value. An exact Kepler drift conserves E, so the
        // measured drift IS the accumulated numerical error of the f-and-g —
        // where the 1-cos / sin-x cancellations compound. This is the money plot:
        // F64 ~ machine floor, plain-FP32 drifts most, Herbie-FP32 in between.
        double *e0 = new double[N];
        for (int i = 0; i < N; i++) e0[i] = body_energy(h_x[i], h_v[i], mu);

        double4 *xi = d_x, *vi = d_v, *xo = d_xo, *vo = d_vo;
        cudaEvent_t s0, s1; cudaEventCreate(&s0); cudaEventCreate(&s1);
        cudaEventRecord(s0);
        for (int s = 0; s < steps; s++) {
            if (use_gold) kepler_gold<<<blocks, BLOCK_SIZE>>>(xi, vi, xo, vo, N, dt, mu);
            else          kepler_kernel<<<blocks, BLOCK_SIZE>>>(xi, vi, xo, vo, N, dt, mu);
            double4 *t; t=xi; xi=xo; xo=t; t=vi; vi=vo; vo=t;   // final output ends up in xi/vi
        }
        cudaEventRecord(s1); cudaEventSynchronize(s1);
        float ms = 0; cudaEventElapsedTime(&ms, s0, s1);
        cudaMemcpy(h_xo, xi, N*sizeof(double4), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vo, vi, N*sizeof(double4), cudaMemcpyDeviceToHost);

        double sum_rel = 0, max_rel = 0; int cnt = 0;
        for (int i = 0; i < N; i++) {
            if (e0[i] == 0.0) continue;
            double ef = body_energy(h_xo[i], h_vo[i], mu);
            double rel = fabs((ef - e0[i]) / e0[i]);
            sum_rel += rel; if (rel > max_rel) max_rel = rel; cnt++;
        }
        cudaError_t err = cudaGetLastError();
        printf("kernel = %s\n", use_gold ? "gold-F64" : "opt");
        printf("N = %d  steps = %d  dt = %g\n", N, steps, dt);
        printf("Mean energy drift = %.6e\n", cnt ? sum_rel / cnt : 0.0);
        printf("Max energy drift = %.6e\n", max_rel);
        printf("Elapsed time = %.6f\n", ms / steps);
        printf("CUDA status: %s\n", cudaGetErrorString(err));
        delete[] e0;
    } else {
        // F64 gold
        kepler_gold<<<blocks, BLOCK_SIZE>>>(d_x, d_v, d_xg, d_vg, N, dt, mu);
        cudaDeviceSynchronize();
        cudaMemcpy(h_xg, d_xg, N*sizeof(double4), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vg, d_vg, N*sizeof(double4), cudaMemcpyDeviceToHost);

        for (int i = 0; i < WARMUP; i++)
            kepler_kernel<<<blocks, BLOCK_SIZE>>>(d_x, d_v, d_xo, d_vo, N, dt, mu);
        cudaDeviceSynchronize();
        cudaEvent_t s0, s1; cudaEventCreate(&s0); cudaEventCreate(&s1);
        cudaEventRecord(s0);
        for (int i = 0; i < BENCH_ITERS; i++)
            kepler_kernel<<<blocks, BLOCK_SIZE>>>(d_x, d_v, d_xo, d_vo, N, dt, mu);
        cudaEventRecord(s1); cudaEventSynchronize(s1);
        float ms = 0; cudaEventElapsedTime(&ms, s0, s1);
        cudaMemcpy(h_xo, d_xo, N*sizeof(double4), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vo, d_vo, N*sizeof(double4), cudaMemcpyDeviceToHost);

        // Accuracy: opt vs F64 gold over (x,v) components.
        double sum_sq_err = 0, sum_sq_gold = 0, max_abs = 0;
        auto acc = [&](double o, double gg){ double d=o-gg; sum_sq_err+=d*d; sum_sq_gold+=gg*gg; if(fabs(d)>max_abs)max_abs=fabs(d); };
        for (int i = 0; i < N; i++) {
            acc(h_xo[i].x,h_xg[i].x); acc(h_xo[i].y,h_xg[i].y); acc(h_xo[i].z,h_xg[i].z);
            acc(h_vo[i].x,h_vg[i].x); acc(h_vo[i].y,h_vg[i].y); acc(h_vo[i].z,h_vg[i].z);
        }
        double l2 = (sum_sq_gold>0.0) ? sqrt(sum_sq_err/sum_sq_gold) : 0.0;
        cudaError_t err = cudaGetLastError();
        printf("N = %d  blocks = %d  dt = %g\n", N, blocks, dt);
        printf("L2 relative error = %.6e\n", l2);
        printf("Max absolute error = %.6e\n", max_abs);
        printf("Elapsed time = %.6f\n", ms / BENCH_ITERS);
        printf("CUDA status: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_x);cudaFree(d_v);cudaFree(d_xo);cudaFree(d_vo);cudaFree(d_xg);cudaFree(d_vg);
    delete[] h_x;delete[] h_v;delete[] h_xo;delete[] h_vo;delete[] h_xg;delete[] h_vg;
    return 0;
}
