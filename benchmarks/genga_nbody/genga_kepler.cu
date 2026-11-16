// GENGA Kepler drift: keplerDrift is GENGA's fastfg (https://bitbucket.org/sigrimm/genga,
// source/FG2.h) with a fixed-count Newton iteration for the Kepler solve; see LICENSE.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#include "poseidon/poseidon.h"

#define N_BODIES    1048576
#define N_PROFILE   256       // the reverse-mode tape lives in the device heap
#define BLOCK_SIZE  256
#define WARMUP      10
#define BENCH_ITERS 50
#define N_INTEG_STEPS 20000

#define def_ksq 1.0
#define KEPLER_ITERS 8        // a fixed trip count keeps the AD tape static

// From GENGA source/FG2.h fastfg (Grimm & Stadel 2014).
__device__
void keplerDrift(const double4* __restrict__ x_in,
                 const double4* __restrict__ v_in,
                 double4* __restrict__ x_out,
                 double4* __restrict__ v_out,
                 int N, double dt, double mu) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N) return;

    double4 x4i = x_in[id];
    double4 v4i = v_in[id];

    // An unbound body keeps its state (GENGA steps it with BSSinglestep).
    double dEj;
    double s,c,wp;
    double f = 1.0;	  // Gauss's f, g, fdot and gdot
    double g = 0.0;
    double fd = 0.0;
    double gd = 1.0;

    double rsq, ir,vsq;
    double u;		  // r v cos(phi)
    double ia;		  // semi-major axis
    double ec,es;		  // e cos(E), e sin(E)
    double ien, en;		  // mean motion
    double dM;		  // delta mean anomoly
    double t1;
    double ria, air_, iwp;
    double3 t;

    rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z;
    vsq = v4i.x*v4i.x + v4i.y*v4i.y + v4i.z*v4i.z;
    u =  x4i.x*v4i.x + x4i.y*v4i.y + x4i.z*v4i.z;
    ir = 1.0/sqrt(rsq);
    ia = 2.0*ir-vsq/mu;

    if(ia > 0.0){

        t1 = ia*ia;
        ria = rsq*ir*ia;
        en = sqrt(mu*t1*ia);
        ien = 1.0/en;
        ec = 1.0-ria;
        es = u*t1*ien;
        dM = en * dt;

        dEj = dM;
        dM -= es;

        // Separate sin/cos: the pointer outputs of sincos break Enzyme reverse mode.
        // KEPLER_ROLLED keeps dEj a loop-carried leaf, so the f-and-g subgraphs stay small.
#ifdef KEPLER_ROLLED
        #pragma clang loop unroll(disable)
#else
        #pragma unroll
#endif
        for (int i = 0; i < KEPLER_ITERS; ++i) {
            s = sin(dEj); c = cos(dEj);
            double f2 = es*c + ec*s;
            double f3 = ec*c - es*s;
            double f0 = dEj - dM - f2;
            double f1 = 1.0 - f3;
            dEj -= f0/f1;
        }
        s = sin(dEj); c = cos(dEj);

        air_ = -1.0/ria;
        t1 = 1.0 - c;
        wp = 1.0 - ec*c + es*s;
        iwp = 1.0/wp;
        f = 1.0 + air_ * t1;
        g = dt + (s-dEj)*ien;
        fd = air_ * iwp * en * s;
        gd = 1.0 - t1*iwp;

    }
    t.x = f*x4i.x+g*v4i.x;
    t.y = f*x4i.y+g*v4i.y;
    t.z = f*x4i.z+g*v4i.z;
    v4i.x = fd*x4i.x+gd*v4i.x;
    v4i.y = fd*x4i.y+gd*v4i.y;
    v4i.z = fd*x4i.z+gd*v4i.z;
    x4i.x = t.x;
    x4i.y = t.y;
    x4i.z = t.z;
    x_out[id] = x4i;
    v_out[id] = v4i;
}

POSEIDON_OPTIMIZE __global__
void kepler_kernel(const double4* __restrict__ x_in, const double4* __restrict__ v_in,
                   double4* __restrict__ x_out, double4* __restrict__ v_out,
                   int N, double dt, double mu) {
    keplerDrift(x_in, v_in, x_out, v_out, N, dt, mu);
}

__global__
void kepler_gold(const double4* __restrict__ x_in, const double4* __restrict__ v_in,
                 double4* __restrict__ x_out, double4* __restrict__ v_out,
                 int N, double dt, double mu) {
    keplerDrift(x_in, v_in, x_out, v_out, N, dt, mu);
}

static double urand() { return (double)rand() / RAND_MAX; }

static void kepler_ic(double4 *x, double4 *v, int N, double mu) {
    const double PI = 3.14159265358979323846;
    for (int i = 0; i < N; i++) {
        double a = 0.5 + 4.5 * urand();
        double e = 0.001 + 0.5 * urand();
        double fa = 2.0*PI*urand();
        double p = a*(1.0 - e*e);
        double r = p/(1.0 + e*cos(fa));
        double m = 1.0e-6;
        x[i] = make_double4(r*cos(fa), r*sin(fa), 0.0, m);
        double vmag = sqrt(mu/p);
        v[i] = make_double4(-vmag*sin(fa), vmag*(e + cos(fa)), 0.0, 0.0);
    }
}

// Orbital energy, conserved exactly by an exact drift.
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
        else if (strcmp(argv[i], "--gold") == 0) use_gold = true;
        else if (strncmp(argv[i], "--steps=", 8) == 0) steps = atoi(argv[i] + 8);
        else { int vv = atoi(argv[i]); if (vv > 0) Narg = vv; }
    }
    int N = profiling ? N_PROFILE : (Narg > 0 ? Narg : N_BODIES);
    if (N % BLOCK_SIZE != 0) N = (N / BLOCK_SIZE) * BLOCK_SIZE;

    const double mu = def_ksq * 1.0;
    const double dt = 0.01;

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
        double *e0 = new double[N];
        for (int i = 0; i < N; i++) e0[i] = body_energy(h_x[i], h_v[i], mu);

        double4 *xi = d_x, *vi = d_v, *xo = d_xo, *vo = d_vo;
        cudaEvent_t s0, s1; cudaEventCreate(&s0); cudaEventCreate(&s1);
        cudaEventRecord(s0);
        for (int s = 0; s < steps; s++) {
            if (use_gold) kepler_gold<<<blocks, BLOCK_SIZE>>>(xi, vi, xo, vo, N, dt, mu);
            else          kepler_kernel<<<blocks, BLOCK_SIZE>>>(xi, vi, xo, vo, N, dt, mu);
            double4 *t; t=xi; xi=xo; xo=t; t=vi; vi=vo; vo=t;
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
