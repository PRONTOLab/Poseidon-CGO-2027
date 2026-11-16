// GENGA kick-then-drift step in one annotated kernel: accEncFull (source/directAcc.h)
// and fastfg (source/FG2.h) from GENGA (https://bitbucket.org/sigrimm/genga); see LICENSE.
// The Kepler solve is the one of genga_kepler.cu.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#include "poseidon/poseidon.h"

#define N_BODIES    65536    // smallest N that saturates the GPU
#define N_PROFILE   256      // the reverse-mode tape lives in the device heap
#define BLOCK_SIZE  256
#define WARMUP      10
#define BENCH_ITERS 20
#define KEPLER_ITERS 8       // a fixed trip count keeps the AD tape static
#define N_INTEG_STEPS 2000
#define def_ksq 1.0

// A plain multiply: llvm.nvvm.mul.rn.d has no Enzyme reverse-mode rule.
#define __dmul_rn(a, b) ((a) * (b))

// From GENGA source/directAcc.h (Grimm & Stadel 2014).
__device__ void accEncFull(double4 x4i, double4 x4j, double3 &ac, double &test, int i, int j, double MinMass, int UseTestParticles){

	int c = 0;
	if(UseTestParticles == 0 && x4i.w >= 0.0 && x4j.w >= 0.0) c = 1;
	if(UseTestParticles == 1 && x4i.w >= 0.0 && x4j.w > MinMass) c = 1;
	if(UseTestParticles == 2 && (x4i.w >= 0.0 && x4j.w >= 0.0) && (x4i.w > MinMass || x4j.w > MinMass)) c = 1;

	if(c == 1 && i != j){

		double3 r3;
		double rsq;
		double ir, ir3;
		double s;

		r3.x = x4j.x - x4i.x;
		r3.y = x4j.y - x4i.y;
		r3.z = x4j.z - x4i.z;

		rsq = r3.x*r3.x + r3.y*r3.y + r3.z*r3.z + 1.0e-30;

		ir = 1.0 / sqrt(rsq);
		ir3 = ir * ir * ir;
		s = x4j.w * ir3 * def_ksq;

		ac.x += __dmul_rn(r3.x, s);
		ac.y += __dmul_rn(r3.y, s);
		ac.z += __dmul_rn(r3.z, s);
	}
}

__device__
void gengaStepBody(const double4* __restrict__ pos,
                   const double4* __restrict__ vel,
                   double4* __restrict__ pos_out,
                   double4* __restrict__ vel_out,
                   int N, double dt, double mu) {
    extern __shared__ double4 sharedPos[];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N) return;

    double4 x4i = pos[id];

    double3 acc = {0.0, 0.0, 0.0};
    double test = 0.0;
    int numTiles = N / blockDim.x;
    for (int tile = 0; tile < numTiles; tile++) {
        sharedPos[threadIdx.x] = pos[tile * blockDim.x + threadIdx.x];
        __syncthreads();
        #pragma unroll 128
        for (unsigned int j = 0; j < blockDim.x; j++) {
            int gj = tile * blockDim.x + j;
            accEncFull(x4i, sharedPos[j], acc, test, id, gj, 0.0, 0);
        }
        __syncthreads();
    }
    double4 v4i = vel[id];
    v4i.x += acc.x * dt; v4i.y += acc.y * dt; v4i.z += acc.z * dt;

    // From GENGA source/FG2.h fastfg; an unbound body keeps its state.
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
    pos_out[id] = x4i;
    vel_out[id] = v4i;
}

POSEIDON_OPTIMIZE __global__
void genga_kernel(const double4* __restrict__ pos, const double4* __restrict__ vel,
                  double4* __restrict__ pos_out, double4* __restrict__ vel_out,
                  int N, double dt, double mu) {
    gengaStepBody(pos, vel, pos_out, vel_out, N, dt, mu);
}

__global__
void genga_gold(const double4* __restrict__ pos, const double4* __restrict__ vel,
                double4* __restrict__ pos_out, double4* __restrict__ vel_out,
                int N, double dt, double mu) {
    gengaStepBody(pos, vel, pos_out, vel_out, N, dt, mu);
}

// Small masses keep the orbits Kepler-dominated.
static double urand() { return (double)rand() / RAND_MAX; }
static void planet_ic(double4 *x, double4 *v, int N, double mu) {
    const double PI = 3.14159265358979323846;
    for (int i = 0; i < N; i++) {
        double a = 0.5 + 4.5 * urand();
        double e = 0.001 + 0.2 * urand();
        double fa = 2.0*PI*urand();
        double p = a*(1.0 - e*e);
        double r = p/(1.0 + e*cos(fa));
        double m = 1.0e-4 / N * 1.0e3;
        x[i] = make_double4(r*cos(fa), r*sin(fa), 0.0, m);
        double vmag = sqrt(mu/p);
        v[i] = make_double4(-vmag*sin(fa), vmag*(e + cos(fa)), 0.0, 0.0);
    }
}

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
    const double mu = def_ksq * 1.0;
    const double dt = 0.01;
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t smem = BLOCK_SIZE * sizeof(double4);

    double4 *h_x = new double4[N], *h_v = new double4[N];
    double4 *h_xo = new double4[N], *h_vo = new double4[N];
    srand(12345);
    planet_ic(h_x, h_v, N, mu);

    double4 *d_x,*d_v,*d_xo,*d_vo, *d_xg,*d_vg;
    cudaMalloc(&d_x,N*sizeof(double4));  cudaMalloc(&d_v,N*sizeof(double4));
    cudaMalloc(&d_xo,N*sizeof(double4)); cudaMalloc(&d_vo,N*sizeof(double4));
    cudaMalloc(&d_xg,N*sizeof(double4)); cudaMalloc(&d_vg,N*sizeof(double4));
    cudaMemcpy(d_x,h_x,N*sizeof(double4),cudaMemcpyHostToDevice);
    cudaMemcpy(d_v,h_v,N*sizeof(double4),cudaMemcpyHostToDevice);

    if (profiling) {
        for (int it = 0; it < 3; it++)
            genga_kernel<<<blocks,BLOCK_SIZE,smem>>>(d_x,d_v,d_xo,d_vo,N,dt,mu);
        cudaError_t err = cudaDeviceSynchronize();
        printf("GPU profiling complete. N=%d status=%s\n", N, cudaGetErrorString(err));
    } else if (integrate) {
        double *e0 = new double[N];
        for (int i = 0; i < N; i++) e0[i] = body_energy(h_x[i], h_v[i], mu);
        double4 *xi=d_x,*vi=d_v,*xo=d_xo,*vo=d_vo;
        cudaEvent_t s0,s1; cudaEventCreate(&s0); cudaEventCreate(&s1);
        cudaEventRecord(s0);
        for (int s = 0; s < steps; s++) {
            if (use_gold) genga_gold<<<blocks,BLOCK_SIZE,smem>>>(xi,vi,xo,vo,N,dt,mu);
            else          genga_kernel<<<blocks,BLOCK_SIZE,smem>>>(xi,vi,xo,vo,N,dt,mu);
            double4 *t; t=xi;xi=xo;xo=t; t=vi;vi=vo;vo=t;
        }
        cudaEventRecord(s1); cudaEventSynchronize(s1);
        float ms=0; cudaEventElapsedTime(&ms,s0,s1);
        cudaMemcpy(h_xo,xi,N*sizeof(double4),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vo,vi,N*sizeof(double4),cudaMemcpyDeviceToHost);
        double sum_rel=0,max_rel=0; int cnt=0;
        for (int i=0;i<N;i++){ if(e0[i]==0.0) continue; double ef=body_energy(h_xo[i],h_vo[i],mu);
            double rel=fabs((ef-e0[i])/e0[i]); sum_rel+=rel; if(rel>max_rel)max_rel=rel; cnt++; }
        cudaError_t err=cudaGetLastError();
        printf("kernel = %s\n", use_gold?"gold-F64":"opt");
        printf("N = %d  steps = %d  dt = %g\n", N, steps, dt);
        printf("Mean energy drift = %.6e\n", cnt?sum_rel/cnt:0.0);
        printf("Max energy drift = %.6e\n", max_rel);
        printf("Elapsed time = %.6f\n", ms/steps);
        printf("CUDA status: %s\n", cudaGetErrorString(err));
        delete[] e0;
    } else {
        genga_gold<<<blocks,BLOCK_SIZE,smem>>>(d_x,d_v,d_xg,d_vg,N,dt,mu);
        cudaDeviceSynchronize();
        genga_kernel<<<blocks,BLOCK_SIZE,smem>>>(d_x,d_v,d_xo,d_vo,N,dt,mu);
        cudaDeviceSynchronize();
        double4 *h_xg=new double4[N], *h_vg=new double4[N];
        cudaMemcpy(h_xg,d_xg,N*sizeof(double4),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vg,d_vg,N*sizeof(double4),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_xo,d_xo,N*sizeof(double4),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vo,d_vo,N*sizeof(double4),cudaMemcpyDeviceToHost);
        double se=0,sg=0,maxabs=0;
        for (int i=0;i<N;i++){
            double d[6]={h_xo[i].x-h_xg[i].x,h_xo[i].y-h_xg[i].y,h_xo[i].z-h_xg[i].z,
                         h_vo[i].x-h_vg[i].x,h_vo[i].y-h_vg[i].y,h_vo[i].z-h_vg[i].z};
            double gg[6]={h_xg[i].x,h_xg[i].y,h_xg[i].z,h_vg[i].x,h_vg[i].y,h_vg[i].z};
            for(int k=0;k<6;k++){ se+=d[k]*d[k]; sg+=gg[k]*gg[k]; if(fabs(d[k])>maxabs)maxabs=fabs(d[k]); }
        }
        double l2=(sg>0)?sqrt(se/sg):0.0;
        cudaEvent_t s0,s1; cudaEventCreate(&s0); cudaEventCreate(&s1);
        for(int i=0;i<WARMUP;i++) genga_kernel<<<blocks,BLOCK_SIZE,smem>>>(d_x,d_v,d_xo,d_vo,N,dt,mu);
        cudaDeviceSynchronize(); cudaEventRecord(s0);
        for(int i=0;i<BENCH_ITERS;i++) genga_kernel<<<blocks,BLOCK_SIZE,smem>>>(d_x,d_v,d_xo,d_vo,N,dt,mu);
        cudaEventRecord(s1); cudaEventSynchronize(s1);
        float ms=0; cudaEventElapsedTime(&ms,s0,s1);
        printf("N = %d\n", N);
        printf("L2 relative error = %.6e\n", l2);
        printf("Max absolute error = %.6e\n", maxabs);
        printf("Elapsed time = %.6f\n", ms/BENCH_ITERS);
        printf("CUDA status: %s\n", cudaGetErrorString(cudaGetLastError()));
        delete[] h_xg; delete[] h_vg;
    }
    return 0;
}
