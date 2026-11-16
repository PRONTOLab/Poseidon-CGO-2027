// GENGA all-pairs force kernel. accEncFull and def_ksq are from GENGA
// (https://bitbucket.org/sigrimm/genga, source/directAcc.h, define.h); see LICENSE.
// Modes: --profiling (small N), --integrate (energy drift), --gold (time the un-annotated kernel).

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#include "poseidon/poseidon.h"

#define N_BODIES    65536   // smallest N that saturates the GPU
#define N_PROFILE   256     // the reverse-mode tape lives in the device heap
#define BLOCK_SIZE  256
#define WARMUP      10
#define BENCH_ITERS 20

#define N_INTEG     8192
#define INTEG_STEPS 4000
#define INTEG_DT    0.001
#define INTEG_EVERY 50

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

// Shared-memory tiled all-pairs accumulation (GPU Gems 3, Ch. 31).
__device__
void computeBodyAccel(const double4* __restrict__ pos,
                      double3* __restrict__ acc,
                      int N) {
    extern __shared__ double4 sharedPos[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double4 myPos = pos[i];
    double3 myAcc = {0.0, 0.0, 0.0};
    double test = 0.0;
    int numTiles = N / blockDim.x;

    for (int tile = 0; tile < numTiles; tile++) {
        sharedPos[threadIdx.x] = pos[tile * blockDim.x + threadIdx.x];
        __syncthreads();

        #pragma unroll 128
        for (unsigned int j = 0; j < blockDim.x; j++) {
            int gj = tile * blockDim.x + j;
            accEncFull(myPos, sharedPos[j], myAcc, test, i, gj, 0.0, 0);
        }
        __syncthreads();
    }

    acc[i] = myAcc;
}

POSEIDON_OPTIMIZE __global__
void nbody_kernel(const double4* __restrict__ pos,
                  double3* __restrict__ acc,
                  int N) {
    computeBodyAccel(pos, acc, N);
}

__global__
void nbody_force_gold(const double4* __restrict__ pos,
                      double3* __restrict__ acc, int N) {
    computeBodyAccel(pos, acc, N);
}

static double urand() { return (double)rand() / RAND_MAX; }

// Plummer sphere, G = M = a = 1; vel may be null.
static void plummer_ic(double4 *pos, double3 *vel, int N) {
    const double PI = 3.14159265358979323846;
    for (int i = 0; i < N; i++) {
        double m = 1.0 / N;
        double X1 = urand();
        double r = 1.0 / sqrt(pow(X1, -2.0/3.0) - 1.0);
        double X2 = urand(), X3 = urand();
        double z  = (1.0 - 2.0 * X2) * r;
        double rxy = sqrt(fmax(0.0, r*r - z*z));
        double x  = rxy * cos(2.0 * PI * X3);
        double y  = rxy * sin(2.0 * PI * X3);
        pos[i] = make_double4(x, y, z, m);
        if (vel) {
            double q, g;
            do { q = urand(); g = 0.1 * urand(); } while (g > q*q * pow(1.0 - q*q, 3.5));
            double ve = sqrt(2.0) * pow(1.0 + r*r, -0.25);
            double v  = q * ve;
            double X6 = urand(), X7 = urand();
            double vz = (1.0 - 2.0 * X6) * v;
            double vrxy = sqrt(fmax(0.0, v*v - vz*vz));
            vel[i] = make_double3(vrxy * cos(2.0 * PI * X7), vrxy * sin(2.0 * PI * X7), vz);
        }
    }
}

// Total energy in FP64 with accEncFull's softening.
static double compute_energy_f64(const double4 *pos, const double3 *vel, int N) {
    double ke = 0.0, pe = 0.0;
    for (int i = 0; i < N; i++)
        ke += 0.5 * pos[i].w * (vel[i].x*vel[i].x + vel[i].y*vel[i].y + vel[i].z*vel[i].z);
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            double dx = pos[i].x-pos[j].x, dy = pos[i].y-pos[j].y, dz = pos[i].z-pos[j].z;
            double rsq = dx*dx + dy*dy + dz*dz + 1.0e-30;
            pe -= def_ksq * pos[i].w * pos[j].w / sqrt(rsq);
        }
    return ke + pe;
}

int main(int argc, char **argv) {
    bool profiling = false, integrate = false, encounter = false, use_gold = false;
    double sep = 1e-14;
    int Narg = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--profiling") == 0) profiling = true;
        else if (strcmp(argv[i], "--gold") == 0) use_gold = true;
        else if (strcmp(argv[i], "--integrate") == 0) integrate = true;
        else if (strcmp(argv[i], "--encounter") == 0) encounter = true;
        else if (strncmp(argv[i], "--sep=", 6) == 0) sep = atof(argv[i] + 6);
        else { int v = atoi(argv[i]); if (v > 0) Narg = v; }
    }
    int N = profiling ? N_PROFILE
          : integrate ? (Narg > 0 ? Narg : N_INTEG)
          : (Narg > 0 ? Narg : N_BODIES);
    if (N % BLOCK_SIZE != 0) N = (N / BLOCK_SIZE) * BLOCK_SIZE;

    // The Enzyme reverse-mode tape uses device malloc.
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1ULL << 30);

    double4 *h_pos = new double4[N];
    double3 *h_vel = new double3[N];
    double3 *h_opt = new double3[N];
    double3 *h_gold = new double3[N];
    srand(42);
    plummer_ic(h_pos, h_vel, N);

    if (encounter && N >= 2) {
        h_pos[1].x = h_pos[0].x + sep;
        h_pos[1].y = h_pos[0].y;
        h_pos[1].z = h_pos[0].z;
    }

    double4 *d_pos;
    double3 *d_acc, *d_gold;
    cudaMalloc(&d_pos,  N * sizeof(double4));
    cudaMalloc(&d_acc,  N * sizeof(double3));
    cudaMalloc(&d_gold, N * sizeof(double3));
    cudaMemcpy(d_pos, h_pos, N * sizeof(double4), cudaMemcpyHostToDevice);

    int blocks = N / BLOCK_SIZE;
    int smem = BLOCK_SIZE * sizeof(double4);

    if (integrate) {
        // Velocity Verlet in FP64 on the host; only the GPU force varies in precision.
        auto force = [&]() {
            cudaMemcpy(d_pos, h_pos, N * sizeof(double4), cudaMemcpyHostToDevice);
            nbody_kernel<<<blocks, BLOCK_SIZE, smem>>>(d_pos, d_acc, N);
            cudaDeviceSynchronize();
            cudaMemcpy(h_opt, d_acc, N * sizeof(double3), cudaMemcpyDeviceToHost);
        };
        force();
        double E0 = compute_energy_f64(h_pos, h_vel, N);
        printf("# integrate N=%d steps=%d dt=%g\n", N, INTEG_STEPS, INTEG_DT);
        printf("step %d t %.4f E %.15e relerr %.6e\n", 0, 0.0, E0, 0.0);
        for (int s = 1; s <= INTEG_STEPS; s++) {
            for (int i = 0; i < N; i++) {
                h_vel[i].x += 0.5 * INTEG_DT * h_opt[i].x;
                h_vel[i].y += 0.5 * INTEG_DT * h_opt[i].y;
                h_vel[i].z += 0.5 * INTEG_DT * h_opt[i].z;
                h_pos[i].x += INTEG_DT * h_vel[i].x;
                h_pos[i].y += INTEG_DT * h_vel[i].y;
                h_pos[i].z += INTEG_DT * h_vel[i].z;
            }
            force();
            for (int i = 0; i < N; i++) {
                h_vel[i].x += 0.5 * INTEG_DT * h_opt[i].x;
                h_vel[i].y += 0.5 * INTEG_DT * h_opt[i].y;
                h_vel[i].z += 0.5 * INTEG_DT * h_opt[i].z;
            }
            if (s % INTEG_EVERY == 0) {
                bool bad = false;
                for (int i = 0; i < N && !bad; i++) {
                    double px = h_pos[i].x, py = h_pos[i].y, pz = h_pos[i].z;
                    if (px != px || py != py || pz != pz ||
                        fabs(px) > 1e300 || fabs(py) > 1e300 || fabs(pz) > 1e300) bad = true;
                }
                double E = bad ? NAN : compute_energy_f64(h_pos, h_vel, N);
                double relerr = bad ? NAN : fabs((E - E0) / E0);
                printf("step %d t %.4f E %.15e relerr %.6e%s\n",
                       s, s * INTEG_DT, E, relerr, bad ? " BLOWUP" : "");
                if (bad) { printf("BLOWUP at step %d\n", s); break; }
            }
        }
    } else if (profiling) {
        nbody_kernel<<<blocks, BLOCK_SIZE, smem>>>(d_pos, d_acc, N);
        cudaError_t err = cudaDeviceSynchronize();
        printf("GPU profiling complete. N=%d. status=%s\n", N, cudaGetErrorString(err));
    } else {
        nbody_force_gold<<<blocks, BLOCK_SIZE, smem>>>(d_pos, d_gold, N);
        cudaDeviceSynchronize();
        cudaMemcpy(h_gold, d_gold, N * sizeof(double3), cudaMemcpyDeviceToHost);

        auto timed = [&]() {
            if (use_gold) nbody_force_gold<<<blocks, BLOCK_SIZE, smem>>>(d_pos, d_acc, N);
            else          nbody_kernel<<<blocks, BLOCK_SIZE, smem>>>(d_pos, d_acc, N);
        };
        for (int i = 0; i < WARMUP; i++) timed();
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < BENCH_ITERS; i++) timed();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0; cudaEventElapsedTime(&ms, start, stop);
        cudaMemcpy(h_opt, d_acc, N * sizeof(double3), cudaMemcpyDeviceToHost);

        int total = 3 * N;
        int bit_exact = 0, gold_zero = 0, diff_count = 0;
        double sum_log_err = 0, max_rel_err = 0, max_abs_err = 0;
        double sum_sq_err = 0, sum_sq_gold = 0;
        const double *o = (const double *)h_opt;
        const double *g = (const double *)h_gold;
        for (int i = 0; i < total; i++) {
            double diff = o[i] - g[i];
            double ae = fabs(diff);
            if (ae == 0.0) bit_exact++;
            if (ae > max_abs_err) max_abs_err = ae;
            sum_sq_err += diff * diff;
            sum_sq_gold += g[i] * g[i];
            if (g[i] == 0.0) { gold_zero++; continue; }
            double rel = ae / fabs(g[i]);
            if (rel > max_rel_err) max_rel_err = rel;
            if (rel > 0.0) { sum_log_err += log(rel); diff_count++; }
        }
        double l2_rel_err = (sum_sq_gold > 0.0) ? sqrt(sum_sq_err / sum_sq_gold) : 0.0;

        cudaError_t err = cudaGetLastError();
        printf("kernel = %s\n", use_gold ? "gold-F64" : "opt");
        printf("N = %d  blocks = %d\n", N, blocks);
        printf("Elements = %d (bit-exact %d, gold==0 %d, differing %d)\n",
               total, bit_exact, gold_zero, diff_count);
        printf("L2 relative error = %.6e\n", l2_rel_err);
        printf("Geomean rel error = %.6e\n", diff_count ? exp(sum_log_err / diff_count) : 0.0);
        printf("Max relative error = %.6e\n", max_rel_err);
        printf("Max absolute error = %.6e\n", max_abs_err);
        printf("Elapsed time = %.6f\n", ms / BENCH_ITERS);
        printf("CUDA status: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_pos); cudaFree(d_acc); cudaFree(d_gold);
    delete[] h_pos; delete[] h_vel; delete[] h_opt; delete[] h_gold;
    return 0;
}
