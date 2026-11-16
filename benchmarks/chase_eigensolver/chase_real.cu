// R-ChFSI eigensolver with five annotated matrix-product sites.
// Portions follow ChASE (https://github.com/ChASE-library/ChASE); see LICENSE.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "poseidon/poseidon.h"

#ifndef CHASE_N
#define CHASE_N 1024
#endif
#ifndef CHASE_NEV
#define CHASE_NEV 64
#endif
#define N    CHASE_N
#define NEV  CHASE_NEV          // wanted (lowest) eigenpairs
#define NEX_RAW (NEV/2 + 4)    // ChASE's nex
#define NEVEX (((NEV + NEX_RAW + 15)/16)*16)  // WMMA tiles need whole 16-column tiles
#define NEX  (NEVEX - NEV)
#define LANCZOS_M 25
#define MAX_DEG  12
#define MIN_DEG  6
#define MAX_OUTER 40
#define TOL  1e-10             // locking tolerance

#define CK(x)  do{ cudaError_t e=(x); if(e){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define CB(x)  do{ cublasStatus_t s=(x); if(s){printf("cuBLAS %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);} }while(0)
#define CS(x)  do{ cusolverStatus_t s=(x); if(s){printf("cuSOLVER %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);} }while(0)

static cublasHandle_t g_cublas; static cusolverDnHandle_t g_cusolver;
static double *g_H;                 // N x N symmetric, column-major

// Filter: W = alpha*(H*V) + beta*W.
__device__
void hemm_body(double *__restrict__ W, const double *__restrict__ H,
               const double *__restrict__ V, double alpha, double beta) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= NEVEX) return;
    const double *hrow = H + (size_t)row * N;   // row of H = column by symmetry: K-contiguous
    const double *vcol = V + (size_t)col * N;
    double mm = 0.0;
#pragma unroll 1
    for (int l = 0; l < N; ++l)
        mm += hrow[l] * vcol[l];
    size_t idx = (size_t)col * N + row;
    W[idx] = alpha * mm + beta * W[idx];
}
POSEIDON_OPTIMIZE __global__
void hemm_opt(double *W, const double *H, const double *V,
              double alpha, double beta) {
    hemm_body(W, H, V, alpha, beta);
}
__global__ void hemm_gold(double *W, const double *H, const double *V,
                          double alpha, double beta) {
    hemm_body(W, const_cast<double *>(H), const_cast<double *>(V), alpha, beta);
}

// Residual: D = HX - X diag(ritz).
__device__
void resid_body(double *__restrict__ D, const double *__restrict__ H,
                const double *__restrict__ X, const double *__restrict__ ritz) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= NEVEX) return;
    const double *hrow = H + (size_t)row * N;
    const double *xcol = X + (size_t)col * N;
    double hx = 0.0;
#pragma unroll 1
    for (int l = 0; l < N; ++l)
        hx += hrow[l] * xcol[l];
    D[(size_t)col * N + row] = hx - ritz[col] * xcol[row];
}
POSEIDON_OPTIMIZE __global__
void resid_opt(double *D, const double *H, const double *X, const double *ritz) {
    resid_body(D, H, X, ritz);
}
__global__ void resid_gold(double *D, const double *H, const double *X, const double *ritz) {
    resid_body(D, const_cast<double *>(H), const_cast<double *>(X),
               const_cast<double *>(ritz));
}
// resid[col] = column 2-norm of D
__global__ void k_colnorm(const double *D, double *resid) {
    int col = blockIdx.x, tx = threadIdx.x; __shared__ double ss[256];
    double a = 0.0;
    for (int row = tx; row < N; row += blockDim.x) { double r=D[(size_t)col*N+row]; a += r*r; }
    ss[tx] = a; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tx < s) ss[tx] += ss[tx + s]; __syncthreads(); }
    if (tx == 0) resid[col] = sqrt(ss[0]);
}

// Gram: G = V^T V, for Cholesky-QR.
__device__
void gram_body(double *__restrict__ G, const double *__restrict__ V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= NEVEX || j >= NEVEX) return;
    const double *vi = V + (size_t)i * N;
    const double *vj = V + (size_t)j * N;
    double s = 0.0;
#pragma unroll 1
    for (int l = 0; l < N; ++l) s += vi[l] * vj[l];
    G[(size_t)j * NEVEX + i] = s;
}
static dim3 g_gblk(16,16), g_ggrid;
POSEIDON_OPTIMIZE __global__
void gram_opt(double *G, const double *V) { gram_body(G, V); }
__global__ void gram_gold(double *G, const double *V) { gram_body(G, const_cast<double *>(V)); }

// Rayleigh-Ritz: W = H V.
__device__
void rrhv_body(double *__restrict__ W, const double *__restrict__ H, const double *__restrict__ V) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= NEVEX) return;
    const double *hrow = H + (size_t)row * N;
    const double *vcol = V + (size_t)col * N;
    double mm = 0.0;
#pragma unroll 1
    for (int l = 0; l < N; ++l) mm += hrow[l] * vcol[l];
    W[(size_t)col * N + row] = mm;
}
POSEIDON_OPTIMIZE __global__
void rrhv_opt(double *W, const double *H, const double *V) { rrhv_body(W, H, V); }
__global__ void rrhv_gold(double *W, const double *H, const double *V) {
    rrhv_body(W, const_cast<double *>(H), const_cast<double *>(V));
}

// Rayleigh-Ritz projection: A = V^T W.
__device__
void rrproj_body(double *__restrict__ A, const double *__restrict__ V, const double *__restrict__ W) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= NEVEX || j >= NEVEX) return;
    const double *vi = V + (size_t)i * N;
    const double *wj = W + (size_t)j * N;
    double s = 0.0;
#pragma unroll 1
    for (int l = 0; l < N; ++l) s += vi[l] * wj[l];
    A[(size_t)j * NEVEX + i] = s;
}
POSEIDON_OPTIMIZE __global__
void rrproj_opt(double *A, const double *V, const double *W) { rrproj_body(A, V, W); }
__global__ void rrproj_gold(double *A, const double *V, const double *W) {
    rrproj_body(A, const_cast<double *>(V), const_cast<double *>(W));
}

__global__ void k_col_axpy(double* Y,const double* R,const double* s){           // Y += R diag(s)
    int col=blockIdx.y, i=blockIdx.x*blockDim.x+threadIdx.x;
    if(col<NEVEX && i<N){ size_t k=(size_t)col*N+i; Y[k]+=s[col]*R[k]; }
}
__global__ void k_recon(double* Y,const double* X,const double* Lm,const double* Z){ // Y = X diag(Lm) + Z
    int col=blockIdx.y, i=blockIdx.x*blockDim.x+threadIdx.x;
    if(col<NEVEX && i<N){ size_t k=(size_t)col*N+i; Y[k]=X[k]*Lm[col]+Z[k]; }
}

__global__ void k_shift_diag(double *A, int n, double s){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) A[(size_t)i*n+i]+=s; }
__global__ void k_scale_cols(const double* Q,const double* l,double* M,int n){ int c=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){size_t k=(size_t)c*n+i; M[k]=Q[k]*l[c];} }
__global__ void k_symmetrize(double* A,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y*blockDim.y+threadIdx.y; if(i>=n||j>=n||i>=j)return; double a=0.5*(A[(size_t)j*n+i]+A[(size_t)i*n+j]); A[(size_t)j*n+i]=a; A[(size_t)i*n+j]=a; }

static dim3 g_hblk(16,16), g_hgrid;
static inline void HEMM(double* W,double* V,double alpha,double beta,int fmode){
    if(fmode==0) hemm_opt<<<g_hgrid,g_hblk>>>(W,g_H,V,alpha,beta);
    else         hemm_gold<<<g_hgrid,g_hblk>>>(W,g_H,V,alpha,beta);
}

// Spectral bounds from m Lanczos steps, after ChASE lanczos().
static void lanczos_bounds(double* dv, double* dw, double* scratch,
                           double& lower, double& upper, double& lambda1){
    int m = LANCZOS_M;
    std::vector<double> a(m,0.0), b(m,0.0);
    double one=1.0,zero=0.0,nrm;
    std::vector<double> h(N); srand(91); for(int i=0;i<N;i++) h[i]=(double)rand()/RAND_MAX-0.5;
    CK(cudaMemcpy(dv,h.data(),N*sizeof(double),cudaMemcpyHostToDevice));
    CB(cublasDnrm2(g_cublas,N,dv,1,&nrm)); { double sc=1.0/nrm; CB(cublasDscal(g_cublas,N,&sc,dv,1)); }
    CK(cudaMemset(dw,0,N*sizeof(double)));
    double bprev=0.0;
    for(int j=0;j<m;j++){
        CB(cublasDsymv(g_cublas,CUBLAS_FILL_MODE_LOWER,N,&one,g_H,N,dv,1,&zero,dw,1));
        double alpha; CB(cublasDdot(g_cublas,N,dv,1,dw,1,&alpha)); a[j]=alpha;
        double na=-alpha; CB(cublasDaxpy(g_cublas,N,&na,dv,1,dw,1));
        if(j>0){ double nb=-bprev; CB(cublasDaxpy(g_cublas,N,&nb,scratch,1,dw,1)); }
        CB(cublasDnrm2(g_cublas,N,dw,1,&nrm)); b[j]=nrm;
        CK(cudaMemcpy(scratch,dv,N*sizeof(double),cudaMemcpyDeviceToDevice));
        if(nrm>1e-14){ double sc=1.0/nrm; CB(cublasDscal(g_cublas,N,&sc,dw,1)); }
        CK(cudaMemcpy(dv,dw,N*sizeof(double),cudaMemcpyDeviceToDevice));
        bprev=nrm;
    }
    std::vector<double> T((size_t)m*m,0.0);
    for(int i=0;i<m;i++){ T[(size_t)i*m+i]=a[i]; if(i+1<m){ T[(size_t)i*m+(i+1)]=b[i]; T[(size_t)(i+1)*m+i]=b[i]; } }
    double *dT,*dTe; int *dinf; CK(cudaMalloc(&dT,(size_t)m*m*sizeof(double))); CK(cudaMalloc(&dTe,m*sizeof(double))); CK(cudaMalloc(&dinf,sizeof(int)));
    CK(cudaMemcpy(dT,T.data(),(size_t)m*m*sizeof(double),cudaMemcpyHostToDevice));
    int lw=0; CS(cusolverDnDsyevd_bufferSize(g_cusolver,CUSOLVER_EIG_MODE_NOVECTOR,CUBLAS_FILL_MODE_LOWER,m,dT,m,dTe,&lw));
    double* w2; CK(cudaMalloc(&w2,lw*sizeof(double))); CS(cusolverDnDsyevd(g_cusolver,CUSOLVER_EIG_MODE_NOVECTOR,CUBLAS_FILL_MODE_LOWER,m,dT,m,dTe,w2,lw,dinf));
    std::vector<double> te(m); CK(cudaMemcpy(te.data(),dTe,m*sizeof(double),cudaMemcpyDeviceToHost));
    lambda1 = te[0];
    upper   = te[m-1] + b[m-1];
    lower   = lambda1 + (upper - lambda1) * (double)NEVEX / (double)N;
    CK(cudaFree(dT)); CK(cudaFree(dTe)); CK(cudaFree(dinf)); CK(cudaFree(w2));
}

// One R-ChFSI outer iteration (arXiv:2503.22652), after DFT-FE reformulatedChebyshevFilter
// (https://github.com/dftfeDevelopers/dftfe). Returns the largest wanted residual.
static double rchfsi_outer(double* dV,double* dZc,double* dZp,double* dW,double* dR,
                           double* dVcur,double* dG,double* dA,double* dZ,double* dritz,double* dLk,
                           double* dresid,std::vector<double>& hritz,
                           double lower,double upper,double mu,
                           double* solw,int lw,int* dinfo,int fm,int& locked){
    double one=1.0,zero=0.0;
    double c=0.5*(upper+lower), e=0.5*(upper-lower);
    int nk=N*NEVEX; size_t NKb=(size_t)nk*sizeof(double);
    dim3 cg((N+255)/256, NEVEX);

    if(fm==0) resid_opt<<<g_hgrid,g_hblk>>>(dR,g_H,dV,dritz);
    else      resid_gold<<<g_hgrid,g_hblk>>>(dR,g_H,dV,dritz);
    k_colnorm<<<NEVEX,256>>>(dR,dresid);
    std::vector<double> rs(NEVEX); CK(cudaMemcpy(rs.data(),dresid,NEVEX*sizeof(double),cudaMemcpyDeviceToHost));
    int nl=0; double maxr=0; for(int q=0;q<NEV;q++){ if(rs[q]<TOL) nl++; maxr=std::max(maxr,rs[q]); } locked=nl;
    CK(cudaMemcpy(hritz.data(),dritz,NEVEX*sizeof(double),cudaMemcpyDeviceToHost));

    double sig1=e/(c-mu), sig_k=sig1;
    CK(cudaMemcpy(dZc,dR,NKb,cudaMemcpyDeviceToDevice));
    { double scl=sig1/e; CB(cublasDscal(g_cublas,nk,&scl,dZc,1)); }
    CK(cudaMemset(dZp,0,NKb));
    std::vector<double> Lkm1(NEVEX,1.0),Lk(NEVEX),Lkp1(NEVEX),inj(NEVEX);
    for(int j=0;j<NEVEX;j++) Lk[j]=(sig1/e)*(hritz[j]-c);
    for(int step=1; step<MAX_DEG; step++){
        double sig_kp1=1.0/(2.0/sig1 - sig_k);
        double a=2.0*sig_kp1/e, b=-2.0*sig_kp1*c/e, cc=-sig_k*sig_kp1;
        HEMM(dW,dZc,a,0.0,fm);
        CB(cublasDaxpy(g_cublas,nk,&b,dZc,1,dW,1));
        CB(cublasDaxpy(g_cublas,nk,&cc,dZp,1,dW,1));
        for(int j=0;j<NEVEX;j++) inj[j]=a*Lk[j];
        CK(cudaMemcpy(dLk,inj.data(),NEVEX*sizeof(double),cudaMemcpyHostToDevice));
        k_col_axpy<<<cg,256>>>(dW,dR,dLk);
        for(int j=0;j<NEVEX;j++) Lkp1[j]=a*hritz[j]*Lk[j]+b*Lk[j]+cc*Lkm1[j];
        CK(cudaMemcpy(dZp,dZc,NKb,cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(dZc,dW,NKb,cudaMemcpyDeviceToDevice));
        Lkm1=Lk; Lk=Lkp1; sig_k=sig_kp1;
    }
    CK(cudaMemcpy(dLk,Lk.data(),NEVEX*sizeof(double),cudaMemcpyHostToDevice));
    k_recon<<<cg,256>>>(dVcur,dV,dLk,dZc);
    CK(cudaMemcpy(dV,dVcur,NKb,cudaMemcpyDeviceToDevice));

    if(fm==0) gram_opt<<<g_ggrid,g_gblk>>>(dG,dV);
    else      gram_gold<<<g_ggrid,g_gblk>>>(dG,dV);
    CK(cudaDeviceSynchronize());
    CS(cusolverDnDpotrf(g_cusolver,CUBLAS_FILL_MODE_LOWER,NEVEX,dG,NEVEX,solw,lw,dinfo));
    CB(cublasDtrsm(g_cublas,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,N,NEVEX,&one,dG,NEVEX,dV,N));

    if(fm==0) rrhv_opt<<<g_hgrid,g_hblk>>>(dW,g_H,dV);
    else      rrhv_gold<<<g_hgrid,g_hblk>>>(dW,g_H,dV);
    if(fm==0) rrproj_opt<<<g_ggrid,g_gblk>>>(dA,dV,dW);
    else      rrproj_gold<<<g_ggrid,g_gblk>>>(dA,dV,dW);
    CS(cusolverDnDsyevd(g_cusolver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,NEVEX,dA,NEVEX,dritz,solw,lw,dinfo));
    CK(cudaMemcpy(dZ,dA,(size_t)NEVEX*NEVEX*sizeof(double),cudaMemcpyDeviceToDevice));
    CB(cublasDgemm(g_cublas,CUBLAS_OP_N,CUBLAS_OP_N,N,NEVEX,NEVEX,&one,dV,N,dZ,NEVEX,&zero,dVcur,N));
    CK(cudaMemcpy(dV,dVcur,NKb,cudaMemcpyDeviceToDevice));
    return maxr;
}

// FP64 Cholesky-QR and Rayleigh-Ritz: the first residual needs Ritz values.
static void seed_ritz(double* dV,double* dW,double* dVcur,double* dG,double* dA,double* dZ,
                      double* dritz,double* solw,int lw,int* dinfo){
    double one=1.0,zero=0.0;
    gram_gold<<<g_ggrid,g_gblk>>>(dG,dV); CK(cudaDeviceSynchronize());
    CS(cusolverDnDpotrf(g_cusolver,CUBLAS_FILL_MODE_LOWER,NEVEX,dG,NEVEX,solw,lw,dinfo));
    CB(cublasDtrsm(g_cublas,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,N,NEVEX,&one,dG,NEVEX,dV,N));
    CB(cublasDgemm(g_cublas,CUBLAS_OP_N,CUBLAS_OP_N,N,NEVEX,N,&one,g_H,N,dV,N,&zero,dW,N));
    CB(cublasDgemm(g_cublas,CUBLAS_OP_T,CUBLAS_OP_N,NEVEX,NEVEX,N,&one,dV,N,dW,N,&zero,dA,NEVEX));
    CS(cusolverDnDsyevd(g_cusolver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,NEVEX,dA,NEVEX,dritz,solw,lw,dinfo));
    CK(cudaMemcpy(dZ,dA,(size_t)NEVEX*NEVEX*sizeof(double),cudaMemcpyDeviceToDevice));
    CB(cublasDgemm(g_cublas,CUBLAS_OP_N,CUBLAS_OP_N,N,NEVEX,NEVEX,&one,dV,N,dZ,NEVEX,&zero,dVcur,N));
    CK(cudaMemcpy(dV,dVcur,(size_t)N*NEVEX*sizeof(double),cudaMemcpyDeviceToDevice));
}

int main(int argc,char**argv){
    setbuf(stdout,NULL);
    int fmode=0;                  // 1 (--gold): the un-annotated kernels
    bool profiling=false;
    for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"--gold")) fmode=1; else if(!strcmp(argv[i],"--profiling")) profiling=true; }
    printf("chase_real: N=%d NEV=%d NEX=%d NEVEX=%d (R-ChFSI)\n",N,NEV,NEX,NEVEX);
    CK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 4ULL<<30)); // the reverse-mode tape lives in the device heap
    CB(cublasCreate(&g_cublas)); CS(cusolverDnCreate(&g_cusolver));
    size_t NN=(size_t)N*N, NK=(size_t)N*NEVEX;
    g_hblk=dim3(16,16); g_hgrid=dim3((N+15)/16,(NEVEX+15)/16);
    g_gblk=dim3(16,16); g_ggrid=dim3((NEVEX+15)/16,(NEVEX+15)/16);

    // Planted spectrum: the wanted NEV in [0.1, 1], the rest in [3, 30].
    std::vector<double> lam(N);
    for(int i=0;i<N;i++) lam[i]=(i<NEV)? 0.1+0.9*(double)i/std::max(1,NEV-1) : 3.0+27.0*(double)(i-NEV)/std::max(1,N-NEV-1);
    double *dQ,*dM,*dlam; CK(cudaMalloc(&dQ,NN*sizeof(double))); CK(cudaMalloc(&dM,NN*sizeof(double)));
    CK(cudaMalloc(&g_H,NN*sizeof(double))); CK(cudaMalloc(&dlam,N*sizeof(double)));
    CK(cudaMemcpy(dlam,lam.data(),N*sizeof(double),cudaMemcpyHostToDevice));
    { std::vector<double> hA(NN); srand(12345); for(size_t i=0;i<NN;i++) hA[i]=(double)rand()/RAND_MAX-0.5;
      CK(cudaMemcpy(dQ,hA.data(),NN*sizeof(double),cudaMemcpyHostToDevice));
      double*tau,*wq; int*inf,lwq=0; CS(cusolverDnDgeqrf_bufferSize(g_cusolver,N,N,dQ,N,&lwq));
      CK(cudaMalloc(&tau,N*sizeof(double))); CK(cudaMalloc(&inf,sizeof(int))); CK(cudaMalloc(&wq,lwq*sizeof(double)));
      CS(cusolverDnDgeqrf(g_cusolver,N,N,dQ,N,tau,wq,lwq,inf));
      int lwo=0; CS(cusolverDnDorgqr_bufferSize(g_cusolver,N,N,N,dQ,N,tau,&lwo)); if(lwo>lwq){CK(cudaFree(wq));CK(cudaMalloc(&wq,lwo*sizeof(double)));lwq=lwo;}
      CS(cusolverDnDorgqr(g_cusolver,N,N,N,dQ,N,tau,wq,lwq,inf)); CK(cudaFree(tau));CK(cudaFree(inf));CK(cudaFree(wq)); }
    { int t=256,b=(N+t-1)/t; dim3 g(b,N); k_scale_cols<<<g,t>>>(dQ,dlam,dM,N); }
    { double one=1.0,zero=0.0; CB(cublasDgemm(g_cublas,CUBLAS_OP_N,CUBLAS_OP_T,N,N,N,&one,dM,N,dQ,N,&zero,g_H,N)); }
    { dim3 b(16,16),g((N+15)/16,(N+15)/16); k_symmetrize<<<g,b>>>(g_H,N); } CK(cudaDeviceSynchronize());

    double *dV,*dVp,*dW,*dV0,*dG,*dA,*dZ,*dritz,*dresid;
    CK(cudaMalloc(&dV,NK*sizeof(double)));  CK(cudaMalloc(&dVp,NK*sizeof(double)));
    CK(cudaMalloc(&dW,NK*sizeof(double)));  CK(cudaMalloc(&dV0,NK*sizeof(double)));
    CK(cudaMalloc(&dG,(size_t)NEVEX*NEVEX*sizeof(double))); CK(cudaMalloc(&dA,(size_t)NEVEX*NEVEX*sizeof(double)));
    CK(cudaMalloc(&dZ,(size_t)NEVEX*NEVEX*sizeof(double))); CK(cudaMalloc(&dritz,NEVEX*sizeof(double)));
    CK(cudaMalloc(&dresid,NEVEX*sizeof(double)));
    double *dD; CK(cudaMalloc(&dD,NK*sizeof(double)));
    double *dZp,*dVcur,*dLk;
    CK(cudaMalloc(&dZp,NK*sizeof(double))); CK(cudaMalloc(&dVcur,NK*sizeof(double))); CK(cudaMalloc(&dLk,NEVEX*sizeof(double)));
    std::vector<double> hV0(NK); srand(777); for(size_t i=0;i<NK;i++) hV0[i]=(double)rand()/RAND_MAX-0.5;
    CK(cudaMemcpy(dV0,hV0.data(),NK*sizeof(double),cudaMemcpyHostToDevice));

    int lw_po=0,lw_ev=0; CS(cusolverDnDpotrf_bufferSize(g_cusolver,CUBLAS_FILL_MODE_LOWER,NEVEX,dG,NEVEX,&lw_po));
    CS(cusolverDnDsyevd_bufferSize(g_cusolver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,NEVEX,dA,NEVEX,dritz,&lw_ev));
    int lw=std::max(lw_po,lw_ev); double* solw; int* dinfo; CK(cudaMalloc(&solw,lw*sizeof(double))); CK(cudaMalloc(&dinfo,sizeof(int)));

    double lower,upper,lambda1; lanczos_bounds(dW,dVp,dV,lower,upper,lambda1);
    printf("Lanczos bounds: lambda1=%.4f lower=%.4f upper=%.4f (true min=%.4f max=%.4f)\n",lambda1,lower,upper,lam[0],lam[N-1]);

    auto solve=[&](int fm)->void{
        CK(cudaMemcpy(dV,dV0,NK*sizeof(double),cudaMemcpyDeviceToDevice));
        int locked=0; std::vector<double> hritz(NEVEX);
        cudaEvent_t _t0,_t1; cudaEventCreate(&_t0); cudaEventCreate(&_t1);
        CK(cudaDeviceSynchronize()); cudaEventRecord(_t0);
        seed_ritz(dV,dW,dVcur,dG,dA,dZ,dritz,solw,lw,dinfo);
        double mu = lambda1;
        int iters=0; double last_maxr=0.0; bool nonfinite=false;
        for(int it=0; it<MAX_OUTER && locked<NEV; it++){
            double maxr = rchfsi_outer(dV,dVp,dZp,dW,dD,dVcur,dG,dA,dZ,dritz,dLk,dresid,
                                       hritz,lower,upper,mu,solw,lw,dinfo,fm,locked);
            CK(cudaMemcpy(hritz.data(),dritz,NEVEX*sizeof(double),cudaMemcpyDeviceToHost));
            // The unscaled recurrence overflows at high degree and Rayleigh-Ritz returns NaN.
            {
                int bad = -1;
                for (int j = 0; j < NEVEX; ++j)
                    if (!std::isfinite(hritz[j])) { bad = j; break; }
                if (bad >= 0) {
                    nonfinite = true;
                    // Under the condition-number probe a perturbed site may diverge; the
                    // non-finite metric is the measurement.
                    if (profiling) {
                        fprintf(stderr,
                            "[guard] non-finite Ritz at iter %d under profiling; continuing so the "
                            "condition-number probe scores this perturbation (this is expected for the "
                            "eps-perturbed Gram site)\n", it);
                    } else {
                    fprintf(stderr,
                        "\nFATAL: non-finite Ritz value (index %d) at outer iteration %d.\n"
                        "Cause: Chebyshev recurrence overflow at this degree and spectral\n"
                        "interval. The unscaled three-term recurrence in this bench is\n"
                        "numerically valid only for deg <~ 70 with lower=%.4f upper=%.4f;\n"
                        "this run used deg=%d. Use a lower filter degree.\n",
                        bad, it, lower, upper, MAX_DEG);
                    exit(7);
                    }
                }
            }
            mu = hritz[0]; iters=it+1; last_maxr=maxr;
            if(profiling && it>=8) break;
            printf("  iter %2d: locked=%d/%d  max-resid=%.3e ritz[0]=%.6f\n",it,locked,NEV,maxr,hritz[0]);
        }
        // Quantity of interest for the condition-number probe: the residual that decides locking.
        if(profiling) poseidon_metric("max_resid",
            nonfinite ? std::numeric_limits<double>::quiet_NaN() : last_maxr);
        cudaEventRecord(_t1); CK(cudaEventSynchronize(_t1));
        float _ms=0; cudaEventElapsedTime(&_ms,_t0,_t1);
        std::vector<double> rz(NEVEX); CK(cudaMemcpy(rz.data(),dritz,NEVEX*sizeof(double),cudaMemcpyDeviceToHost));
        double se=0,st=0; for(int i=0;i<NEV;i++){ double d=rz[i]-lam[i]; se+=d*d; st+=lam[i]*lam[i]; }
        printf("%s eigval L2 vs planted = %.6e (locked %d/%d)\n", fm?"GOLD":"OPT", sqrt(se/st), locked, NEV);
        printf("%s solve_ms = %.3f (iters=%d)\n", fm?"GOLD":"OPT", _ms, iters);
        { std::vector<double> rsq(NEVEX);
          CK(cudaMemcpy(rsq.data(),dresid,NEVEX*sizeof(double),cudaMemcpyDeviceToHost));
          printf("%s EIGRESID =", fm?"GOLD":"OPT");
          for(int q=0;q<NEV;q++) printf(" %.3e", rsq[q]);
          printf("\n"); }
    };

    if(profiling){
        solve(0); printf("Profiling done.\n"); return 0;
    }
    solve(fmode);
    return 0;
}
