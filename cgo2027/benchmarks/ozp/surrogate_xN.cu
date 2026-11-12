// Surrogate-generality check: the per-matmul precision Poseidon selects from a
// CHEAP profile (N=512) is a scale-invariant property of the operation, so it
// should clear the chemistry bar at every DEPLOY scale. This runs the full TRS4
// purification at a fixed uniform precision and reports RMSD-vs-Kref + time, so
// a driver can sweep N in {512,1024,2048,4096} x {fp64,nm8,nm10}:
//   nm10 (the surrogate-selected, chemically-sufficient rung) clears RMSD<=1e-7
//   at all N ; nm8 (one order cheaper) misses at all N -> the choice transfers.
#include "include/matrix_arithmetic.h"
#include "include/matrix_helpers.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#ifndef OZP_N
#define OZP_N 512
#endif
#define N OZP_N
#define NN ((size_t)N * N)
#define MAXIT 30
typedef double dfloat;
#define CK(x) do{cudaError_t e_=(x);if(e_!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);}}while(0)
extern "C" void __poseidon_ozaki_dgemm(double*,const double*,const double*,int,int,int,int,int,int,double,double,cudaStream_t,int);
enum Method { M_FP64, M_CUBLAS, M_OZ };
static cublasHandle_t g_cublas = nullptr;
static int g_nm = 10; // num_moduli for M_OZ (set from the "nm<int>" arg)

__global__ void g_fp64(dfloat*C,const dfloat*A,const dfloat*B){
  int r=blockIdx.x*blockDim.x+threadIdx.x,c=blockIdx.y*blockDim.y+threadIdx.y;
  if(r>=N||c>=N)return;const dfloat*ar=A+(size_t)r*N;double v=0;
#pragma unroll 1
  for(int k=0;k<N;k++)v+=ar[k]*B[(size_t)k*N+c];C[(size_t)r*N+c]=v;
}
__global__ void combine2(dfloat*o,const dfloat*A,const dfloat*B,dfloat a,dfloat b){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=NN)return;o[i]=a*A[i]+b*B[i];}
__global__ void add_diag(dfloat*M,dfloat c){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;M[(size_t)i*N+i]+=c;}
__global__ void symm(dfloat*P){int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y*blockDim.y+threadIdx.y;if(i>=N||j>=N||i>=j)return;dfloat a=0.5*(P[(size_t)i*N+j]+P[(size_t)j*N+i]);P[(size_t)i*N+j]=a;P[(size_t)j*N+i]=a;}
__global__ void pdot(const dfloat*A,const dfloat*B,dfloat*r,size_t nn){__shared__ dfloat sh[256];size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;sh[threadIdx.x]=(i<nn)?A[i]*B[i]:0.0;__syncthreads();for(int s=blockDim.x/2;s>0;s>>=1){if(threadIdx.x<(unsigned)s)sh[threadIdx.x]+=sh[threadIdx.x+s];__syncthreads();}if(threadIdx.x==0)atomicAdd(r,sh[0]);}
static dim3 G2((NN+255)/256),B2(256),GG((N+15)/16,(N+15)/16),GB(16,16);
static double dpdot(const dfloat*a,const dfloat*b,dfloat*s){CK(cudaMemset(s,0,8));pdot<<<G2,B2>>>(a,b,s,NN);dfloat h;CK(cudaMemcpy(&h,s,8,cudaMemcpyDeviceToHost));return h;}
static void do_gemm(Method m,dfloat*C,const dfloat*A,const dfloat*B){
  switch(m){
    case M_FP64: g_fp64<<<GG,GB>>>(C,A,B); break;
    case M_CUBLAS: { // row-major C=A@B via column-major C^T=B^T A^T (square N)
      const double al=1.0, be=0.0;
      cublasDgemm(g_cublas,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&al,B,N,A,N,&be,C,N);
      break; }
    case M_OZ:   __poseidon_ozaki_dgemm(C,A,B,N,N,N,N,0,0,1.0,0.0,0,g_nm); break;
  }
}
struct Bufs{dfloat *X,*X2,*fx,*gx,*g2,*tmp,*scal;};
static void trs4(Method m,Bufs&b,double nel,const dfloat*dX0){
  const double SMIN=0,SMAX=6,CONV=1e-8;
  CK(cudaMemcpy(b.X,dX0,NN*8,cudaMemcpyDeviceToDevice));
  for(int it=0;it<MAXIT;it++){
    do_gemm(m,b.X2,b.X,b.X);
    combine2<<<G2,B2>>>(b.fx,b.X,b.X2,4.0,-3.0);
    combine2<<<G2,B2>>>(b.gx,b.X2,b.X,1.0,-2.0);
    add_diag<<<(N+255)/256,256>>>(b.gx,1.0);
    double tfx=dpdot(b.X2,b.fx,b.scal),tgx=dpdot(b.X2,b.gx,b.scal);
    double sigma=(fabs(tgx)<CONV/10)?0.5*(SMAX-SMIN):(nel-tfx)/tgx;
    if(sigma>SMAX) combine2<<<G2,B2>>>(b.tmp,b.X,b.X2,2.0,-1.0);
    else if(sigma<SMIN) CK(cudaMemcpy(b.tmp,b.X2,NN*8,cudaMemcpyDeviceToDevice));
    else { combine2<<<G2,B2>>>(b.g2,b.fx,b.gx,1.0,sigma); do_gemm(m,b.tmp,b.X2,b.g2); }
    CK(cudaMemcpy(b.X,b.tmp,NN*8,cudaMemcpyDeviceToDevice));
    symm<<<GG,GB>>>(b.X);
  }
}

int main(int argc,char**argv){
  if(argc<4){printf("usage: %s H.mtx K.mtx <fp64|nm8|nm10> [nel]\n",argv[0]);return 1;}
  Method m;
  if (!strcmp(argv[3],"fp64")) m=M_FP64;
  else if (!strcmp(argv[3],"cublas")) m=M_CUBLAS;
  else if (!strncmp(argv[3],"nm",2)) { m=M_OZ; g_nm=atoi(argv[3]+2); }
  else { printf("unknown method %s (use fp64|cublas|nm<int>)\n",argv[3]); return 1; }
  double nel=argc>4?atof(argv[4]):N/2;
  CK(cudaDeviceSetLimit(cudaLimitMallocHeapSize,8ULL<<30));
  cublasCreate(&g_cublas);
  unsigned sz=0,szk=0;double*H=read_mm<double>(std::string(argv[1]),sz);double*Kref=read_mm<double>(std::string(argv[2]),szk);
  if(sz!=(unsigned)N){printf("size %u != N=%d\n",sz,N);return 1;}
  double hmin,hmax;get_eig_circle(H,N,hmin,hmax);
  std::vector<double> X0(NN);double inv=1.0/(hmax-hmin);
  for(int i=0;i<N;i++)for(int j=0;j<N;j++)X0[(size_t)i*N+j]=((i==j?hmax:0.0)-H[(size_t)i*N+j])*inv;
  Bufs b; auto al=[&](dfloat**p){CK(cudaMalloc(p,NN*8));};
  al(&b.X);al(&b.X2);al(&b.fx);al(&b.gx);al(&b.g2);al(&b.tmp);CK(cudaMalloc(&b.scal,8));
  dfloat*dX0;al(&dX0);CK(cudaMemcpy(dX0,X0.data(),NN*8,cudaMemcpyHostToDevice));
  // accuracy
  trs4(m,b,nel,dX0);CK(cudaDeviceSynchronize());
  std::vector<double> hx(NN);CK(cudaMemcpy(hx.data(),b.X,NN*8,cudaMemcpyDeviceToHost));
  double sse=0;for(size_t i=0;i<NN;i++){double d=hx[i]-Kref[i];sse+=d*d;}
  double rmsd=sqrt(sse/(double)NN);
  // timing
  cudaEvent_t a,e;CK(cudaEventCreate(&a));CK(cudaEventCreate(&e));
  for(int w=0;w<1;w++) trs4(m,b,nel,dX0);
  CK(cudaDeviceSynchronize());CK(cudaEventRecord(a));
  int IT=(m==M_FP64?2:5); for(int i=0;i<IT;i++) trs4(m,b,nel,dX0);
  CK(cudaEventRecord(e));CK(cudaEventSynchronize(e));
  float ms=0;CK(cudaEventElapsedTime(&ms,a,e));ms/=IT;
  printf("N=%d method=%s RMSD=%.3e time_ms=%.3f\n",N,argv[3],rmsd,ms);
  return 0;
}
