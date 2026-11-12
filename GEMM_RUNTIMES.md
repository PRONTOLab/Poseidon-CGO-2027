# GEMM runtimes: what the compiled programs call

A matrix product Poseidon rewrites into a library call is dispatched, at run
time, to one of three runtimes of the compiler, built into
`libposeidon_rt.a`, which `poseidon-clang++ -poseidon-profile-use` links
(sources under `enzyme/Poseidon/runtime/` of the compiler; the reference arms of
the benchmarks compile `OZAKI_RT`, `TCEC_RT` and `DIRECT_RT` from there with
plain clang, as `scripts/detect.sh` resolves them):

| runtime | entry points | what it computes |
|---|---|---|
| `runtime/ozaki/ozaki_rt.cu` | `__poseidon_ozaki_dgemm`, `__poseidon_ozaki_dgemm_ex` | Ozaki-II (INT8 tensor cores + residue arithmetic) with `nm = 3 ... 14` moduli: the fast mode of Ozawa, Uchino and Imamura (GEMMul8 v3.1.1), whose result it returns bit for bit; `nm = 0` is a cuBLAS DGEMM (the native-DGEMM candidate) |
| `runtime/tcec/tcec_rt.cu` | `__poseidon_tcec_dgemm`, `__poseidon_tcec_dgemm_ex` | an error-corrected FP16 tensor-core GEMM (FP16TCEC), delegated to cuMpSGEMM in the configuration the results come from (`-DPOSEIDON_TCEC_USE_CUMPSGEMM=ON`, `scripts/build_cumpsgemm.sh`); without it, a built-in three-GEMM cuBLAS backend whose accumulated error grows with K |
| `runtime/direct/direct_rt.cu` | `__poseidon_direct_dgemm`, `__poseidon_direct_dgemm_ex` | one rounding of each operand to FP16/BF16/TF32 and one `cublasGemmEx` with an FP32 accumulator |

The compiler prices each family from cost-model rows that `poseidon-calibrate`
measures (`ozaki_dispatch_rel,<nm>` and, for a product whose extents are
compile-time constants, `ozaki_dispatch_rect_rel,dgemm_m<M>n<N>k<K>`;
`tcec_dispatch_rel`; `direct_dispatch_rel,<class>`; the in-kernel raises
`wmma_inkernel_rel,<class>`), all present in the shipped
`cost_models/cm_sm_120_RTX5090.csv`. The Ozaki-II runtime's bit-for-bit match
with GEMMul8's fast mode is a test of the compiler
(`test/Poseidon/Integration/ozaki_published_shift.cu`).
