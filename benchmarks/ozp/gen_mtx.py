#!/usr/bin/env python3
"""gen_mtx.py <N> <out_prefix> [gap]: gapped symmetric H and its half-filling projector K (MatrixMarket)."""
import sys, numpy as np

N = int(sys.argv[1])
prefix = sys.argv[2]
gap = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
rng = np.random.default_rng(12345)

Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
nocc = N // 2
D = np.empty(N)
D[:nocc] = rng.uniform(-1.0, -gap, nocc)
D[nocc:] = rng.uniform(gap, 1.0, N - nocc)
H = (Q * D) @ Q.T
H = 0.5 * (H + H.T)
occ = np.zeros(N); occ[:nocc] = 1.0
K = (Q * occ) @ Q.T
K = 0.5 * (K + K.T)

def write_sym(fn, M):
    n = M.shape[0]
    ii, jj = np.tril_indices(n)
    vals = M[ii, jj]
    with open(fn, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real symmetric\n")
        f.write(f"{n} {n} {ii.size}\n")
        np.savetxt(f, np.column_stack([ii + 1, jj + 1, vals]),
                   fmt="%d %d %.17g")

write_sym(f"{prefix}_h.mtx", H)
write_sym(f"{prefix}_k.mtx", K)
print(f"wrote {prefix}_h.mtx / {prefix}_k.mtx  N={N} nel={nocc} gap={gap}")
