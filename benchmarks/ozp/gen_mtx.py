#!/usr/bin/env python3
# Generate a gapped symmetric Hamiltonian H = Q diag(D) Q^T and its exact
# projector K (occupied eigenvectors), in MatrixMarket symmetric-coordinate
# format that ozp's read_mm understands. nel = N/2 (half filling).
# Usage: gen_mtx.py <N> <out_prefix> [gap]
import sys, numpy as np

N = int(sys.argv[1])
prefix = sys.argv[2]
gap = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
rng = np.random.default_rng(12345)

# orthonormal Q
Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
nocc = N // 2
D = np.empty(N)
D[:nocc] = rng.uniform(-1.0, -gap, nocc)      # occupied
D[nocc:] = rng.uniform(gap, 1.0, N - nocc)     # virtual
H = (Q * D) @ Q.T
H = 0.5 * (H + H.T)
occ = np.zeros(N); occ[:nocc] = 1.0
K = (Q * occ) @ Q.T
K = 0.5 * (K + K.T)

def write_sym(fn, M):
    n = M.shape[0]
    ii, jj = np.tril_indices(n)                # lower triangle incl diagonal
    vals = M[ii, jj]
    with open(fn, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real symmetric\n")
        f.write(f"{n} {n} {ii.size}\n")
        # 1-indexed row col value
        np.savetxt(f, np.column_stack([ii + 1, jj + 1, vals]),
                   fmt="%d %d %.17g")

write_sym(f"{prefix}_h.mtx", H)
write_sym(f"{prefix}_k.mtx", K)
print(f"wrote {prefix}_h.mtx / {prefix}_k.mtx  N={N} nel={nocc} gap={gap}")
