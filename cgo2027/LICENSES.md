# Licences and provenance

This file lists every part of the artifact archive that was not written for
it, where it comes from, its licence and what that licence asks for. Where a
file does not state its origin, the entry says so and gives the evidence for
the origin listed.

## The artifact's own files

Everything in `cgo2027/` not listed under "Third-party code" below was written
for this artifact: the driver and pipeline scripts, Makefiles, the benchmark
drivers and harnesses around the ported kernels, the plotting scripts, the
documentation, and the data (cost models, profiles, Herbie answers, golden
outputs, `expected/`). The Herbie answers are output of Herbie, not Herbie
code.

Licence of these files: **TODO (owner): choose and add a top-level `LICENSE`.**

## Third-party code in `cgo2027/`

### `benchmarks/fig2_drift/mX_real/` (whole directory)

- Origin: not stated in the tree. The directory is identical to
  https://github.com/RIKEN-RCCS/mX_real at commit
  `72a35d697b247fb44400a282a88e22ea3211a327` (2025-01-04), minus one file of
  it (`Ozaki-QW/a.awk`).
- Licence: BSD 2-Clause, Copyright (c) 2023-24, RIKEN (`mX_real/LICENSE`).
- Requires: the copyright notice, conditions and disclaimer kept with the
  source. Done: `mX_real/LICENSE` is shipped.
- Note: `mX_real/Ozaki-QW/all_functions.c` carries the upstream author's
  comment "Debug is not sufficiently done, so please use it carefully and not
  spread it." The file is distributed publicly by the upstream repository under
  the licence above. `fig2_drift` does not compile it (`mX_real.hpp` includes
  `Ozaki-QW/fp_const.hpp` and `Ozaki-QW/qxw.hpp` only).

### `benchmarks/genga_nbody/genga_nbody.cu`, `genga_kepler.cu`, `genga_unified.cu`

- Origin (stated in each file): the per-pair acceleration (`accEncFull`,
  `source/directAcc.h`) and the Kepler drift (`fastfg`, `source/FG2.h`) of
  GENGA, https://bitbucket.org/sigrimm/genga (Grimm and Stadel 2014). The
  tiling, drivers, inputs and harnesses are this artifact's own.
- Licence (GENGA `Documentation.md`, section "License"): "GENGA is free to use,
  but when results from GENGA are published, then the following paper has to
  be referenced: The GENGA Code: Gravitational Encounters in N-body
  simulations with GPU Acceleration (Grimm & Stadel 2014)."
- Requires: citing Grimm and Stadel 2014. Done: each file and the paper cite
  it. The licence text says nothing further about redistribution.

### `benchmarks/ozp/include/matrix_arithmetic.h`, `matrix_arithmetic_imp.h`, `matrix_helpers.h`, `matrix_helpers_imp.h`

- Origin: not stated in these files (`ozp_poseidon.cu` says it "Reuses ozp's
  read_mm + get_eig_circle"). The four files are byte-identical to `include/`
  of https://gitlab.com/wddawson/ozp at commit
  `9084fc5a3b67dfb6a0e139003bff652dbbc4c5a9` (2024-12-20).
- Licence: none found. The ozp repository at that commit has no licence file
  and its `ReadMe.md` states none.
- Open: redistribution needs the ozp author's permission (or a licence added
  upstream), or these four files are replaced.

### `benchmarks/ozp/ozp_poseidon.cu`

- Origin (stated in the file): the TRS4 purification recurrence follows ozp's
  `include/purify_imp.h` (same repository as above); the file is written for
  this artifact. Same open point as above if any code text was copied.

### `benchmarks/chase_eigensolver/chase_real.cu`

- Origin (stated in the file): "Faithful pieces ported from ChASE_ref/", a
  port of the GPU path of ChASE, https://github.com/ChASE-library/ChASE
  (Lanczos bounds, Chebyshev filter, Cholesky-QR, Rayleigh-Ritz, residual,
  locking), with the R-ChFSI reformulation (arXiv:2503.22652) following
  DFT-FE's `reformulatedChebyshevFilter` (https://github.com/dftfeDevelopers/dftfe,
  commit 621294a0, cited line by line in the file).
- ChASE licence: BSD 3-Clause, "Copyright (c) 2015, Simulation and Data
  Laboratory Quantum Materials -- Forschungszentrum Juelich GmbH". Requires,
  for redistributed source, the copyright notice, conditions and disclaimer.
  The file does not reproduce them.
- DFT-FE licence: not checked here (no copy at hand); the file cites DFT-FE as
  the reference of the algorithm.
- Open: if the port contains ChASE source text (rather than a CUDA
  reimplementation of its algorithm), add ChASE's BSD 3-Clause notice to the
  file or next to it.

## Sources next to the artifact (`sources/`)

| Source | Upstream | Commit | Licence | Notes |
|---|---|---|---|---|
| `sources/Enzyme` (git submodule) | Enzyme, https://github.com/EnzymeAD/Enzyme (Poseidon is `enzyme/Poseidon`) | `9e2e0572931248b4610940f2906e583914f9913e` | Apache License 2.0 with LLVM Exceptions (`LICENSE`) | See "Inside the compiler" below. |
| `sources/cuMpSGEMM` (git submodule) | cuMpSGEMM, https://github.com/enp1s0/cuMpSGEMM | `42dd92ea1c0e3d603bbb3f8f354f53e0967a9f4b` | MIT, Copyright (c) 2022 enp1s0 (`LICENSE`) | Submodules at the commits it records: `cutf` `97339d83` (MIT, `LICENSE`), `wmma_extension` `03772a7f` (MIT, `LICENSE`), `cuGEMM-Mx2x2` `1f68ec6d` (https://github.com/enp1s0/cuGEMM-Mx2x2: no licence file found). |
| `sources/herbie` (git submodule) and `sources/herbie-num-taylor-terms.patch` | Herbie, https://github.com/herbie-fp/herbie | `73ba1fe76f97d4cdb852a602addd82e3a983b209`; the patch adds the `--num-taylor-terms` option (written for this artifact, same licence) | MIT, Copyright (c) 2015-2024 Herbie Project, modified work Copyright 2016 Google Inc. (`LICENSE.md`) | Needed only for `GENGA_HERBIE=run`. |

LLVM is not shipped. The compiler is built against llvm-project
(https://github.com/llvm/llvm-project) at commit
`9d618fc0243103263ebf9bdcd08ef25e871d9c78`, Apache License 2.0 with LLVM
Exceptions.

### Inside the compiler

The compiler's files state their own provenance; the ones that follow external
code:

- `enzyme/Poseidon/runtime/ozaki/ozaki_rt.cu`: derived from GEMMul8
  (https://github.com/RIKEN-RCCS/GEMMul8), "Copyright (c) 2025- RIKEN R-CCS,
  MIT License", as the file states. The MIT licence asks that its copyright and
  permission notice be included in copies or substantial portions; the file
  names the copyright and the licence but does not reproduce the permission
  notice.
- `enzyme/Poseidon/Core/ExpansionRoutines.h`: reproduces MultiFloats.jl
  routines (commit 28041a3, MIT License), as the file states, with the same
  remark about the permission notice.
- `enzyme/Poseidon/runtime/tcec/tcec_rt.cu`: calls cuMpSGEMM (above) when
  built with `-DPOSEIDON_TCEC_USE_CUMPSGEMM=ON`.

## What the container image adds (not part of the archive)

`docker/Dockerfile` builds an image that contains, besides the above:

- the base image `nvidia/cuda:12.9.1-devel-ubuntu24.04`: the CUDA Toolkit
  under the NVIDIA CUDA End User License Agreement, the image under the NVIDIA
  Deep Learning Container License (`/NGC-DL-CONTAINER-LICENSE` in the image),
  and Ubuntu 24.04 packages under their own licences (among the ones installed
  here: MPFR and GMP, LGPL 3; CMake, BSD 3-Clause; Python, PSF);
- LLVM built from the source tarball of the commit above (Apache License 2.0
  with LLVM Exceptions);
- numpy 2.4.2 (BSD 3-Clause) and matplotlib 3.10.9 (Matplotlib licence, a
  PSF-style licence) from PyPI.

`docker/Dockerfile.herbie` (optional) adds Racket 8.15 (MIT or Apache 2.0),
the Rust 1.85.1 toolchain (MIT or Apache 2.0), Herbie's Rust dependencies and
the Racket packages `fpbench`, `rival` and `rival3`, each under its own
licence.
