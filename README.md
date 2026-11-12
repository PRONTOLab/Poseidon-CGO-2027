# Poseidon: CGO 2027 artifact

Artifact of the CGO 2027 paper *Floating up the Peak: Automatic Numerical
Rewriting Enables Fast and Tensorized Accelerator Code*. Poseidon is a
profile-guided compiler pass (an LLVM plugin, part of the Enzyme repository)
that rewrites GPU kernels under a user accuracy target: algebraic rewrites,
precision changes, floating-point expansions, tensor-core raising and
tensor-core emulation.

## What is where

| Path | Content |
|---|---|
| `cgo2027/` | The artifact: the four benchmarks, the shipped RTX 5090 cost model, profiles, Herbie answers, scripts, `run_artifact.sh`. Start with `cgo2027/README.md`; what a reviewer should see is `cgo2027/EXPECTED_RESULTS.md`. |
| `cgo2027/docker/Dockerfile` | One image with every dependency, built from this directory (below). |
| `modal-b300/` | Optional: measures the paper's B300 rows on one NVIDIA B300 at Modal (`modal-b300/README.md`). |
| `cgo2027/LICENSES.md` | Licences and provenance of everything in this repository. |
| `sources/Enzyme` | Git submodule, the compiler: the Enzyme repository (https://github.com/EnzymeAD/Enzyme) at commit `9e2e0572931248b4610940f2906e583914f9913e`; Poseidon is `enzyme/Poseidon`. Apache License 2.0 with LLVM Exceptions. |
| `sources/cuMpSGEMM` | Git submodule: cuMpSGEMM (https://github.com/enp1s0/cuMpSGEMM) at commit `42dd92ea1c0e3d603bbb3f8f354f53e0967a9f4b` with its three submodules at the commits it records: the error-corrected tensor-core GEMM the compiler's TCEC runtime calls. |

Clone with the submodules:

```sh
git clone --recursive https://github.com/PRONTOLab/Poseidon-CGO-2027.git
# in an existing clone: git submodule update --init --recursive
```

Herbie (https://github.com/herbie-fp/herbie, upstream commit 73ba1fe76 plus the
`--num-taylor-terms` option) is needed only to search GENGA's rewrites again;
the artifact replays the shipped Herbie answers by default. Its source is in
the archival deposit (`sources/herbie-50ffcda04.tar.gz`), not in this
repository.

LLVM is not included. The compiler is built against llvm-project
(https://github.com/llvm/llvm-project) at commit `9d618fc0243103263ebf9bdcd08ef25e871d9c78`; the
Dockerfile downloads that commit's source tarball from GitHub and checks its
sha256.

## Setup

**A. Container (recommended).** Needs Docker with the NVIDIA Container
Toolkit. From this directory:

```sh
docker build -f cgo2027/docker/Dockerfile -t poseidon-cgo2027 .
docker run --rm -it --gpus all poseidon-cgo2027      # or --device nvidia.com/gpu=0 (CDI)
./run_artifact.sh --check                            # inside the container
```

**B. From source.** `cgo2027/README.md`, section Setup, gives the exact
commands for LLVM, cuMpSGEMM and the compiler from the submodules above.

Then follow `cgo2027/README.md`: `./run_artifact.sh --picks` (every solver
decision, compile-only, no GPU), `./run_artifact.sh --quick` (smoke tests on
the GPU), `./run_artifact.sh` (the full evaluation).

Hardware: an NVIDIA RTX 5090 or another GPU of compute capability 12.0 for the
results of `EXPECTED_RESULTS.md`; other NVIDIA GPUs run the same pipeline on a
cost model measured on them.
