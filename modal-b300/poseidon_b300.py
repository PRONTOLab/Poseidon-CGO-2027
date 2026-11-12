"""Poseidon CGO 2027 artifact on a Modal B300: the Slurm-job counterpart.

    modal run --detach poseidon_b300.py --stage build      # CPU only: LLVM, Enzyme+Poseidon
    modal run poseidon_b300.py --stage smoke               # B300, minutes: toolchain checks
    modal run poseidon_b300.py --stage calibrate           # B300: fresh cost model vs the saved one
    modal run --detach poseidon_b300.py --stage run        # B300: replay the decided rows, measure
    modal run --detach poseidon_b300.py --stage all        # the four above in one call
    modal run --detach poseidon_b300.py --stage full       # not default: calibrate/profile/solve on the device

The run stage replays the replay pack (solve_b300_locally.sh decided every B300
row on the sender's machine with the saved B300 cost model): each row is
compiled by candidate name, with no Herbie search and no DP solve, and measured.
Options: --run-id <id>, --rows "<row> ..." (run), --jobs N, --force "<parts>",
--with-herbie (build, for --stage full), --benches/--mode/--genga-herbie (full).
See README.md.

Every call carries the bundle made by make_bundle.sh (compiler source, Herbie
source, the artifact, the replay pack, MANIFEST); the app refuses to start when
a tarball is missing or differs from MANIFEST.
"""

import hashlib
import os
import pathlib
import subprocess
import sys
import time

import modal

HERE = pathlib.Path(__file__).resolve().parent
LOCAL_BUNDLE = HERE / "bundle"
LOCAL_SCRIPTS = HERE / "remote"

# ---- what runs where ----------------------------------------------------------
GPU = "B300"
# B300 needs CUDA 13.1 or newer in the container; the host driver offers
# CUDA 13.3, so the toolkit is the oldest 13.1 release line image.
CUDA_IMAGE = "nvidia/cuda:13.1.2-devel-ubuntu24.04"
# CMAKE_CUDA_ARCHITECTURES of the Poseidon runtimes and calibrators: the B300
# is compute capability 10.3. The smoke stage checks the device against it.
CUDA_ARCHS = "103-real"
PYTHON_VERSION = "3.12"
PIP_PACKAGES = ["numpy==2.4.2", "matplotlib==3.10.9"]
APT_PACKAGES = [
    "ca-certificates", "curl", "git", "xz-utils", "file", "bc", "time", "procps",
    "cmake", "ninja-build", "clang", "lld",
    "libmpfr-dev", "libgmp-dev", "zlib1g-dev", "libzstd-dev", "libxml2-dev", "liblzma-dev",
]
RACKET_VERSION = "8.15"
RUST_VERSION = "1.85.1"
RACKET_INSTALL = (
    f"curl -fsSL --retry 5 -o /tmp/racket.sh https://download.racket-lang.org/releases/"
    f"{RACKET_VERSION}/installers/racket-{RACKET_VERSION}-x86_64-linux-cs.sh"
    " && sh /tmp/racket.sh --in-place --dest /opt/racket && rm -f /tmp/racket.sh"
    " && /opt/racket/bin/racket --version"
)
RUST_ENV = {"RUSTUP_HOME": "/opt/rust/rustup", "CARGO_HOME": "/opt/rust/cargo"}
RUST_INSTALL = (
    "export RUSTUP_HOME=/opt/rust/rustup CARGO_HOME=/opt/rust/cargo"
    " && curl -fsSL --retry 5 -o /tmp/rustup-init"
    " https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init"
    " && chmod +x /tmp/rustup-init"
    f" && /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain {RUST_VERSION}"
    " && rm -f /tmp/rustup-init && /opt/rust/cargo/bin/cargo --version"
)

TOOLCHAIN_VOLUME = "poseidon-b300-toolchain"
RESULTS_VOLUME = "poseidon-b300-results"
TC = "/toolchain"       # toolchain Volume: llvm/, herbie/, poseidon-src/, poseidon-build/, logs/
RES = "/results"        # results Volume: <run id>/...
BUNDLE = "/bundle"      # the bundle, mounted read-only into every container
SCRIPTS = "/opt/poseidon-b300"

BUILD_CPUS = 32
BUILD_TIMEOUT = 8 * 3600
# GPU stages: limits a few times the expected length (smoke and calibrate
# minutes, the replay well under an hour), so a hung step does not hold a B300.
SMOKE_TIMEOUT = 1800
RUN_TIMEOUT = 24 * 3600     # Modal's maximum
REPLAY_TIMEOUT = 2 * 3600
BUNDLE_FILES = ("poseidon-src.tar.gz", "cumpsgemm-src.tar.gz", "artifact.tar.gz", "replay_pack.tar.gz")
# Bundled only by make_bundle.sh --herbie-repo; needed by --with-herbie.
OPTIONAL_BUNDLE_FILES = ("herbie-src.tar.gz",)
BENCHES = ("fig2_drift", "genga_nbody", "ozp", "chase_eigensolver")


def read_manifest(directory: pathlib.Path) -> dict:
    path = directory / "MANIFEST"
    if not path.is_file():
        raise SystemExit(f"bundle: {path} is missing; run make_bundle.sh first")
    entries, meta = {}, {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if "=" in line and "  " not in line:
            key, value = line.split("=", 1)
            meta[key] = value
        else:
            digest, size, name = line.split()
            entries[name] = (digest, int(size))
    return {"files": entries, **meta}


def verify_bundle(directory: pathlib.Path) -> dict:
    """Refuse to go on when a tarball is missing or differs from MANIFEST."""
    manifest = read_manifest(directory)
    for name in BUNDLE_FILES + OPTIONAL_BUNDLE_FILES:
        if name not in manifest["files"]:
            if name in OPTIONAL_BUNDLE_FILES:
                continue
            raise SystemExit(f"bundle: MANIFEST does not list {name}; rerun make_bundle.sh")
        path = directory / name
        if not path.is_file():
            raise SystemExit(f"bundle: {path} is missing; rerun make_bundle.sh")
        digest, size = manifest["files"][name]
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                sha.update(chunk)
        if path.stat().st_size != size or sha.hexdigest() != digest:
            raise SystemExit(f"bundle: {name} differs from MANIFEST; rerun make_bundle.sh")
    return manifest


if modal.is_local():
    MANIFEST = verify_bundle(LOCAL_BUNDLE)

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python=PYTHON_VERSION)
    .env({"DEBIAN_FRONTEND": "noninteractive"})
    .apt_install(*APT_PACKAGES)
    .pip_install(*PIP_PACKAGES)
    .run_commands(RACKET_INSTALL, RUST_INSTALL)
    .env(RUST_ENV)
    .add_local_dir(LOCAL_SCRIPTS, SCRIPTS)
    .add_local_file(LOCAL_BUNDLE / "MANIFEST", f"{BUNDLE}/MANIFEST")
)
for _name in BUNDLE_FILES:
    image = image.add_local_file(LOCAL_BUNDLE / _name, f"{BUNDLE}/{_name}")
for _name in OPTIONAL_BUNDLE_FILES:
    if (_name in MANIFEST["files"]) if modal.is_local() else pathlib.Path(BUNDLE, _name).is_file():
        image = image.add_local_file(LOCAL_BUNDLE / _name, f"{BUNDLE}/{_name}")

app = modal.App("poseidon-b300")
toolchain_volume = modal.Volume.from_name(TOOLCHAIN_VOLUME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)


def run_logged(cmd: list, log_path: str, env: dict) -> int:
    """Run cmd, streaming its output to this container's log and to log_path."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    full_env = {**os.environ, **{k: str(v) for k, v in env.items()}}
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=full_env)
        for line in iter(proc.stdout.readline, b""):
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return proc.wait()


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


@app.function(
    image=image,
    cpu=float(BUILD_CPUS),
    memory=65536,
    timeout=BUILD_TIMEOUT,
    volumes={TC: toolchain_volume},
)
def build(jobs: int = 0, force: str = "", with_herbie: bool = False) -> str:
    verify_bundle(pathlib.Path(BUNDLE))
    log = f"{TC}/logs/build-{stamp()}.log"
    # os.cpu_count() and nproc can report the host's cores, not the request.
    env = {"TC": TC, "BUNDLE": BUNDLE, "CUDA_ARCHS": CUDA_ARCHS, "FORCE": force,
           "JOBS": jobs or BUILD_CPUS, "WITH_HERBIE": int(with_herbie)}
    try:
        rc = run_logged(["bash", f"{SCRIPTS}/build.sh", "all"], log, env)
    finally:
        toolchain_volume.commit()
    if rc != 0:
        raise RuntimeError(f"build failed (exit {rc}); log in Volume {TOOLCHAIN_VOLUME}: {log[len(TC):]}")
    stamps = []
    for part in ("llvm", "herbie", "poseidon"):
        p = pathlib.Path(TC) / f"{part}.stamp"
        stamps.append(f"{part}: {p.read_text().strip() if p.exists() else 'not built'}")
    return "\n".join(stamps)


GPU_FUNCTION = dict(image=image, gpu=GPU, volumes={TC: toolchain_volume, RES: results_volume})


def gpu_stage(script: str, out: str, log: str, env: dict) -> int:
    verify_bundle(pathlib.Path(BUNDLE))
    try:
        return run_logged(["bash", f"{SCRIPTS}/{script}", out], log,
                          {"TC": TC, "BUNDLE": BUNDLE, "WORK": "/work", **env})
    finally:
        results_volume.commit()


@app.function(cpu=8.0, memory=32768, timeout=SMOKE_TIMEOUT, **GPU_FUNCTION)
def smoke(run_id: str) -> tuple:
    out = f"{RES}/{run_id}/smoke"
    rc = gpu_stage("smoke.sh", out, f"{out}/smoke.log", {})
    return rc, f"smoke: {'ALL OK' if rc == 0 else f'{rc} step(s) failed'} (log: /{run_id}/smoke/smoke.log)"


@app.function(cpu=8.0, memory=32768, timeout=SMOKE_TIMEOUT, **GPU_FUNCTION)
def calibrate(run_id: str) -> tuple:
    """Informational: this B300's cost model against the saved one. Nothing
    uses the fresh model."""
    out = f"{RES}/{run_id}/calibration_check"
    rc = gpu_stage("calibrate_check.sh", out, f"{out}/calibrate_check.log", {})
    return rc, f"calibration check: exit {rc} (reports: /{run_id}/calibration_check/compare_*.txt)"


@app.function(cpu=16.0, memory=65536, timeout=REPLAY_TIMEOUT, **GPU_FUNCTION)
def replay(run_id: str, rows: str = "") -> tuple:
    """The default run: the configurations decided by solve_b300_locally.sh,
    compiled by name (no Herbie search, no DP solve) and measured."""
    out = f"{RES}/{run_id}/replay"
    cmd_rows = rows.split()
    verify_bundle(pathlib.Path(BUNDLE))
    try:
        rc = run_logged(["bash", f"{SCRIPTS}/replay.sh", out, *cmd_rows], f"{out}/replay.log",
                        {"TC": TC, "BUNDLE": BUNDLE, "WORK": "/work"})
    finally:
        results_volume.commit()
    table = pathlib.Path(out) / "replay.tsv"
    return rc, table.read_text() if table.exists() else f"replay.sh exit {rc}, no table"


@app.function(cpu=16.0, memory=65536, timeout=RUN_TIMEOUT, **GPU_FUNCTION)
def full_run(run_id: str, benches: str = "", mode: str = "full", genga_herbie: str = "auto",
             deadline: float = 0.0) -> tuple:
    """Not the default: the artifact's own on-device flow (calibrate, profile,
    solve, Herbie when built with --with-herbie)."""
    start = time.time()
    # A caller with its own limit (all_stages) passes the time it must be done by.
    end = min(start + RUN_TIMEOUT - 300, deadline or float("inf"))
    out = f"{RES}/{run_id}/full"
    rc = gpu_stage("run.sh", out, f"{out}/run-{stamp()}.log", {
        "MODE": mode, "BENCHES": benches or " ".join(BENCHES), "GENGA_HERBIE": genga_herbie,
        # Stop a benchmark early enough to save its partial state.
        "DEADLINE": int(end), "DEADLINE_MARGIN": 2700})
    summary = pathlib.Path(out) / "SUMMARY.txt"
    return rc, summary.read_text() if summary.exists() else f"run.sh exit {rc}, no summary"


@app.function(image=modal.Image.debian_slim(), timeout=RUN_TIMEOUT)
def all_stages(run_id: str, jobs: int = 0, force: str = "") -> str:
    """build, smoke, calibration check and the replay run, called from one
    server-side function so that `modal run --detach` keeps the chain alive."""
    report = [build.remote(jobs=jobs, force=force)]
    rc, text = smoke.remote(run_id)
    report.append(text)
    if rc != 0:
        report.append("replay: not started because the smoke stage failed")
        return "\n".join(report)
    report.append(calibrate.remote(run_id)[1])
    report.append(replay.remote(run_id)[1])
    return "\n".join(report)


STAGES = ("build", "smoke", "calibrate", "run", "all", "full")


@app.local_entrypoint()
def main(stage: str = "", run_id: str = "", rows: str = "", benches: str = "", mode: str = "full",
         genga_herbie: str = "auto", jobs: int = 0, force: str = "", with_herbie: bool = False):
    if stage not in STAGES:
        raise SystemExit(f"--stage must be one of {', '.join(STAGES)} (see README.md)")
    if mode not in ("full", "quick"):
        raise SystemExit("--mode must be full or quick")
    if genga_herbie not in ("auto", "replay", "run"):
        raise SystemExit("--genga-herbie must be auto, replay or run")
    unknown = set(benches.split()) - set(BENCHES)
    if unknown:
        raise SystemExit(f"--benches: unknown {' '.join(sorted(unknown))}; choose from {' '.join(BENCHES)}")
    run_id = run_id or f"b300-{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"bundle: compiler {MANIFEST.get('compiler_commit')}, artifact {MANIFEST.get('artifact_files')} "
          f"files, made {MANIFEST.get('created')}")
    get = f"modal volume get {RESULTS_VOLUME} {run_id} ."
    if with_herbie and "herbie-src.tar.gz" not in MANIFEST["files"]:
        raise SystemExit("--with-herbie needs a bundle made with make_bundle.sh --herbie-repo <Herbie clone>")
    if stage != "build":
        print(f"run id: {run_id}\nresults, when done: {get}")

    if stage == "build":
        print(build.remote(jobs=jobs, force=force, with_herbie=with_herbie))
        return
    if stage == "smoke":
        text = smoke.remote(run_id)[1]
    elif stage == "calibrate":
        text = calibrate.remote(run_id)[1]
    elif stage == "run":
        text = replay.remote(run_id, rows=rows)[1]
    elif stage == "full":
        text = full_run.remote(run_id, benches=benches, mode=mode, genga_herbie=genga_herbie)[1]
    else:
        text = all_stages.remote(run_id, jobs=jobs, force=force)
    print(text)
    print(f"results: {get}")
