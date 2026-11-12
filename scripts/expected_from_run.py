#!/usr/bin/env python3
"""Fill the `expected` cells of expected/claims.tsv from one run of the
artifact, so the numbers of EXPECTED_RESULTS.md are a saved measurement.

    python3 scripts/expected_from_run.py <results/<GPU>_<arch>_<stamp>> <tag> [--rows id,id,...] [--dry]
    python3 scripts/expected_results.py

Each benchmark's part of the bundle is copied to benchmarks/<bench>/golden/<tag>/
(the cited source of every cell this fills, and the figure scripts' fallback); a cell is read from the row of a frontier
table whose configuration (`pin`) is the claim's row of expected/picks.tsv, so
a moved decision leaves its cell untouched and is reported. Without --rows only
`pending` cells are filled; `--rows all` refills every row the run measured.
"""
import csv, glob, math, os, re, shutil, sys

ART = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from expected_results import load_picks, load_claims  # noqa: E402


def tsv(path):
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f, delimiter="\t")]


def pinset(pin):
    return frozenset(i for i in (pin or "").split(",") if i)


def g(x, fmt="%.6e"):
    try:
        return fmt % float(x)
    except (TypeError, ValueError):
        return str(x)


def main():
    run, tag = sys.argv[1], sys.argv[2]
    only = None
    if "--rows" in sys.argv:
        only = set(sys.argv[sys.argv.index("--rows") + 1].split(","))
        if only == {"all"}:
            only = {c["id"] for c in load_claims()}
    data = os.path.join(run, "data")
    picks = {p["id"]: p for p in load_picks()}
    if "--dry" not in sys.argv:
        # The whole bundle of each benchmark becomes its golden/<tag>/: the cited
        # source of the cells, and what the figure scripts plot without a run.
        for b in sorted(os.listdir(data)):
            dst = os.path.join(ART, "benchmarks", b, "golden", tag)
            if os.path.isdir(os.path.join(ART, "benchmarks", b)):
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(os.path.join(data, b), dst)

    def table(bench, name):
        p = os.path.join(data, bench, name)
        if not os.path.exists(p):
            return None, None
        return tsv(p), "benchmarks/%s/golden/%s/%s" % (bench, tag, name)

    def by_pin(rows, row_id):
        want = pinset(picks[row_id]["pin"])
        for i, r in enumerate(rows or []):
            if pinset(r.get("pin")) == want:
                return i + 2, r
        return None, None

    out = {}  # (id, metric index within the id) -> (expected, source)
    # fig2_drift: the run's printed timings and its drift file
    fl = os.path.join(run, "logs", "fig2_drift.log")
    if os.path.exists(fl):
        txt = open(fl, errors="replace").read()
        gf = re.search(r"GPU FP64 \(as written\)\s+([0-9.]+) ms.*?GPU 3xFP32 emulation\s+([0-9.]+) ms\s+([0-9.]+)x", txt, re.S)
        cf = re.search(r"CPU FP64 \(as written\)\s+([0-9.]+) ms.*?CPU 3xFP32 emulation\s+([0-9.]+) ms", txt, re.S)
        dst = "benchmarks/fig2_drift/golden/%s/fig2_drift.log" % tag
        if "--dry" not in sys.argv:
            os.makedirs(os.path.dirname(os.path.join(ART, dst)), exist_ok=True)
            shutil.copy(fl, os.path.join(ART, dst))
        if gf:
            out[("fig2-gpu", 0)] = ("%.2fx faster (%s -> %s ms)" % (float(gf.group(1)) / float(gf.group(2)), gf.group(1), gf.group(2)), dst)
        if cf:
            out[("fig2-gpu", 1)] = ("%.1fx slower (%s -> %s ms)" % (float(cf.group(2)) / float(cf.group(1)), cf.group(1), cf.group(2)), dst)
        dc = os.path.join(data, "fig2_drift", "drift.csv")
        if os.path.exists(dc):
            last = open(dc).read().split()[-1].split(",")
            out[("fig2-gpu", 2)] = (last[-1], "benchmarks/fig2_drift/golden/%s/drift.csv" % tag)
    # GENGA
    ff, ffs = table("genga_nbody", "force_full_g0.tsv")
    for rid in ("genga-force-fp32", "genga-force-expansion"):
        ln, r = by_pin(ff, rid)
        if r:
            out[(rid, 0)] = ("%.3fx" % float(r["speedup"]), "%s:%d" % (ffs, ln))
            out[(rid, 1)] = (g(r["L2"]), "%s:%d" % (ffs, ln))
    mf, mfs = table("genga_nbody", "force_mfoff.tsv")
    if mf:
        cand = [(float(r["L2"]), i + 2, r) for i, r in enumerate(mf)
                if r["L2"] not in ("", "nan") and float(r["L2"]) > 0]
        if cand:
            l2, ln, r = min(cand)
            out[("genga-force-expoff", 0)] = ("L2 %s at %.3fx" % (g(l2), float(r["speedup"])), "%s:%d" % (mfs, ln))
    kh, khs = table("genga_nbody", "kepler_herbieoff_g0.tsv")
    kf, kfs = table("genga_nbody", "kepler_full_g0.tsv")
    ln, r = by_pin(kh, "genga-drift-naive")
    if r:
        out[("genga-drift-naive", 0)] = (g(r["Edrift"]), "%s:%d" % (khs, ln))
        out[("genga-drift-naive", 1)] = ("%.3fx" % float(r["speedup"]), "%s:%d" % (khs, ln))
    ln, r = by_pin(kf, "genga-drift-algebraic")
    alg = r
    if r:
        out[("genga-drift-algebraic", 0)] = (g(r["Edrift"]), "%s:%d" % (kfs, ln))
        out[("genga-drift-algebraic", 1)] = ("%.3fx" % float(r["speedup"]), "%s:%d" % (kfs, ln))
    nai = by_pin(kh, "genga-drift-naive")[1]
    if alg and nai:
        out[("genga-drift-ablation", 0)] = (
            "%.0fx (%s at %.3fx vs %s at %.3fx)" % (float(nai["Edrift"]) / float(alg["Edrift"]), g(nai["Edrift"]),
                                                   float(nai["speedup"]), g(alg["Edrift"]), float(alg["speedup"])),
            "%s, %s" % (khs, kfs))
    ts, tss = table("genga_nbody", "kepler_timeseries.tsv")
    if ts:
        last = ts[-1]
        out[("genga-drift-timeseries", 0)] = (g(last["naive_fp32"]), "%s:%d" % (tss, len(ts) + 1))
        out[("genga-drift-timeseries", 1)] = (
            "%s (%.0fx below naive)" % (g(last["poseidon"]), float(last["naive_fp32"]) / float(last["poseidon"])),
            "%s:%d" % (tss, len(ts) + 1))
    un, uns = table("genga_nbody", "unified.tsv")
    ln, r = by_pin(un, "genga-unified")
    if r:
        out[("genga-unified", 0)] = (g(r["L2"]), "%s:%d" % (uns, ln))
        out[("genga-unified", 1)] = ("%.3fx" % float(r["speedup"]), "%s:%d" % (uns, ln))
    # ozp
    ap, aps = table("ozp", "autopick.tsv")
    for i, r in enumerate(ap or []):
        rid = "ozp-tau" + r["tau"]
        if rid in picks and r["pick"] == picks[rid]["pin"].split(":M0=")[-1]:
            s = "%s:%d" % (aps, i + 2)
            out[(rid, 0)] = (g(r["deployed_RMSD"]), s)
            out[(rid, 1)] = ("%.3fx" % float(r["x_scalar_fp64"]), s)
            if rid == "ozp-tau1e-7":
                out[(rid, 2)] = ("%.3fx" % float(r["x_cublas_dgemm"]), s)
    dr, drs = table("ozp", "dispatch_rungs.tsv")
    for rid, key in (("ozp-direct-bf16", "direct-dispatch_cublas_bf16/f32"),
                     ("ozp-direct-tf32", "direct-dispatch_cublas_tf32/f32")):
        for i, r in enumerate(dr or []):
            if r.get("pick") == key:
                out[(rid, 0)] = ("%.3fx / %s" % (float(r["speedup"]), g(r["RMSD"])), "%s:%d" % (drs, i + 2))
    lib, libs = table("ozp", "direct_tc_lib.tsv")
    for i, r in enumerate(lib or []):
        if r.get("tag") == "LIB_f16":
            out[("ozp-direct-f16", 0)] = ("%.3fx / %s" % (float(r["speedup"]), g(r["RMSD"])), "%s:%d" % (libs, i + 2))
    pa, pas = table("ozp", "pareto.tsv")
    if pa:
        d = {r["tag"]: (i + 2, r) for i, r in enumerate(pa)}
        if all(k in d for k in ("M0_1", "M0_2", "M0_3")):
            out[("ozp-inkernel-ladder", 0)] = (", ".join("%.3fx / %s" % (float(d[k][1]["speedup"]), g(d[k][1]["RMSD"]))
                                                        for k in ("M0_1", "M0_2", "M0_3")),
                                              "%s:%d-%d" % (pas, d["M0_1"][0], d["M0_3"][0]))
        if "M0_4" in d:
            out[("ozp-inkernel-ladder", 1)] = ("%.3fx / %s" % (float(d["M0_4"][1]["speedup"]), g(d["M0_4"][1]["RMSD"])),
                                              "%s:%d" % (pas, d["M0_4"][0]))
    su, sus = table("ozp", "surrogate.tsv")
    if su:
        nm = {(r["N"], r["method"]): (i + 2, r) for i, r in enumerate(su)}
        if ("1024", "nm9") in nm and ("8192", "nm9") in nm:
            out[("ozp-profile-transfer", 0)] = ("%.2fx / %.2fx" % (float(nm[("1024", "nm9")][1]["speedup_naive"]),
                                                                float(nm[("8192", "nm9")][1]["speedup_naive"])), sus)
            out[("ozp-profile-transfer", 1)] = (g(nm[("8192", "nm9")][1]["rmsd"]), "%s:%d" % (sus, nm[("8192", "nm9")][0]))
    # R-ChFSI
    on, ons = table("chase_eigensolver", "scvh_probe_on.tsv")
    mo, mos = table("chase_eigensolver", "chase_money_data.tsv")
    ab, abs_ = table("chase_eigensolver", "ablation_knee.tsv")
    fp64 = None
    for r in mo or []:
        if r["budget"] == "0":
            fp64 = float(r["solve_ms"])
    for rid in ("rchfsi-tf32tcec", "rchfsi-bf16tcec", "rchfsi-bf16tcec-nm14", "rchfsi-knee"):
        share = float(picks[rid]["select"].split("=", 1)[1])
        for i, r in enumerate(on or []):
            if float(r["budget"]) == share and fp64:
                out[(rid, 0)] = ("%.3fx / %s / %s" % (fp64 / float(r["solve_ms_avg5"]), g(r["eigL2"]),
                                                       r["locked"].replace("locked ", "").replace("/", " of ")),
                                 "%s:%d" % (ons, i + 2))
    for i, r in enumerate(ab or []):
        if r["arm"] == "sensitivity-blind" and fp64:
            out[("rchfsi-blind", 0)] = ("%s / %.3fx" % (r["locked"].replace("locked ", "").replace("/", " of "),
                                                        fp64 / float(r["solve_ms_avg5"])), "%s:%d" % (abs_, i + 2))

    if "--dry" in sys.argv:   # print every value read, write nothing
        for (cid, k), (v, srcf) in sorted(out.items()):
            print("%s[%d]\t%s\t%s" % (cid, k, v, srcf))
        return
    claims = load_claims()
    seen, filled = {}, []
    for c in claims:
        k = seen.get(c["id"], 0)
        seen[c["id"]] = k + 1
        if (c["id"], k) not in out:
            continue
        if only is not None and c["id"] not in only:
            continue
        if only is None and c["expected"] != "pending":
            continue
        c["expected"], c["source"] = out[(c["id"], k)]
        filled.append("%s[%d] = %s" % (c["id"], k, c["expected"]))
    with open(os.path.join(ART, "expected", "claims.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(list(claims[0].keys()))
        for c in claims:
            w.writerow(list(c.values()))
    print("\n".join(filled))
    missing = [c["id"] for c in claims if c["section"] == "required" and c["expected"] == "pending"]
    if missing:
        print("still pending:", " ".join(sorted(set(missing))))


if __name__ == "__main__":
    main()
