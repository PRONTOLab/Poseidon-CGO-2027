"""Read the compiler's explanation files (-poseidon-report-path) for the B300
replay pack.

  pack_tools.py frontier <reports dir>
      every frontier point of the solve: share, modelled error, configuration
      (site:unit=name items, the spelling -poseidon-apply-rewrites takes);
      share None: a point no budget selects (match and matchclass skip it)
  pack_tools.py match <reports dir> <pin value>
      the share of the frontier point whose configuration is exactly <pin
      value> (comma-joined items); exit 1 when no point has it
  pack_tools.py matchclass <reports dir> <regex> [<regex> ...]
      of the frontier points whose configuration has, for every regex, an item
      matching it: the one with the smallest modelled error (share, error,
      configuration); exit 1 when no point qualifies
  pack_tools.py alt <reports dir> <site:unit=name>
      that candidate's priced row: cost_delta (share), modelled_error,
      modelled_domain_error, status, reason
  pack_tools.py result <reports dir>
      the applied configuration's cost before/after (shares) and modelled error
"""
import glob
import json
import os
import sys


def load(reports):
    files = sorted(glob.glob(os.path.join(reports, "*.explain.json")))
    if not files:
        sys.exit(f"pack_tools: no *.explain.json in {reports}")
    return [json.load(open(f)) for f in files]


def names(doc):
    site = doc["function"]["symbol"]
    out = {}
    for sg in doc.get("subgraphs", []):
        for alt in sg.get("alternatives", []):
            if "id" not in alt:          # a candidate pricing dropped has no id
                continue
            unit = alt["id"].rsplit("_", 1)[0]
            name = alt["summary"]
            # A precision candidate's summary is its name, ": ", then the
            # operations it moves; the pin carries the name alone.
            if unit.startswith("PT"):
                name = name.split(": ", 1)[0]
            # A Herbie candidate is named by its expression (its fpcore).
            elif unit.startswith("R") and "fpcore" in alt:
                name = alt["fpcore"]
            out[alt["id"]] = (f"{site}:{unit}={name}", alt)
    return out


def frontier(docs):
    # A point no budget selects (a penalized one) carries no "budget"; it is
    # listed with budget None, keyed by its cost, and never matched.
    points = {}
    for doc in docs:
        idmap = names(doc)
        for p in doc.get("pareto", []):
            key = p["budget"] if "budget" in p else ("unselectable", p.get("cost"))
            pt = points.setdefault(key, {"budget": p.get("budget"), "items": set(),
                                                  "err": p.get("modelled_rel_error"),
                                                  "cost": p.get("cost")})
            for i in p.get("applied", []):
                pt["items"].add(idmap[i][0] if i in idmap else f"{doc['function']['symbol']}:{i}")
    return sorted(points.values(), key=lambda q: (q["budget"] is None, q["budget"] if q["budget"] is not None else q["cost"] or 0))


def split(value):
    return {s.strip() for s in value.split(",") if s.strip()}


def main():
    cmd, reports = sys.argv[1], sys.argv[2]
    docs = load(reports)
    if cmd == "frontier":
        for p in frontier(docs):
            print(f"{p['budget']!r}\t{p['err']}\t{','.join(sorted(p['items']))}")
    elif cmd == "match":
        want = split(sys.argv[3])
        hits = [p for p in frontier(docs) if p["items"] == want and p["budget"] is not None]
        if not hits:
            sys.exit(1)
        print(repr(hits[0]["budget"]))
    elif cmd == "matchclass":
        import re
        pats = [re.compile(r) for r in sys.argv[3:]]
        hits = [p for p in frontier(docs) if p["budget"] is not None
                and all(any(rx.search(i) for i in p["items"]) for rx in pats)]
        if not hits:
            sys.exit(1)
        best = min(hits, key=lambda p: (p["err"] if p["err"] is not None else float("inf")))
        print(f"{best['budget']!r}\t{best['err']}\t{','.join(sorted(best['items']))}\t{len(hits)}")
    elif cmd == "alt":
        item = sys.argv[3]
        for doc in docs:
            for full, alt in names(doc).values():
                if full == item:
                    print("\t".join(str(alt.get(k, "")) for k in
                                    ("cost_delta", "modelled_error", "modelled_domain_error",
                                     "status", "reason")))
                    return
        print("not priced\t\t\t\t")
    elif cmd == "result":
        for doc in docs:
            r = doc.get("result", {})
            print("\t".join([doc["function"]["symbol"], str(r.get("cost_before", "")),
                             str(r.get("cost_after", "")), str(r.get("modelled_rel_error", ""))]))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
