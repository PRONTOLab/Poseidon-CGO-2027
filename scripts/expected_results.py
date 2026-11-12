#!/usr/bin/env python3
"""Generate the tables of EXPECTED_RESULTS.md from expected/picks*.tsv and
expected/claims.tsv, the one place the artifact's expected numbers and
decisions are written.

    python3 scripts/expected_results.py           # rewrite the generated tables in place
    python3 scripts/expected_results.py --check   # exit 1 if EXPECTED_RESULTS.md is stale

Only the text between `<!-- BEGIN GENERATED <name> -->` and
`<!-- END GENERATED <name> -->` is rewritten; the prose around it is not.

expected/picks.tsv   one row per decision: id, bench, select, what (in words), pin
                     (see scripts/expected.sh); picks_b300.tsv and picks_gh200.tsv
                     the same for the saved B300 and GH200 cost models
expected/claims.tsv  one row per checked quantity:
  id         claim id (a picks.tsv id when the claim is about a decision row)
  section    required | b300 | gh200
  bench      benchmark directory
  claim      what the paper says, in a few words
  run        the command that produces the number (from the artifact root)
  metric     the quantity
  paper      the paper's value
  expected   the value to expect with this artifact, or "pending" (to be
             re-measured on the RTX 5090 with this compiler; `run` produces it)
  tolerance  how close a reproduction must come, and why
  source     the artifact file holding the saved measurement behind `expected`
"""
import csv, os, re, sys

ART = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOC = os.path.join(ART, "EXPECTED_RESULTS.md")


def _rows(name):
    with open(os.path.join(ART, "expected", name), newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_picks(device="rtx5090"):
    return _rows("picks.tsv" if device == "rtx5090" else "picks_%s.tsv" % device)


def load_claims():
    return _rows("claims.tsv")


SITE = re.compile(r"^_Z\d+([a-z]+)_opt[^:]*:")


def short_item(item):
    """One pin item as a reader names it: the R-ChFSI site, the unit, the name."""
    item = item.strip()
    site = SITE.match(item)
    if site:
        item = site.group(1) + ": " + item[site.end():]
    else:
        item = item.split(":", 1)[1] if ":" in item and item.startswith("_Z") else item
    unit, _, name = item.partition("=")
    if unit.startswith("R") or unit.split(": ")[-1].startswith("R"):
        # Herbie rewrites are long expressions; picks.tsv has them in full.
        name = name if len(name) <= 48 else name[:45] + "..."
    return f"`{unit}={name}`" if name else f"`{unit}`"


def short_pin(pin):
    if pin in ("", "?"):
        return "as written" if pin == "" else "?"
    return "<br>".join(short_item(i) for i in pin.split(",") if i.strip())


def md_escape(text):
    return text.replace("|", "\\|")


def decisions_table(picks, model):
    out = ["| Row | Selected by | Configuration (%s) | Compiler's names (`[poseidon] pin:`) |" % model,
           "|---|---|---|---|"]
    for p in picks:
        sel = p["select"]
        if sel.startswith("share="):
            sel = "`-poseidon-cost-budget=%s`" % sel[len("share="):]
        elif sel.startswith("tau="):
            sel = "`-poseidon-tau=%s`" % sel[len("tau="):]
        else:
            sel = {"head": "cheapest frontier point", "zero": "budget 0"}.get(sel, sel)
        out.append(f"| `{p['id']}` | {sel} | {md_escape(p['what'])} | {md_escape(short_pin(p['pin']))} |")
    return out


def claims_table(claims, section):
    out = ["| Claim | Metric | Paper | Expected | Tolerance | Source | Run |",
           "|---|---|---|---|---|---|---|"]
    last = None
    for c in claims:
        if c["section"] != section:
            continue
        first = c["id"] != last
        last = c["id"]
        exp = "**pending**" if c["expected"] == "pending" else c["expected"]
        src = f"`{c['source']}`" if c["source"] and not c["source"].startswith("paper") else c["source"]
        out.append("| " + " | ".join(md_escape(x) for x in [
            (f"`{c['id']}`: {c['claim']}" if first else ""), c["metric"], c["paper"], exp,
            c["tolerance"], src, (f"`{c['run']}`" if first and c["run"] else "")]) + " |")
    return out


def render():
    claims = load_claims()
    return {
        "decisions": decisions_table(load_picks(), "shipped RTX 5090 model"),
        "decisions_b300": decisions_table(load_picks("b300"), "saved B300 model"),
        "decisions_gh200": decisions_table(load_picks("gh200"), "saved GH200 model"),
        "required": claims_table(claims, "required"),
        "b300": claims_table(claims, "b300"),
        "gh200": claims_table(claims, "gh200"),
    }


def splice(text, tables):
    for name, lines in tables.items():
        pat = re.compile(r"(<!-- BEGIN GENERATED %s -->\n).*?(<!-- END GENERATED %s -->)" % (name, name), re.S)
        if not pat.search(text):
            sys.exit(f"{DOC}: no generated region '{name}'")
        text = pat.sub(lambda m: m.group(1) + "\n".join(lines) + "\n" + m.group(2), text)
    return text


def main():
    old = open(DOC).read()
    new = splice(old, render())
    if "--check" in sys.argv:
        if new != old:
            sys.exit("EXPECTED_RESULTS.md is stale: run python3 scripts/expected_results.py")
        return
    if new != old:
        open(DOC, "w").write(new)
        print("rewrote the generated tables of EXPECTED_RESULTS.md")


if __name__ == "__main__":
    main()
