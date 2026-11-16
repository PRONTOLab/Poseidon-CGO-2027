#!/usr/bin/env python3
"""Reader of expected/picks*.tsv."""
import csv, os

ART = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_picks(device="rtx5090"):
    name = "picks.tsv" if device == "rtx5090" else "picks_%s.tsv" % device
    with open(os.path.join(ART, "expected", name), newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))
