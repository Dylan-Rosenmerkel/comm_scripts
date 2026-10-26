"""Paths of the Am-241 (runs_29img) analysis used by the crowded-CCD scripts.

AN: the runs_29img analysis folder (outside the repo: its scan.load / load2 readers, final_fit.csv, the WADERs
cluster tables). setup() puts it on sys.path but makes `amfit` (the constrained Np L fit) resolve to the repo's
tested copy, docs/am241_note/amfit.py (numerically identical to the copy in AN).
"""
import os
import sys

AN = os.path.expanduser("~/Documents/JH1_Testing/AMTEst/runs_29img/analysis")
AMFIT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "am241_note")


def setup():
    for p in (AN, AMFIT_DIR):          # AMFIT_DIR ends up first
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    return AN
