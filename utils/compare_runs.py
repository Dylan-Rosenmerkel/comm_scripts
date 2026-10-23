#!/usr/bin/env python3
"""Compare two run_WADERs output folders: are the results identical?

Use it to check that a code change did not change the results: run run_WADERs on the same images with the old and the
new code (into different -o folders) and compare. The pipeline is deterministic, so identical inputs and code give
byte-identical results; any difference reported here comes from the change.

    python utils/compare_runs.py OLD NEW        (each: a WADERs_analysis_<run> folder, or the -o folder containing it)

Compared:
  summary.txt          line by line, ignoring the first line (it holds the run time) and the folder paths
  fit_quality.csv      byte for byte (gain, noise, dark current per CCD with errors)
  json/*/final/*.json  the panaSKImg configurations that were run, byte for byte
  gain_vs_charge/*.json   the measured g(n) curves (the 'extensions' part)
  root*/panaSKImg_clustersRec_*.root   cluster energies (Energy) and pixel energies (pixels_E), exactly
  composite*/.../*.csv the MakeComposite result tables
Not compared: PDFs and PNGs (they embed creation dates) and logs (they hold timings).
Exit code 0 if everything is identical, 1 otherwise. Needs PyROOT and numpy.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def find_run(path):
    """The WADERs_analysis_<run> folder: path itself, or the single one inside it."""
    if os.path.basename(os.path.normpath(path)).startswith("WADERs_analysis_"):
        return os.path.normpath(path)
    found = glob.glob(os.path.join(path, "WADERs_analysis_*"))
    if len(found) != 1:
        sys.exit(f"compare_runs: expected one WADERs_analysis_* folder in {path}, found {len(found)}")
    return found[0]


def summary_lines(path, root):
    """summary.txt without its first line (run time) and with the run folder replaced by <OUT>."""
    text = open(path).read().replace(root, "<OUT>")
    return [l for l in text.splitlines() if not l.startswith("run_WADERs summary:")]


def cluster_arrays(path):
    """(cluster energies, pixel energies) of one cluster file, all entries concatenated."""
    import ROOT
    f = ROOT.TFile.Open(path)
    t = f.Get("clustersRec")
    E, P = [], []
    for i in range(t.GetEntries()):
        t.GetEntry(i)
        E.append(np.array(list(t.Energy)))
        P.append(np.concatenate([np.array(list(v)) for v in t.pixels_E]) if len(t.pixels_E) else np.zeros(0))
    f.Close()
    return (np.concatenate(E) if E else np.zeros(0)), (np.concatenate(P) if P else np.zeros(0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old"); ap.add_argument("new")
    ap.add_argument("-q", "--quiet", action="store_true", help="print only differences and the result")
    args = ap.parse_args()
    A, B = find_run(args.old), find_run(args.new)
    bad = 0

    def report(ok, what):
        nonlocal bad
        bad += not ok
        if not ok or not args.quiet:
            print(("  ok    " if ok else "  DIFF  ") + what)

    report(summary_lines(f"{A}/summary.txt", A) == summary_lines(f"{B}/summary.txt", B), "summary.txt")
    for rel in ["fit_quality.csv"] + sorted(os.path.relpath(p, A) for p in glob.glob(f"{A}/json/*/final/*.json")):
        pa, pb = f"{A}/{rel}", f"{B}/{rel}"
        report(os.path.exists(pa) and os.path.exists(pb) and open(pa).read() == open(pb).read(), rel)
    for pa in sorted(glob.glob(f"{A}/gain_vs_charge/*.json")):
        rel = os.path.relpath(pa, A); pb = f"{B}/{rel}"
        same = os.path.exists(pb) and json.dumps(json.load(open(pa)).get("extensions"), sort_keys=True) == \
            json.dumps(json.load(open(pb)).get("extensions"), sort_keys=True)
        report(same, rel)
    for pa in sorted(glob.glob(f"{A}/root*/panaSKImg_clustersRec_*.root")):
        rel = os.path.relpath(pa, A); pb = f"{B}/{rel}"
        if not os.path.exists(pb):
            report(False, rel + " (missing in NEW)"); continue
        (ea, xa), (eb, xb) = cluster_arrays(pa), cluster_arrays(pb)
        report(ea.shape == eb.shape and np.array_equal(ea, eb) and np.array_equal(xa, xb), f"{rel} ({len(ea)} clusters)")
    for pa in sorted(glob.glob(f"{A}/composite*/Fe_55_Analysis_Plots/*/*/*.csv")):
        rel = os.path.relpath(pa, A); pb = f"{B}/{rel}"
        report(os.path.exists(pb) and open(pa).read() == open(pb).read(), rel)
    print("RESULT:", "IDENTICAL" if not bad else f"{bad} difference(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
