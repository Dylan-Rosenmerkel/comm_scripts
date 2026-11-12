#!/usr/bin/env python3
"""Copy the L0 / L1 readers and trace plotters of ccd_opt_ml into comm_scripts/src/raw_levels/ (run_WADERs --engine
native) and superWADERs/commissioning/raw_levels/ (run_superWADERs); --dest chooses one of them.

run_WADERs --engine native (src/raw_data.py) and run_superWADERs (superWADERs/commissioning/raw_data.py) decode
and plot level-0 traces and turn level-1 images into FITS images with these modules. They are maintained in
ccd_opt_ml (github Dylan-Rosenmerkel/ccd_opt_ml, src/); comm_scripts keeps copies so that it does not need that
repository. The copies are unchanged except for a first line naming the source
file and its commit; they import each other by their plain names (raw_data.py puts raw_levels/ on sys.path).

    python utils/sync_from_ccd_opt_ml.py [--source ~/Documents/Software/ccd_opt_ml]          copy
    python utils/sync_from_ccd_opt_ml.py --check                                              report differences only
    python utils/sync_from_ccd_opt_ml.py --dest superWADERs                                   only superWADERs' copy
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESTS = {"src": os.path.join(ROOT, "src", "raw_levels"),
         "superWADERs": os.path.join(ROOT, "superWADERs", "commissioning", "raw_levels")}
FILES = ["decode_trace.py", "trace_geometry.py", "trace_regions.py", "plot_trace.py", "view_trace.py", "decode_l1.py"]
MARK = "# copied from ccd_opt_ml/src/{name} (commit {commit}) by utils/sync_from_ccd_opt_ml.py: edit it there, not here\n"


def commit_of(src):
    """Short commit of the ccd_opt_ml checkout (with '+' if it has uncommitted changes), or 'unknown'."""
    try:
        c = subprocess.run(["git", "-C", src, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                           check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", src, "status", "--porcelain", "src"], capture_output=True, text=True).stdout
        return c + ("+" if dirty.strip() else "")
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=os.environ.get("CCD_OPT_ML", os.path.expanduser("~/Documents/Software/ccd_opt_ml")),
                    help="ccd_opt_ml checkout (default: $CCD_OPT_ML or ~/Documents/Software/ccd_opt_ml)")
    ap.add_argument("--check", action="store_true", help="only report which copies differ from the source")
    ap.add_argument("--dest", choices=("both", "src", "superWADERs"), default="both",
                    help="which copy to update (default both)")
    a = ap.parse_args()
    rc = 0
    for key, dst in DESTS.items():
        if a.dest in ("both", key):
            rc |= sync(a, dst)
    sys.exit(rc)


def sync(a, DST):
    """Copy (or with --check compare) FILES into DST; returns 1 if --check found differences."""
    commit = commit_of(a.source)
    os.makedirs(DST, exist_ok=True)
    stale = []
    for name in FILES:
        src = os.path.join(a.source, "src", name)
        if not os.path.isfile(src):
            sys.exit(f"sync_from_ccd_opt_ml: {src} not found (give --source)")
        body = open(src).read()
        dst = os.path.join(DST, name)
        old = open(dst).read() if os.path.exists(dst) else None
        if old is not None and old.split("\n", 1)[-1] == body:
            continue
        stale.append(name)
        if not a.check:
            with open(dst, "w") as fh:
                fh.write(MARK.format(name=name, commit=commit) + body)
    init = os.path.join(DST, "__init__.py")
    if not a.check and not os.path.exists(init):
        with open(init, "w") as fh:
            fh.write('"""Copies of ccd_opt_ml/src modules (utils/sync_from_ccd_opt_ml.py)."""\n')
    if a.check:
        print(f"{DST}: up to date with " + a.source if not stale else f"{DST}: differ from the source: " + ", ".join(stale))
        return 1 if stale else 0
    print(f"{DST}: copied {', '.join(stale) or 'nothing (up to date)'} from {a.source} (commit {commit})")
    return 0


if __name__ == "__main__":
    main()
