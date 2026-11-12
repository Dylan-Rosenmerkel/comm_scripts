#!/usr/bin/env python3
"""Copy the defect-map script of LSM_Module_Testing_Analysis (D. Venegas-Vargas: src/Image_4.py) into
comm_scripts/src/lsm_defects/ (run_WADERs --engine native) and superWADERs/commissioning/lsm_defects/ (run_superWADERs),
unchanged except for a first line naming the source and its commit (--dest: only one of them).

--defect-map (src/defects.py, superWADERs/commissioning/defects.py) runs it on the run's images (its geometry
constants set from each image's header) and adds the defect log and the interactive defect map.

    python utils/sync_from_lsm_analysis.py [--source ~/Documents/LSM_Module_Testing_Analysis] [--check]
                                           [--dest both|src|superWADERs]
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESTS = {"src": os.path.join(ROOT, "src", "lsm_defects"),
         "superWADERs": os.path.join(ROOT, "superWADERs", "commissioning", "lsm_defects")}
FILES = ["Image_4.py"]
MARK = ("# copied from LSM_Module_Testing_Analysis/src/{name} (commit {commit}) by utils/sync_from_lsm_analysis.py: "
        "edit it there, not here\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=os.path.expanduser("~/Documents/LSM_Module_Testing_Analysis"))
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
    try:
        commit = subprocess.run(["git", "-C", a.source, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        commit = "unknown"
    os.makedirs(DST, exist_ok=True)
    stale = []
    for name in FILES:
        body = open(os.path.join(a.source, "src", name)).read()
        dst = os.path.join(DST, name)
        if os.path.exists(dst) and open(dst).read().split("\n", 1)[-1] == body:
            continue
        stale.append(name)
        if not a.check:
            with open(dst, "w") as fh:
                fh.write(MARK.format(name=name, commit=commit) + body)
    if not a.check and not os.path.exists(os.path.join(DST, "__init__.py")):
        open(os.path.join(DST, "__init__.py"), "w").write('"""Copies of LSM_Module_Testing_Analysis/src defect scripts."""\n')
    if a.check:
        print(f"{DST}: up to date" if not stale else f"{DST}: differ from the source: " + ", ".join(stale))
        return 1 if stale else 0
    print(f"{DST}: copied {', '.join(stale) or 'nothing (up to date)'} (commit {commit})")
    return 0


if __name__ == "__main__":
    main()
