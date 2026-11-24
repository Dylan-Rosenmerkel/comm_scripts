#!/usr/bin/env python3
"""Copy the defect-map script of LSM_Module_Testing_Analysis (D. Venegas-Vargas: src/Image_4.py) into
comm_scripts/src/lsm_defects/ (run_WADERs --engine native), unchanged except for a first line naming the source
and its commit. CICERO keeps its own copies (its tools/sync_from_lsm_analysis.py).

--defect-map (src/defects.py) runs it on the run's images (its geometry constants set from each image's header)
and adds the defect log and the interactive defect map.

    python utils/sync_from_lsm_analysis.py [--source ~/Documents/LSM_Module_Testing_Analysis] [--check]
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESTS = [(os.path.join(ROOT, "src", "lsm_defects"), ["Image_4.py"])]
MARK = ("{c} copied from LSM_Module_Testing_Analysis/src/{name} (commit {commit}) by utils/sync_from_lsm_analysis.py: "
        "edit it there, not here\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=os.path.expanduser("~/Documents/LSM_Module_Testing_Analysis"))
    ap.add_argument("--check", action="store_true", help="only report which copies differ from the source")
    a = ap.parse_args()
    rc = 0
    for dst, files in DESTS:
        rc |= sync(a, dst, files)
    sys.exit(rc)


def sync(a, DST, FILES):
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
        dirty = subprocess.run(["git", "-C", a.source, "status", "--porcelain", "--", os.path.join("src", name)],
                               capture_output=True, text=True).stdout.strip()
        dst = os.path.join(DST, name)
        if os.path.exists(dst) and open(dst).read().split("\n", 1)[-1] == body:
            continue
        stale.append(name)
        if not a.check:
            with open(dst, "w") as fh:
                fh.write(MARK.format(c="//" if name.endswith(".cc") else "#", name=name,
                                     commit=commit + (" + uncommitted changes" if dirty else "")) + body)
    if not a.check and any(f.endswith(".py") for f in FILES) and not os.path.exists(os.path.join(DST, "__init__.py")):
        open(os.path.join(DST, "__init__.py"), "w").write('"""Copies of LSM_Module_Testing_Analysis/src defect scripts."""\n')
    if a.check:
        print(f"{DST}: up to date" if not stale else f"{DST}: differ from the source: " + ", ".join(stale))
        return 1 if stale else 0
    print(f"{DST}: copied {', '.join(stale) or 'nothing (up to date)'} (commit {commit})")
    return 0


if __name__ == "__main__":
    main()
