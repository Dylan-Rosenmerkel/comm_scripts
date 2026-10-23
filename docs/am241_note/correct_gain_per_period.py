#!/usr/bin/env python3
"""Step 1 of the note: gain-vs-charge correction with one g(n) curve per CCD and *gain period*.

Why
---
run_WADERs --gain-correction measures one gain curve g(n) per CCD from all images together and converts every cluster
with it. That assumes the gain did not change during the run. In the AMtest data it did: the per-image fitted gain is
1-5 % lower for images 41-50 (2026-05-21 to 05-26) than for images 21-40, and stable within each period. One curve for
all images then leaves the later images ~3 % low and splits the summed peaks. The fix is to measure and apply g(n)
separately for each period.

What it does (with comm_scripts/src/gain_vs_charge.py, the same tool run_WADERs uses)
---------------------------------------------------------------------------------------
  1. measure: for each period, g(n) from the electron comb of that period's images (--ext 1 3 4, default model)
       -> <corr>/gain_vs_charge_epochs/epoch<i>_img<a>-<b>_gain_vs_charge.json (+ .png)
  2. apply:   each period's curve to that period's cluster files from <corr>/root/
       -> <corr>/root_gaincorr_epochs/ and its root_files.txt
  3. optional (--composite-macro): MakeComposite_Fe55_Clusters_Extensions.cc on the corrected files
       -> <corr>/composite_epochs/
where <corr> = <run-dir>/gaincorr/WADERs_analysis_<run-name>. Image numbers are the trailing _<N> of the image names.

Run:  python correct_gain_per_period.py --list <run-dir>/AMtest_29.txt --periods 21-40 41-50 [--composite-macro PATH]
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

from common import REPO, add_path_args, paths

GVC = os.path.join(REPO, "src", "gain_vs_charge.py")


def image_number(path):
    """Image number N from a name ending in _<N>.fz / .fits / ... / _<N>_ext<k>.root."""
    m = re.search(r"_(\d+)(?:_ext\d)?\.(?:fz|fits|root)", os.path.basename(path))
    return int(m.group(1)) if m else None


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--list", required=True, help="the .txt image list given to run_WADERs")
    ap.add_argument("--periods", nargs="+", required=True, metavar="A-B",
                    help="gain periods as image-number ranges, e.g. 21-40 41-50")
    ap.add_argument("--ext", nargs="+", default=["1", "3", "4"], help="CCDs to measure (default 1 3 4)")
    ap.add_argument("--composite-macro", help="also run MakeComposite_Fe55_Clusters_Extensions.cc on the result")
    ap.add_argument("--image-type", type=int, default=8, help="image type for MakeComposite (default 8)")
    ap.add_argument("--module", default="ACM106", help="module name for MakeComposite (default ACM106)")
    args = ap.parse_args()
    p = paths(args)
    gdir, odir = os.path.join(p.corr, "gain_vs_charge_epochs"), os.path.join(p.corr, "root_gaincorr_epochs")
    os.makedirs(gdir, exist_ok=True); os.makedirs(odir, exist_ok=True)
    base = os.path.dirname(os.path.abspath(args.list))
    images = [l.strip() if os.path.isabs(l.strip()) else os.path.join(base, l.strip())
              for l in open(args.list) if l.strip() and not l.startswith("#")]

    for i, period in enumerate(args.periods, 1):
        a, b = (int(v) for v in period.split("-"))
        imgs = [f for f in images if a <= (image_number(f) or -1) <= b]
        roots = sorted(f for f in glob.glob(os.path.join(p.corr, "root", "panaSKImg_clustersRec_*.root"))
                       if a <= (image_number(f) or -1) <= b)
        name = f"epoch{i}_img{a}-{b}"
        print(f"period {name}: {len(imgs)} images, {len(roots)} cluster files")
        # 1. measure g(n) from this period's images only
        with open(os.path.join(gdir, f"{name}_measure.log"), "w") as log:
            subprocess.run([sys.executable, GVC, "measure", *imgs, "-o", gdir, "--name", name, "--ext", *args.ext],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        # 2. apply it to this period's cluster files
        with open(os.path.join(gdir, f"{name}_apply.log"), "w") as log:
            subprocess.run([sys.executable, GVC, "apply", os.path.join(gdir, f"{name}_gain_vs_charge.json"), *roots,
                            "-o", odir], stdout=log, stderr=subprocess.STDOUT, check=True)
    files = sorted(glob.glob(os.path.join(odir, "*.root")))
    with open(os.path.join(odir, "root_files.txt"), "w") as fh:
        fh.write("".join(f + "\n" for f in files))
    print(f"corrected cluster files: {len(files)} in {odir}")

    # 3. optional: MakeComposite on the corrected files (same call as run_WADERs --composite)
    if args.composite_macro:
        cdir = os.path.join(p.corr, "composite_epochs"); os.makedirs(os.path.join(cdir, "lists"), exist_ok=True)
        lst = os.path.join(cdir, "lists", f"Image_{args.image_type}_{args.module}_list.txt")
        shutil.copy(os.path.join(odir, "root_files.txt"), lst)
        root_exe = shutil.which("root") or os.path.join(os.path.dirname(sys.executable), "root")
        call = f'{os.path.abspath(args.composite_macro)}+("{lst}",{args.image_type},"{args.module}","{cdir}")'
        with open(os.path.join(p.corr, "logs", f"composite_epochs_Image_{args.image_type}_{args.module}.log"), "w") as log:
            subprocess.run([root_exe, "-l", "-q", "-b", call], cwd=cdir, stdout=log, stderr=subprocess.STDOUT)
        print(f"MakeComposite output in {cdir}")


if __name__ == "__main__":
    main()
