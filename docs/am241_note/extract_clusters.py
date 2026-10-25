#!/usr/bin/env python3
"""Step 2 of the note: read the cluster files of both runs into one table per CCD.

The cluster files (panaSKImg_clustersRec_<image>_ext<N>.root) hold one ROOT tree entry per image, and in it one
vector element per cluster. This script flattens them into numpy arrays (one element per cluster, all images of a CCD
together) and saves them as <work-dir>/clusters2_ext<N>.npz, which every later script reads (common.load_clusters).

Both runs found the same clusters - the gain correction only changes their energies - so the table stores the
corrected quantities (from root_gaincorr_epochs/) plus E_orig, the energy without gain correction (from root/ of
the uncorrected run). The script checks that the cluster positions agree.

Run with the python that has PyROOT (e.g. the conda environment with panaSKImg):
    python extract_clusters.py [--run-dir DIR] [--run-name NAME]
"""
import argparse
import glob
import os
import sys

import numpy as np
import ROOT

from common import add_path_args, cluster_table, paths

# Per-cluster branches copied from the clustersRec tree (panaSKImg names):
#   Energy [keV], Npix (pixels in the cluster), fSTD_X/fSTD_Y (Gaussian-fit widths of the projections) and
#   wSTD_X/Y/XY (charge-weighted RMS widths) [pixels], PCA shape variables, distance to the nearest cluster,
#   position, the brightest pixel's charge Qmax [keV] and the bounding box.
BRANCHES = ["Energy", "Npix", "fSTD_X", "fSTD_Y", "wSTD_X", "wSTD_Y", "wSTD_XY", "pcaElongation", "pcaLinearity",
            "pcaLength95", "closest_cls_dist", "PosX", "PosY", "Qmax", "minX", "maxX", "minY", "maxY"]


def read_ccd(folder, ext):
    """All clusters of CCD 'ext' in the cluster files of 'folder', as {branch: array}; 'img' numbers the images."""
    cols = {b: [] for b in BRANCHES}
    img = []
    for k, f in enumerate(sorted(glob.glob(f"{folder}/panaSKImg_clustersRec_*_ext{ext}.root"))):
        tf = ROOT.TFile.Open(f)
        t = tf.Get("clustersRec")
        for i in range(t.GetEntries()):
            t.GetEntry(i)
            for b in BRANCHES:
                cols[b].append(np.array(list(getattr(t, b)), float))
            img.append(np.full(len(t.Energy), k))
        tf.Close()
    out = {b: np.concatenate(v) for b, v in cols.items()}
    out["img"] = np.concatenate(img)
    return out


def main():
    ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--ext", type=int, nargs="+", default=[1, 3, 4], help="CCDs to read (default: 1 3 4)")
    args = ap.parse_args()
    p = paths(args)
    ROOT.gROOT.SetBatch(True)
    for ext in args.ext:
        o = read_ccd(os.path.join(p.orig, "root"), ext)                    # no gain correction
        c = read_ccd(os.path.join(p.corr, "root_gaincorr_epochs"), ext)    # corrected per gain period
        same = len(o["Energy"]) == len(c["Energy"]) and np.allclose(o["PosX"], c["PosX"]) and np.allclose(o["PosY"], c["PosY"])
        print(f"ext{ext}: {len(c['Energy'])} clusters, same clusters in both runs: {same}")
        if not same:
            # E_orig must belong to the same clusters as the corrected table (fit_all_peaks indexes both with one mask)
            sys.exit(f"ext{ext}: the corrected cluster files do not match the originals (an image missing from "
                     f"root_gaincorr_epochs/? check correct_gain_per_period.py --periods); nothing written")
        np.savez(cluster_table(p.work, ext), E_orig=o["Energy"], **c)


if __name__ == "__main__":
    main()
