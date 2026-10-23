"""Shared settings and helpers for the Am-241 10-30 keV note scripts.

Every script in this folder works on the output of two run_WADERs runs over the same images:

    <run-dir>/no_gaincorr/WADERs_analysis_<run-name>/   run_WADERs ... --composite
    <run-dir>/gaincorr/WADERs_analysis_<run-name>/      run_WADERs ... --composite --gain-correction
        root_gaincorr_epochs/    cluster files corrected with one g(n) curve per gain period
                                 (made by correct_gain_per_period.py)

and keeps its intermediate files (cluster tables, the band-cut grid) in a work folder, <run-dir>/analysis by default.
Figures and LaTeX tables for the note go to note/ next to this file (--note-dir to change).

The defaults reproduce the note (29 AMtest images, folder ~/Documents/JH1_Testing/AMTEst/runs_29img); set
$AM241_RUN_DIR or pass --run-dir to point at other data.
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))                      # comm_scripts/
DEFAULT_RUN_DIR = os.environ.get("AM241_RUN_DIR", os.path.expanduser("~/Documents/JH1_Testing/AMTEst/runs_29img"))
DEFAULT_RUN_NAME = "AMtest_29"

# The cluster-width cut chosen in the note (sigma_xy < 0.7 pixels; Sec. 4 of the note): compact clusters are the
# low-energy x rays absorbed close to the surface, the continuum under them is mostly wider.
CUT = 0.7


def add_path_args(ap):
    """Add the --run-dir / --run-name / --work-dir / --note-dir options shared by all scripts."""
    ap.add_argument("--run-dir", default=DEFAULT_RUN_DIR,
                    help="folder with no_gaincorr/ and gaincorr/ run_WADERs outputs (default: $AM241_RUN_DIR or "
                         "~/Documents/JH1_Testing/AMTEst/runs_29img)")
    ap.add_argument("--run-name", default=DEFAULT_RUN_NAME,
                    help="run name of the run_WADERs outputs, WADERs_analysis_<run-name> (default: AMtest_29)")
    ap.add_argument("--work-dir", default=None,
                    help="intermediate files (cluster tables, band grid); default <run-dir>/analysis")
    ap.add_argument("--note-dir", default=os.path.join(HERE, "note"),
                    help="where figures/ and tables/ of the note are written (default: note/ next to this script)")
    return ap


def paths(args):
    """Resolve the folders used by the scripts from the parsed command-line arguments."""
    run = os.path.abspath(os.path.expanduser(args.run_dir))
    p = argparse.Namespace(
        run=run,
        orig=os.path.join(run, "no_gaincorr", f"WADERs_analysis_{args.run_name}"),   # no gain correction
        corr=os.path.join(run, "gaincorr", f"WADERs_analysis_{args.run_name}"),      # gain-corrected run
        work=os.path.abspath(os.path.expanduser(args.work_dir or os.path.join(run, "analysis"))),
        note=os.path.abspath(os.path.expanduser(args.note_dir)))
    os.makedirs(p.work, exist_ok=True)
    return p


def cluster_table(work, ext):
    """Path of the per-cluster table of one CCD written by extract_clusters.py."""
    return os.path.join(work, f"clusters2_ext{ext}.npz")


def load_clusters(work, ext):
    """Per-cluster quantities of one CCD as a dict of equal-length arrays (one entry per cluster).

    Read from the table written by extract_clusters.py. Energies are in keV:
      Energy   corrected with one g(n) curve per gain period,   E_orig  without gain correction.
    Added here:
      sxy      cluster width sigma_xy = sqrt((sigma_x^2 + sigma_y^2) / 2) [pixels], from panaSKImg's Gaussian fits of
               the cluster's column and row projections (fSTD_X, fSTD_Y) - the width the note cuts on;
      qfrac    fraction of the cluster's charge in its brightest pixel (Qmax / Energy);
      nx, ny   size of the cluster's bounding box in columns and rows [pixels].
    'img' is the index of the image in name order (0 = first image).
    """
    d = dict(np.load(cluster_table(work, ext)))
    d["sxy"] = np.sqrt(0.5 * (d["fSTD_X"] ** 2 + d["fSTD_Y"] ** 2))
    d["qfrac"] = d["Qmax"] / np.maximum(d["Energy"], 1e-6)
    d["nx"] = d["maxX"] - d["minX"] + 1
    d["ny"] = d["maxY"] - d["minY"] + 1
    return d


# Plot style shared by the figures: recessive grey axes and grid, categorical colours in a fixed order.
INK, MUTED, GRID, DATA = "#1a1a19", "#6b6a63", "#d9d8d2", "#9a9890"
STYLE = {"axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
         "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
         "axes.spines.right": False, "font.size": 9, "pdf.fonttype": 42}
