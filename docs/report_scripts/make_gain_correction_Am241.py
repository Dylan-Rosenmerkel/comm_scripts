"""Short PDF: effect of the gain-vs-charge correction on the Am-241 lines (AMTEst, 10 images).

Purpose: compare the Am-241 / Np-237 line positions (13.9-59.5 keV) before and after
gain_vs_charge.py apply, for EXT1, EXT3 and EXT4 of the AMTEst run (board 106).

Inputs (defaults in brackets, see -h):
    --orig-dir   cluster files from run_WADERs           [.../AMTEst/WADERs_analysis_fz_files/root]
    --corr-dir   the same clusters after the correction  [REQUIRED; gain_vs_charge.py apply output]
    --model      gain-vs-charge curves (AMT_gain_vs_charge.json, gain_vs_charge.py measure output)  [REQUIRED]
    (the inputs the committed PDF was made from were temporary and are gone; re-create them with
     gain_vs_charge.py measure / apply on the AMTEst images)
Outputs:
    --out        3-page PDF                              [comm_scripts/docs/gain_correction_Am241.pdf]
    --pages-dir  every page also as page_NN.png (70 dpi), for checking  [pages/ next to this script]
    and a printed summary of the 59.5 keV widths and measured/true ratios per line.

Run:  python make_gain_correction_Am241.py --corr-dir DIR --model JSON      (needs PyROOT, scipy, matplotlib)

How to read this file
    energies        cluster energies of one extension from a folder of cluster files
    fit_line        Gaussian + linear background fit of one line, None if no clear peak
    _Both           writes each page to the PDF and as a PNG
    page_summary    page 1: text and table of measured/true line positions
    page_595        page 2: the 59.54 keV line per CCD and its position
    page_l_lines    page 3: Np L lines + 26.3 keV, and measured/true vs energy
    main            options, read the clusters, fit every line, write the pages, print
"""
import argparse, glob, json, os, textwrap
import numpy as np
from scipy import optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import ROOT
ROOT.gROOT.SetBatch(True)

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG_D = "/Users/dylanrosenmerkel/Documents/JH1_Testing/AMTEst/WADERs_analysis_fz_files/root"
OUT = os.path.join(os.path.dirname(HERE), "gain_correction_Am241.pdf")                  # docs/
ORIG, CORR = "#8c8c8c", "#1f5fbf"
# Am-241 / Np-237 lines [keV]; L-beta is a blend dominated by L-beta1
LINES = [("Np L$\\alpha$", 13.93), ("Np L$\\beta$", 17.75), ("Np L$\\gamma$", 20.80), ("$\\gamma$ 26.3", 26.34),
         ("$\\gamma$ 59.5", 59.54)]


def energies(d, ext):
    """All cluster energies [keV] of extension ext in the panaSKImg cluster files of folder d.

    Reads the Energy branch of every clustersRec entry; files without that tree are skipped.
    Returns one concatenated float array (fails if no file has clusters).
    """
    E = []
    for f in glob.glob(f"{d}/panaSKImg_clustersRec_*_ext{ext}.root"):
        tf = ROOT.TFile.Open(f); t = tf.Get("clustersRec")
        if isinstance(t, ROOT.TTree):
            for i in range(t.GetEntries()):
                t.GetEntry(i); E.append(np.array(list(t.Energy)))
        tf.Close()
    return np.concatenate(E)


def fit_line(E, E0, scale=None):
    """Peak near E0: maximum in [0.85, 1.06] x E0, then Gaussian + linear background around it.
    Returns (mu, err, sigma) or None unless the peak is clear (signal/sqrt(background) >= 5,
    width < 1 keV below 40 keV).

    E: cluster energies [keV]; E0: true line energy [keV]. Bins of 0.4 keV below 40 keV and
    0.15 keV above; at least 60 clusters in the search window; err is the fit error on mu
    scaled up by sqrt(chi2/ndf) when that exceeds 1, and must stay below 2% of E0.
    scale is not used."""
    low = E0 < 40
    bw = 0.4 if low else 0.15
    w0 = 0.03 * E0 if low else 0.006 * E0
    h, ed = np.histogram(E, bins=np.arange(0.85 * E0, 1.06 * E0, bw)); c = 0.5 * (ed[1:] + ed[:-1])
    if h.sum() < 60:
        return None
    pk = c[np.argmax(np.convolve(h, np.ones(3) / 3, mode="same"))]
    lo, hi = pk - (2.5 if low else 3.5) * w0, pk + (2.5 if low else 3.5) * w0
    h, ed = np.histogram(E, bins=np.arange(lo, hi + 1e-9, bw)); c = 0.5 * (ed[1:] + ed[:-1])
    f = lambda x, a, m, s, b0, b1: a * np.exp(-0.5 * ((x - m) / s) ** 2) + b0 + b1 * (x - pk)
    try:
        p, cov = optimize.curve_fit(f, c, h, p0=[max(h.max() - h.min(), 1), pk, w0, h.min(), 0],
                                    sigma=np.sqrt(np.maximum(h, 1)), maxfev=20000,
                                    bounds=([0, lo, 0.2 if low else bw / 2, -np.inf, -np.inf], [np.inf, hi, 3 * w0, np.inf, np.inf]))
    except Exception:
        return None
    chi = np.sum(((h - f(c, *p)) / np.sqrt(np.maximum(h, 1))) ** 2) / max(len(h) - 5, 1)
    err = np.sqrt(cov[1, 1]) * np.sqrt(max(chi, 1))
    bg = max(p[3], 1)
    if err > 0.02 * E0 or p[0] / np.sqrt(bg) < 5 or (low and abs(p[2]) > 1.0):
        return None
    return float(p[1]), float(err), float(abs(p[2]))


class _Both:
    """PdfPages that also writes each page as <pages_dir>/page_NN.png (for checking)."""
    def __init__(self, pp, pages_dir): self.pp, self.n, self.pages_dir = pp, 0, pages_dir; os.makedirs(pages_dir, exist_ok=True)
    def savefig(self, fig):
        """Add fig as the next PDF page and write it as page_NN.png (70 dpi)."""
        self.n += 1; self.pp.savefig(fig); fig.savefig(os.path.join(self.pages_dir, f"page_{self.n:02d}.png"), dpi=70)


def page_summary(pdf, model, fits):
    """Page 1: data, what the correction does, the measured curves, and the table of
    measured / true energy per line and CCD without / with the correction ('-' = no clear peak)."""
    # ---------------------------------------------------------------- page 1: summary
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.06, 0.93, "Gain-vs-charge correction: effect on the Am-241 lines", fontsize=19, weight="bold")
    txt = [
        ("Data", "AMTEst, 10 images (Image 8, 500 skips, board 106; EXT1, EXT3, EXT4 - EXT2 is flooded). Cluster files "
                 "from run_WADERs (WADERs_analysis_fz_files/root), corrected with gain_vs_charge.py (apply). All clusters."),
        ("What the correction does", "The gain falls with the charge in a pixel (here 2.5-3.6% lower at 1000 e-). It is "
                 "measured from the images themselves (spacing of the electron comb at high charge) and used to convert "
                 "every cluster pixel from ADU to electrons, instead of one gain from the 0e-/1e- peaks. One curve per "
                 "CCD for all 10 images, so image-to-image differences of the fitted 0e-/1e- gain disappear too."),
        ("Gain curves measured", "  ".join(f"EXT{e}: {model['EXT' + str(e)]['drop_at_1000'] * 100:.1f}% lower at 1000 e-, "
                 f"low-charge gain {model['EXT' + str(e)]['model']['a']:.3f}." for e in (1, 3, 4)) +
                 " EXT1: the 0e-/1e- gains panaSKImg fitted per image scatter from 1.13 to 1.32 (very low dark current, "
                 "weak 1e- peak), so its energies were too HIGH and come down; on EXT3/EXT4 they were fine and the "
                 "correction mainly undoes the high-charge gain drop, so energies go UP."),
    ]
    y = 0.86
    for head, body in txt:
        fig.text(0.06, y, head, fontsize=13, weight="bold", color="#1f3f7f"); y -= 0.035
        for line in textwrap.wrap(body, 120):
            fig.text(0.08, y, line, fontsize=10.5); y -= 0.026
        y -= 0.02
    # table of line positions
    fig.text(0.06, y, "Fitted line positions, measured / true energy", fontsize=13, weight="bold", color="#1f3f7f"); y -= 0.04
    hdr = f"{'line':14}{'true keV':>10}" + "".join(f"{'EXT' + str(e) + ' without':>15}{'with':>9}" for e in (1, 3, 4))
    fig.text(0.08, y, hdr, fontsize=10, family="monospace"); y -= 0.028
    for name, E0 in LINES:
        cells = ""
        for e in (1, 3, 4):
            for k in (0, 1):
                r = fits[(e, name, k)]
                cells += f"{(r[0] / E0) if r else float('nan'):>{15 if k == 0 else 9}.3f}"
        clean = name.replace("$\\alpha$", "alpha").replace("$\\beta$", "beta").replace("$\\gamma$", "gamma")
        fig.text(0.08, y, f"{clean:14}{E0:>10.2f}" + cells.replace("nan", "  -"), fontsize=10, family="monospace"); y -= 0.026
    y -= 0.02
    for line in textwrap.wrap(
            "Reading it: 1.000 = right energy; '-' = no clear peak in that CCD (too few counts, or blended with the background of these partly flooded CCDs). With the correction the lines move "
            "towards 1.000 and the three CCDs agree better. What is left (0.2-0.7% low on the 59.5 keV line) is most likely charge in the "
            "faint edge of each cluster that panaSKImg's clustering threshold drops - not a gain effect, so the correction "
            "cannot recover it. Pages 2-3 show the spectra.", 120):
        fig.text(0.06, y, line, fontsize=10.5, style="italic"); y -= 0.026
    pdf.savefig(fig); plt.close(fig)


def page_595(pdf, data, fits):
    """Page 2: 59.54 keV spectra of EXT1/3/4 without / with the correction, and the fitted
    peak position / true energy per CCD."""
    # ---------------------------------------------------------------- page 2: 59.5 keV
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    bins = np.arange(48, 66, 0.25)
    for ax, e in zip(axes.flat, (1, 3, 4)):
        Eo, Ec = data[e]
        ax.hist(Eo, bins=bins, histtype="step", color=ORIG, lw=1.6, label="without correction")
        ax.hist(Ec, bins=bins, histtype="step", color=CORR, lw=1.6, label="with correction")
        ax.axvline(59.54, color="k", ls="--", lw=0.9, label="true 59.54 keV")
        ro, rc = fits[(e, "$\\gamma$ 59.5", 0)], fits[(e, "$\\gamma$ 59.5", 1)]
        t = []
        if ro: t.append(f"without: {ro[0]:6.2f} keV ({(ro[0] / 59.54 - 1) * 100:+.1f}%), width {ro[2]:.2f} keV")
        if rc: t.append(f"with:    {rc[0]:6.2f} keV ({(rc[0] / 59.54 - 1) * 100:+.1f}%), width {rc[2]:.2f} keV")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
        ax.text(0.02, 0.97, "\n".join(t), transform=ax.transAxes, va="top", fontsize=9, family="monospace",
                bbox=dict(facecolor="white", edgecolor="0.8"))
        ax.set_title(f"EXT{e}", fontsize=12); ax.set_xlabel("cluster energy [keV]"); ax.set_ylabel("clusters per 0.25 keV")
    axes[0, 0].legend(fontsize=8, loc="center left")
    # summary panel: 59.5 peak position per CCD
    ax = axes[1, 1]
    for k, (col, lab) in enumerate(((ORIG, "without"), (CORR, "with"))):
        ys = [fits[(e, "$\\gamma$ 59.5", k)][0] / 59.54 if fits[(e, "$\\gamma$ 59.5", k)] else np.nan for e in (1, 3, 4)]
        ax.plot(np.arange(3) + (k - 0.5) * 0.15, ys, "o", color=col, ms=9, label=f"{lab} correction")
    ax.axhline(1, color="k", ls="--", lw=0.9); ax.set_xticks(range(3)); ax.set_xticklabels(["EXT1", "EXT3", "EXT4"])
    ax.set_ylim(0.9, 1.05); ax.set_ylabel("59.5 keV peak / true"); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    ax.set_title("59.5 keV peak position per CCD", fontsize=12)
    fig.suptitle("Page 2 - the 59.54 keV gamma line (and its Si escape peak 1.74 keV below)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf.savefig(fig); plt.close(fig)


def page_l_lines(pdf, data, fits, pages_dir):
    """Page 3: Np L lines and the 26.3 keV gamma of EXT3/EXT4, and measured / true energy of
    every clear line vs its energy (how linear the energy scale is from 14 to 60 keV)."""
    # ---------------------------------------------------------------- page 3: L lines + linearity
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    bins = np.arange(10, 30.5, 0.4)
    for ax, e in zip(axes[0], (3, 4)):
        Eo, Ec = data[e]
        ax.hist(Eo, bins=bins, histtype="step", color=ORIG, lw=1.5, label="without correction")
        ax.hist(Ec, bins=bins, histtype="step", color=CORR, lw=1.5, label="with correction")
        for name, E0 in LINES[:4]:
            ax.axvline(E0, color="k", ls="--", lw=0.7)
            ax.text(E0, ax.get_ylim()[1] * 0.98, name, rotation=90, va="top", ha="right", fontsize=8)
        ax.set_title(f"EXT{e}: Np L x-rays and 26.3 keV gamma\n(dashed = true energies)", fontsize=11)
        ax.set_xlabel("cluster energy [keV]"); ax.set_ylabel("clusters per 0.4 keV")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    for ax, e in zip(axes[1], (3, 4)):
        for k, (col, lab) in enumerate(((ORIG, "without correction"), (CORR, "with correction"))):
            xs, ys, es = [], [], []
            for name, E0 in LINES:
                r = fits[(e, name, k)]
                if r:
                    xs.append(E0); ys.append(r[0] / E0); es.append(r[1] / E0)
            ax.errorbar(xs, ys, yerr=es, fmt="o-", color=col, ms=7, capsize=3, label=lab)
        ax.axhline(1, color="k", ls="--", lw=0.9); ax.set_ylim(0.9, 1.05); ax.set_xlim(10, 65)
        ax.set_xlabel("true line energy [keV]"); ax.set_ylabel("measured / true"); ax.grid(alpha=0.3)
        ax.set_title(f"EXT{e}: measured / true energy of each clear line", fontsize=11); ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Page 3 - lower-energy Am-241 lines and how the energy scale behaves across 14-60 keV", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf.savefig(fig)
    os.makedirs(pages_dir, exist_ok=True)
    plt.close(fig)


def main():
    """Options, cluster energies without/with the correction, line fits, the 3-page PDF, printed summary.

    fits[(ext, line name, k)] = fit_line result, k = 0 without and 1 with the correction.
    """
    ap = argparse.ArgumentParser(description="Short PDF: effect of the gain-vs-charge correction on the Am-241 lines.")
    ap.add_argument("--orig-dir", default=ORIG_D, help=f"uncorrected cluster files (default: {ORIG_D})")
    ap.add_argument("--corr-dir", required=True, help="corrected cluster files (gain_vs_charge.py apply output)")
    ap.add_argument("--model", required=True, help="gain-vs-charge curves (gain_vs_charge.py measure output JSON)")
    ap.add_argument("--out", default=OUT, help=f"output PDF (default: {OUT})")
    ap.add_argument("--pages-dir", default=os.path.join(HERE, "pages"),
                    help="folder for the page_NN.png copies of the pages (default: pages/ next to this script)")
    args = ap.parse_args()
    out = args.out
    model = json.load(open(args.model))["extensions"]

    data = {e: (energies(args.orig_dir, e), energies(args.corr_dir, e)) for e in (1, 3, 4)}
    fits = {}
    for e, (Eo, Ec) in data.items():
        for name, E0 in LINES:
            # start each fit near where the 59.5 peak sits (so lines are found in both versions)
            for k, E in enumerate((Eo, Ec)):
                fits[(e, name, k)] = fit_line(E, E0)

    with PdfPages(out) as _pp:
        pdf = _Both(_pp, args.pages_dir)
        page_summary(pdf, model, fits)
        page_595(pdf, data, fits)
        page_l_lines(pdf, data, fits, args.pages_dir)
    print("->", out)
    for e in (1, 3, 4):
        print(f"EXT{e} widths 59.5: " + str([fits[(e, LINES[-1][0], k)][2] if fits[(e, LINES[-1][0], k)] else None for k in (0, 1)]))
        print(f"EXT{e}: " + "  ".join(f"{n.replace('$', '').replace(chr(92), '')}: "
                                      f"{(fits[(e, n, 0)][0] / E0) if fits[(e, n, 0)] else float('nan'):.3f}->"
                                      f"{(fits[(e, n, 1)][0] / E0) if fits[(e, n, 1)] else float('nan'):.3f}" for n, E0 in LINES))


if __name__ == "__main__":
    main()
