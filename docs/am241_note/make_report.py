#!/usr/bin/env python3
"""The 9-page PDF report on the gain correction and the Am-241 lines (a companion to the note).

Pages: 1 summary (gain curves per gain period, fitted line positions without / with the correction); 2 the gain curves
and the per-image fitted gains they replace; 3 the full Am-241 spectrum per CCD; 4 the 59.54 keV line; 5 the Np L x
rays and 26.3 keV line with the energy scale across 14-60 keV; 6 the 59.54 keV peak image by image, with the gain step
between images 40 and 41 marked; 7 MakeComposite's Am-241 plots from both runs; 8 the cluster-width cut, why and how;
9 the low-energy lines after the cut. Corrected energies are those of root_gaincorr_epochs/ (one g(n) curve per gain
period, correct_gain_per_period.py); line positions come from simple Gaussian + linear fits (fit_line()).

Output: <out-dir>/<run-name>_gain_correction_Am241.pdf and <out-dir>/pages/page_NN.png (default out-dir: the run folder).
Run:  python make_report.py [--run-dir DIR] [--out-dir DIR]
"""
import argparse
import csv, glob, json, os, re, textwrap
import numpy as np
from scipy import optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import ROOT
ROOT.gROOT.SetBatch(True)

ORIG, CORR = "#eb6834", "#2a78d6"            # categorical slots 2 / 1: without / with correction
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#d9d8d2"
LINES = [("Np L$\\alpha$", 13.93), ("Np L$\\beta$", 17.75), ("Np L$\\gamma$", 20.80), ("$\\gamma$ 26.3", 26.34),
         ("$\\gamma$ 59.5", 59.54)]
G595 = LINES[-1][0]
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5})


def clean(name):
    return name.replace("$\\alpha$", "alpha").replace("$\\beta$", "beta").replace("$\\gamma$", "gamma")


def read_run(d):
    """({ext: {image stem: cluster energies [keV]}}, same for the cluster width sigma_xy [pix]) from
    <d>/panaSKImg_clustersRec_*_ext<N>.root; sigma_xy = sqrt((fSTD_X^2 + fSTD_Y^2) / 2) as in MakeComposite."""
    out, wid = {}, {}
    for f in sorted(glob.glob(os.path.join(d, "panaSKImg_clustersRec_*_ext*.root"))):
        m = re.match(r"panaSKImg_clustersRec_(.+)_ext(\d)\.root$", os.path.basename(f))
        tf = ROOT.TFile.Open(f); t = tf.Get("clustersRec")
        E, S = [], []
        if isinstance(t, ROOT.TTree):
            for i in range(t.GetEntries()):
                t.GetEntry(i); E.append(np.array(list(t.Energy), float))
                S.append(np.sqrt(0.5 * (np.array(list(t.fSTD_X), float) ** 2 + np.array(list(t.fSTD_Y), float) ** 2)))
        tf.Close()
        ext, stem = int(m.group(2)), m.group(1)
        out.setdefault(ext, {})[stem] = np.concatenate(E) if E else np.zeros(0)
        wid.setdefault(ext, {})[stem] = np.concatenate(S) if S else np.zeros(0)
    return out, wid


def fit_line(E, E0, min_counts=60):
    """Gaussian + linear background around the line near E0; (mu, err, sigma) or None if not clear."""
    low = E0 < 40
    bw = 0.4 if low else 0.15
    w0 = 0.03 * E0 if low else 0.006 * E0
    h, ed = np.histogram(E, bins=np.arange(0.85 * E0, 1.06 * E0, bw)); c = 0.5 * (ed[1:] + ed[:-1])
    if h.sum() < min_counts:
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
    if err > 0.02 * E0 or p[0] / np.sqrt(max(p[3], 1)) < 5 or (low and abs(p[2]) > 1.0):
        return None
    if abs(p[1] / E0 - 1) > 0.07:             # a bump far from the line is background, not the line
        return None
    return float(p[1]), float(err), float(abs(p[2]))


def fmt_ratio(r, E0):
    return f"{r[0] / E0:.3f}" if r else "-"

from common import add_path_args, paths


def main():
    """Read both runs, fit the lines and write the PDF report."""
    _ap = add_path_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    _ap.add_argument("--out-dir", default=None, help="where the PDF and pages/ go (default: the run folder)")
    _args = _ap.parse_args()
    PATHS = paths(_args)
    HERE = os.path.abspath(_args.out_dir or PATHS.run)
    RUN_O, RUN_C = PATHS.orig, PATHS.corr
    OUT = os.path.join(HERE, f"{_args.run_name}_gain_correction_Am241.pdf")
    # ------------------------------------------------------------------ data
    orig, orig_w = read_run(os.path.join(RUN_O, "root"))
    # gain correction per gain period: the gain stepped down between images 40 and 41, so g(n) was measured and applied
    # separately for each period (gain_vs_charge.py measure/apply -> gain_vs_charge_epochs/, root_gaincorr_epochs/)
    corr, corr_w = read_run(os.path.join(RUN_C, "root_gaincorr_epochs"))
    exts = sorted(set(orig) & set(corr))
    allE = {e: (np.concatenate(list(orig[e].values())), np.concatenate(list(corr[e].values()))) for e in exts}
    allS = {e: (np.concatenate(list(orig_w[e].values())), np.concatenate(list(corr_w[e].values()))) for e in exts}
    # width cut (pages 8-9): keep compact clusters, where the low-energy lines are
    SXY_CUT = 0.7
    CUTC, ALLC = "#1baf7a", "#9a9890"             # categorical slot 3 = after the cut; neutral gray = all clusters
    cutE = {e: tuple(allE[e][k][allS[e][k] < SXY_CUT] for k in (0, 1)) for e in exts}
    cfits = {(e, n, k): fit_line(cutE[e][k], E0) for e in exts for n, E0 in LINES for k in (0, 1)}
    # peak window / sideband window [keV] used for the peak-to-background ratios (gain-corrected energies)
    PB = [("Np L$\\beta$", (17.1, 18.1), (15.5, 16.5)), ("Np L$\\gamma$", (20.5, 21.5), (23.0, 24.0)),
          ("$\\gamma$ 26.3", (25.7, 26.7), (28.0, 29.0))]


    def peak_bg(E, win, side):
        return np.sum((E > win[0]) & (E < win[1])), np.sum((E > side[0]) & (E < side[1]))
    fits = {(e, n, k): fit_line(allE[e][k], E0) for e in exts for n, E0 in LINES for k in (0, 1)}
    stems = sorted(set().union(*[orig[e] for e in exts]))
    per_img = {(e, s, k): fit_line((orig if k == 0 else corr)[e].get(s, np.zeros(0)), 59.54, min_counts=30)
               for e in exts for s in stems for k in (0, 1)}
    periods = []                                  # (label, {EXTn: curve}, set of image stems)
    for p in sorted(glob.glob(os.path.join(RUN_C, "gain_vs_charge_epochs", "*_gain_vs_charge.json"))):
        js = json.load(open(p))
        lab = re.search(r"img(\d+-\d+)", os.path.basename(p)).group(1)
        periods.append((f"images {lab}", js["extensions"], {os.path.basename(i).split(".")[0] for i in js["images"]}))
    fitgain = {}                                  # (stem, EXTn) -> panaSKImg fitted gain (no correction run)
    with open(os.path.join(RUN_O, "fit_quality.csv")) as fh:
        for r in csv.DictReader(fh):
            fitgain[(r["image"], r["ext"])] = (float(r["gain"]), float(r["gain_err_s"]))
    n_img = len(stems)
    date = lambda s: re.search(r"_(\d{8})_\d{6}_(\d+)$", s)


    class Book:
        """PdfPages plus a PNG per page (pages/page_NN.png) for a quick look."""
        def __init__(self, pp): self.pp, self.n = pp, 0; os.makedirs(os.path.join(HERE, "pages"), exist_ok=True)
        def save(self, fig):
            self.n += 1; self.pp.savefig(fig); fig.savefig(os.path.join(HERE, "pages", f"page_{self.n:02d}.png"), dpi=70)
            plt.close(fig)


    def page_title(fig, n, title, sub=None):
        fig.text(0.05, 0.955, f"{n}. {title}", fontsize=15, weight="bold", color=INK)
        if sub:
            fig.text(0.05, 0.94, sub, fontsize=10, color=MUTED, va="top")


    def hist_pair(ax, e, bins, log=False, norm_w=None):
        Eo, Ec = allE[e]
        ax.hist(Eo, bins=bins, histtype="step", color=ORIG, lw=1.4, label="without correction")
        ax.hist(Ec, bins=bins, histtype="step", color=CORR, lw=1.4, label="with correction")
        if log:
            ax.set_yscale("log")


    def mark_lines(ax, lines, top_frac=0.98):
        ymax = ax.get_ylim()[1]
        for name, E0 in lines:
            ax.axvline(E0, color=MUTED, ls="--", lw=0.7)
            ax.text(E0, ymax * (0.6 if ax.get_yscale() == "log" else top_frac), " " + name, rotation=90,
                    va="top", ha="left", fontsize=7.5, color=MUTED)


    with PdfPages(OUT) as pp:
        book = Book(pp)
        # ---------------------------------------------------------------- 1. summary
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.05, 0.94, "AMtest Am-241: gain-vs-charge correction and the Am-241 lines", fontsize=18, weight="bold", color=INK)
        fig.text(0.05, 0.91, f"{n_img} images, Image 8, board (ACM) 106, CCDs EXT{', EXT'.join(map(str, exts))} "
                 f"- run_WADERs --composite; gain correction (interpolated g(n)) per gain period", fontsize=10.5, color=MUTED)
        y = 0.86
        fig.text(0.05, y, "Gain curve g(n) measured from the images (one curve per CCD and gain period)",
                 fontsize=12, weight="bold", color=INK); y -= 0.035
        for (plab, pm, pst), e in [(q, e) for q in periods for e in exts]:
            r = pm.get(f"EXT{e}", {})
            if r.get("ok"):
                m = r["model"]
                gs = [fitgain[(s, f'EXT{e}')][0] for s in stems if (s, f"EXT{e}") in fitgain and s in pst]
                spread = f"; per-image 0e-/1e- fitted gain {min(gs):.3f}-{max(gs):.3f}" if gs else ""
                txt = (f"{plab}, EXT{e}: low-charge gain {m['a']:.4f} ADU/e-, "
                       f"{r['drop_at_1000'] * 100:.1f}% lower at 1000 e-, "
                       f"measured to {r['n_max_measured']:.0f} e-{spread}")
            else:
                txt = f"{plab}, EXT{e}: no correction ({r.get('reason', 'no curve')}) - files copied unchanged"
            fig.text(0.07, y, txt, fontsize=9, color=INK); y -= 0.022
        y -= 0.02
        fig.text(0.05, y, "Fitted line position / true energy (all images together; 1.000 = right energy, - = no clear peak)",
                 fontsize=12, weight="bold", color=INK); y -= 0.04
        hdr = f"{'line':12}{'keV':>7}" + "".join(f"{'EXT' + str(e) + ' without':>15}{'with':>7}" for e in exts)
        fig.text(0.07, y, hdr, fontsize=9.5, family="monospace", color=MUTED); y -= 0.027
        for name, E0 in LINES:
            cells = "".join(f"{fmt_ratio(fits[(e, name, 0)], E0):>15}{fmt_ratio(fits[(e, name, 1)], E0):>7}" for e in exts)
            fig.text(0.07, y, f"{clean(name):12}{E0:>7.2f}" + cells, fontsize=9, family="monospace", color=INK); y -= 0.022
        y -= 0.015
        fig.text(0.05, y, "Reading it", fontsize=12, weight="bold", color=INK); y -= 0.03
        notes = []
        for e in exts:
            ro, rc = fits[(e, G595, 0)], fits[(e, G595, 1)]
            nwin = np.sum((allE[e][0] > 53) & (allE[e][0] < 62))
            if nwin < 1000:
                notes.append(f"EXT{e}: only {nwin} clusters in 53-62 keV over all images - no usable 59.5 keV peak, its "
                             "numbers are not meaningful.")
            elif ro and rc:
                notes.append(f"EXT{e}: 59.5 keV peak {ro[0] / 59.54:.3f} -> {rc[0] / 59.54:.3f} of the true energy, "
                             f"width {ro[2]:.2f} -> {rc[2]:.2f} keV.")
        notes.append("Without correction the peak sits low (the gain falls at high charge) and moves image to image with "
                     "panaSKImg's per-image 0e-/1e- gain (page 6). The gain dropped ~3% between images 40 and 41 (2026-05-18 / 05-21), so one g(n) per CCD and period is used; one curve for all images would leave images 41-50 ~3% low.")
        cut_txt = "; ".join(
            f"EXT{e} " + ", ".join(f"{clean(n)} {fmt_ratio(cfits[(e, n, 0)], E0)} -> {fmt_ratio(cfits[(e, n, 1)], E0)}"
                                   for n, E0 in LINES[:4] if cfits[(e, n, 0)] or cfits[(e, n, 1)])
            for e in exts if any(cfits[(e, n, k)] for n, _ in LINES[:4] for k in (0, 1)))
        notes.append("The Np L lines (14-21 keV) are weak on top of the background with all clusters (table above). A cut on "
                     f"the cluster width (sigma_xy < {SXY_CUT}, pages 8-9) doubles their peak-to-background and lets them be "
                     f"fitted - measured/true without -> with correction: {cut_txt}.")
        notes.append("Only lines that fit cleanly within 7% of their energy are shown. g(n) is measured to 480-1130 e- per "
                     "pixel (see above) and extrapolated above that.")
        for n in notes:
            for line in textwrap.wrap(n, 135):
                fig.text(0.07, y, line, fontsize=9, color=INK); y -= 0.02
        y -= 0.01
        fig.text(0.05, y, "Pages", fontsize=12, weight="bold", color=INK); y -= 0.03
        for line in ["2. gain vs charge per pixel, and the per-image fitted gains it replaces",
                     "3. full Am-241 cluster-energy spectrum per CCD, without / with",
                     "4. the 59.5 keV gamma line per CCD, and its position",
                     "5. the Np L x-rays and 26.3 keV line; measured/true across 14-60 keV",
                     "6. the 59.5 keV peak image by image (stability over all images)",
                     "7. MakeComposite's Am-241 2x2 plots from both runs",
                     f"8. a cut on the cluster width (sigma_xy < {SXY_CUT}) against the background: why and how",
                     "9. the low-energy lines after the cut: spectra, peak-to-background, energy scale"]:
            fig.text(0.07, y, line, fontsize=9, color=INK); y -= 0.02
        book.save(fig)

        # ---------------------------------------------------------------- 2. gain curves + per-image gains
        fig = plt.figure(figsize=(11, 8.5))
        page_title(fig, 2, "Gain vs charge per pixel",
                   "top: g(n) from the electron comb (gain_vs_charge.py).  bottom: the 0e-/1e- gain panaSKImg fitted on each "
                   "image (no correction)\nvs each period's curve low-charge gain a, which the correction uses for the images of that period")
        png = sorted(glob.glob(os.path.join(RUN_C, "gain_vs_charge_epochs", "*_gain_vs_charge.png")))
        for i_, pth in enumerate(png[:2]):
            ax = fig.add_axes([0.02 + 0.49 * i_, 0.42, 0.47, 0.46]); ax.axis("off"); ax.imshow(plt.imread(pth))
            ax.set_title(periods[i_][0] if i_ < len(periods) else "", fontsize=10, color=INK)
        ax = fig.add_axes([0.1, 0.1, 0.62, 0.28])
        for i, e in enumerate(exts):
            pts = [(j, *fitgain[(s, f"EXT{e}")]) for j, s in enumerate(stems) if (s, f"EXT{e}") in fitgain]
            if not pts:
                continue
            j, g, ge = map(np.array, zip(*pts))
            ax.errorbar(g, i + (j - n_img / 2) * 0.012, xerr=ge, fmt="o", ms=4, color=ORIG, ecolor=ORIG, alpha=0.8,
                        label="per-image fitted gain" if i == 0 else None)
            for ip, (plab, pm, pst) in enumerate(periods):
                r = pm.get(f"EXT{e}", {}); jj = [jx for jx, s in enumerate(stems) if s in pst]
                if r.get("ok") and jj:
                    y0, y1 = i + (min(jj) - n_img / 2) * 0.012, i + (max(jj) - n_img / 2) * 0.012
                    ax.plot([r["model"]["a"]] * 2, [y0 - 0.02, y1 + 0.02], color=CORR, lw=2.5,
                            label="curve's low-charge gain a (per period)" if i == 0 and ip == 0 else None)
        ax.set_yticks(range(len(exts))); ax.set_yticklabels([f"EXT{e}" for e in exts]); ax.grid(axis="y", visible=False)
        ax.set_xlabel("gain [ADU/e-]"); ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        book.save(fig)

        # ---------------------------------------------------------------- 3. full spectrum
        fig, axes = plt.subplots(len(exts), 1, figsize=(11, 8.5), sharex=True, squeeze=False)
        for ax, e in zip(axes[:, 0], exts):
            hist_pair(ax, e, np.arange(5, 66, 0.25), log=True)
            ax.set_ylim(bottom=0.8); mark_lines(ax, LINES)
            ax.set_ylabel("clusters / 0.25 keV"); ax.text(0.005, 0.92, f"EXT{e}", transform=ax.transAxes, weight="bold", color=INK)
        axes[-1, 0].set_xlabel("cluster energy [keV]")
        axes[0, 0].legend(fontsize=8, loc="upper right", frameon=False)
        page_title(fig, 3, f"Am-241 cluster-energy spectrum, all {n_img} images",
                   "dashed = true line energies (Np L x-rays, 26.3 and 59.5 keV gammas); log scale")
        fig.tight_layout(rect=[0, 0, 1, 0.91]); book.save(fig)

        # ---------------------------------------------------------------- 4. 59.5 keV
        ncol = 2; nrow = int(np.ceil((len(exts) + 1) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(11, 8.5), squeeze=False)
        for ax, e in zip(axes.flat, exts):
            hist_pair(ax, e, np.arange(50, 64, 0.2))
            ax.axvline(59.54, color=MUTED, ls="--", lw=0.8)
            t = []
            for k, lab in ((0, "without"), (1, "with   ")):
                r = fits[(e, G595, k)]
                t.append(f"{lab} {r[0]:6.2f} keV ({(r[0] / 59.54 - 1) * 100:+.1f}%), sigma {r[2]:.2f}" if r else f"{lab} no clear peak")
            ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
            ax.text(0.02, 0.97, "\n".join(t), transform=ax.transAxes, va="top", fontsize=8, family="monospace", color=INK,
                    bbox=dict(facecolor="white", edgecolor=GRID))
            nwin = np.sum((allE[e][0] > 53) & (allE[e][0] < 62))
            ax.set_title(f"EXT{e}" + ("   (low statistics: few 59.5 keV clusters on this CCD)" if nwin < 1000 else ""),
                         fontsize=11, color=INK, loc="left")
            ax.set_xlabel("cluster energy [keV]"); ax.set_ylabel("clusters / 0.2 keV")
        axes.flat[0].legend(fontsize=8, loc="center left", frameon=False)
        ax = axes.flat[len(exts)]
        for k, (col, lab) in enumerate(((ORIG, "without correction"), (CORR, "with correction"))):
            xs, ys, es = [], [], []
            for i, e in enumerate(exts):
                r = fits[(e, G595, k)]
                if r:
                    xs.append(i + (k - 0.5) * 0.18); ys.append(r[0] / 59.54); es.append(r[1] / 59.54)
            ax.errorbar(xs, ys, yerr=es, fmt="o", ms=8, color=col, capsize=3, label=lab)
        ax.axhline(1, color=MUTED, ls="--", lw=0.9)
        ax.set_xticks(range(len(exts))); ax.set_xticklabels([f"EXT{e}" for e in exts]); ax.set_xlim(-0.6, len(exts) - 0.4)
        ax.set_ylabel("59.5 keV peak / true"); ax.legend(fontsize=8, frameon=False)
        ax.set_title("59.5 keV peak position per CCD", fontsize=11, color=INK, loc="left")
        for ax in list(axes.flat)[len(exts) + 1:]:
            ax.axis("off")
        page_title(fig, 4, "The 59.54 keV gamma line", "Si escape peak 1.74 keV below it")
        fig.tight_layout(rect=[0, 0, 1, 0.91]); book.save(fig)

        # ---------------------------------------------------------------- 5. L lines + linearity
        fig, axes = plt.subplots(2, len(exts), figsize=(11, 8.5), squeeze=False)
        for j, e in enumerate(exts):
            ax = axes[0, j]
            hist_pair(ax, e, np.arange(10, 30.5, 0.4)); mark_lines(ax, LINES[:4])
            ax.set_title(f"EXT{e}", fontsize=11, color=INK, loc="left"); ax.set_xlabel("cluster energy [keV]")
            if j == 0:
                ax.set_ylabel("clusters / 0.4 keV"); ax.legend(fontsize=7.5, loc="upper right", frameon=False)
            ax = axes[1, j]
            for k, (col, lab) in enumerate(((ORIG, "without"), (CORR, "with"))):
                xs, ys, es = [], [], []
                for name, E0 in LINES:
                    r = fits[(e, name, k)]
                    if r:
                        xs.append(E0); ys.append(r[0] / E0); es.append(r[1] / E0)
                ax.errorbar(xs, ys, yerr=es, fmt="o-", color=col, ms=6, lw=1.5, capsize=3, label=f"{lab} correction")
            ax.axhline(1, color=MUTED, ls="--", lw=0.9); ax.set_xlim(10, 65)
            ax.set_xlabel("true line energy [keV]")
            if j == 0:
                ax.set_ylabel("measured / true"); ax.legend(fontsize=7.5, loc="lower right", frameon=False)
        lo = min([a.get_ylim()[0] for a in axes[1]]); hi = max([a.get_ylim()[1] for a in axes[1]])
        for a in axes[1]:
            a.set_ylim(lo, hi)
        page_title(fig, 5, "Np L x-rays, 26.3 keV, and the energy scale across 14-60 keV",
                   "top: 10-30 keV spectra; bottom: fitted position / true energy for every clear line (same y-scale)")
        fig.tight_layout(rect=[0, 0, 1, 0.91]); book.save(fig)

        # ---------------------------------------------------------------- 6. per image
        fig, axes = plt.subplots(len(exts), 1, figsize=(11, 8.5), sharex=True, squeeze=False)
        labels = [f"{date(s).group(1)[4:6]}/{date(s).group(1)[6:]} #{date(s).group(2)}" if date(s) else s for s in stems]
        for ax, e in zip(axes[:, 0], exts):
            stats = []
            for k, (col, lab) in enumerate(((ORIG, "without correction"), (CORR, "with correction"))):
                xs, ys, es = [], [], []
                for j, s in enumerate(stems):
                    r = per_img[(e, s, k)]
                    if r:
                        xs.append(j + (k - 0.5) * 0.2); ys.append(r[0] / 59.54); es.append(r[1] / 59.54)
                ax.errorbar(xs, ys, yerr=es, fmt="o", ms=5, color=col, capsize=2, label=lab)
                if len(ys) > 1:
                    stats.append(f"{lab.split()[0]}: mean {np.mean(ys):.3f}, rms {np.std(ys) * 100:.2f}% ({len(ys)} images)")
            ax.axhline(1, color=MUTED, ls="--", lw=0.9)
            if not stats:
                stats = [f"no clear 59.5 keV peak in single images (~{np.median([np.sum((v > 53) & (v < 62)) for v in orig[e].values()]):.0f}"
                         " clusters per image in 53-62 keV)"]
                ax.set_ylim(0.9, 1.06)
            ax.set_title(f"EXT{e}    " + "    |    ".join(stats), fontsize=9, color=INK, loc="left")
            ax.set_ylabel("59.5 keV / true")
        axes[-1, 0].set_xticks(range(n_img)); axes[-1, 0].set_xticklabels(labels, rotation=60, fontsize=7.5)
        for plab, pm, pst in periods[1:]:                 # where each later gain period starts
            x0 = min(j for j, s in enumerate(stems) if s in pst) - 0.5
            for ax in axes[:, 0]:
                ax.axvline(x0, color=MUTED, lw=1, ls=":")
            axes[0, 0].text(x0, axes[0, 0].get_ylim()[1], f" {plab}: gain period 2", fontsize=8, color=MUTED, va="top")
        axes[0, 0].legend(fontsize=8, loc="lower right", frameon=False)
        page_title(fig, 6, "59.5 keV peak image by image",
                   "fit per image (a missing point = no clean single peak in that image); x = image date (MM/DD) and number")
        fig.tight_layout(rect=[0, 0, 1, 0.91]); book.save(fig)

        # ---------------------------------------------------------------- 7. MakeComposite Am-241 plots
        fig = plt.figure(figsize=(11, 8.5))
        page_title(fig, 7, "MakeComposite's Am-241 spectra (compare_am241_energy_spectrum_2x2.png)",
                   "the macro's own window (10-30 keV) and binning; EXT2 empty (not run: flooded)")
        for i, (run, lab) in enumerate(((RUN_O, "without correction (root/)"), (RUN_C, "with correction (per period, root_gaincorr_epochs/)"))):
            p = glob.glob(os.path.join(run, "composite" if run == RUN_O else "composite_epochs", "Fe_55_Analysis_Plots", "*", "*", "compare",
                                       "compare_am241_energy_spectrum_2x2.png"))
            fig.text(0.05 + 0.48 * i, 0.87, lab, fontsize=11, weight="bold", color=ORIG if i == 0 else CORR)
            ax = fig.add_axes([0.01 + 0.49 * i, 0.03, 0.48, 0.82]); ax.axis("off")
            if p:
                img = plt.imread(p[0]); ax.imshow(img); ax.set_aspect("auto")
            else:
                ax.text(0.5, 0.5, "not found", ha="center", color=MUTED)
        book.save(fig)

        # ---------------------------------------------------------------- 8. width cut: motivation and definition
        by_stats = sorted(exts, key=lambda e: -len(allE[e][1]))
        best, shown = by_stats[0], by_stats[:2]
        E, S = allE[best][1], allS[best][1]
        sbins = [0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.40, 1.70]
        pos = {e: [(0.5 * (lo + hi), fit_line(allE[e][1][(allS[e][1] >= lo) & (allS[e][1] < hi)], 59.54, min_counts=40))
                   for lo, hi in zip(sbins[:-1], sbins[1:])] for e in shown}
        ok = [(x, r[0]) for x, r in pos[best] if r]
        shift = (f"on EXT{best} the 59.5 keV peak moves from {ok[0][1]:.1f} keV at sigma_xy ~ {ok[0][0]:.2f} to "
                 f"{ok[-1][1]:.1f} keV at sigma_xy ~ {ok[-1][0]:.2f} (right; much less on "
                 f"{', '.join('EXT' + str(e) for e in shown[1:]) or 'the other CCDs'})") if len(ok) > 1 else "see right"
        keep595 = ", ".join(
            f"{np.sum(w & (allS[e][1] < SXY_CUT)) / max(np.sum(w), 1) * 100:.0f}% on EXT{e}"
            for e in shown for w in [(allE[e][1] > 59.0) & (allE[e][1] < 60.2)])
        fig = plt.figure(figsize=(11, 8.5))
        page_title(fig, 8,"A cut on the cluster width against the background",
                   "sigma_xy = sqrt((sigma_x^2 + sigma_y^2) / 2) [pix], as in MakeComposite's front/back split; "
                   "gain-corrected clusters, all images")
        paras = [
            ("Why", "The small lines (Np L x-rays at 13.9-21 keV, the 26.3 keV gamma) sit on a continuum: Compton scatters of "
                    "the 59.5 keV gamma and other background, plus 59.5 keV hits that lost part of their charge. Where a photon "
                    "converts sets how far its charge diffuses before it is collected, i.e. the cluster width. Low-energy x-rays "
                    "convert close to the surface they enter (attenuation length in Si ~0.5 mm at 15 keV, ~1 mm at 20 keV), while "
                    "the 59.5 keV gamma converts almost uniformly in depth (~13 mm). The data show the L lines as stripes at small "
                    "width (left) and the continuum spread over all widths (middle)."),
            ("Wide clusters lose charge", f"The faint edge of a diffuse cluster falls below the pixel threshold: {shift}. Those "
                    "clusters make the low-energy tail of the 59.5 keV peak and part of the continuum under the L lines."),
            ("The cut", f"Keep clusters with sigma_xy < {SXY_CUT} pixels. This keeps the very compact clusters (sigma_xy < 0.33, "
                    "2-4 pixels): they hold the cleanest L lines even though they have no 59.5 keV peak, since few 59.5 keV gammas "
                    "convert that close to the surface. Tried and not used: distance to the nearest cluster, PCA elongation and "
                    "linearity, position on the CCD - line and background clusters look the same in those."),
            ("Caveats", "Compact clusters put ~3000 e- in their brightest pixel, above the range where g(n) was measured "
                    "(1100-1500 e-), so their energies rely on the extrapolated part of the correction. The cut keeps "
                    f"{keep595} of the clusters in the 59.5 keV peak window (59.0-60.2 keV)."),
        ]
        y = 0.885
        for head, body in paras:
            fig.text(0.05, y, head, fontsize=10.5, weight="bold", color=INK); y -= 0.024
            for line in textwrap.wrap(body, 128):
                fig.text(0.07, y, line, fontsize=8.8, color=INK); y -= 0.02
            y -= 0.008
        top = y - 0.05
        # (a) width vs energy
        ax = fig.add_axes([0.06, 0.07, 0.26, top - 0.07])
        m = (E > 8) & (E < 64)
        ax.hist2d(E[m], S[m], bins=[np.arange(8, 64, 0.5), np.linspace(0, 2, 60)], norm=matplotlib.colors.LogNorm(), cmap="Blues")
        ax.axhspan(0, SXY_CUT, color=CUTC, alpha=0.12, lw=0); ax.axhline(SXY_CUT, color=CUTC, lw=1.5)
        ax.text(9, SXY_CUT + 0.03, f"kept: sigma_xy < {SXY_CUT}", color=INK, fontsize=8)
        ax.set_xlabel("cluster energy [keV]"); ax.set_ylabel("sigma_xy [pix]"); ax.grid(False)
        ax.set_title(f"EXT{best}: width vs energy", loc="left", fontsize=10, color=INK)
        # (b) width of clusters in the line windows vs the sidebands
        ax = fig.add_axes([0.40, 0.07, 0.26, top - 0.07])
        inwin = np.zeros_like(E, bool); inside = np.zeros_like(E, bool)
        for _, win, side in PB:
            inwin |= (E > win[0]) & (E < win[1]); inside |= (E > side[0]) & (E < side[1])
        for sel, col, lab, ls in ((inwin, CUTC, "in the L / 26.3 keV line windows", "-"), (inside, ALLC, "in the sidebands (continuum)", "-"),
                                  ((E > 58.5) & (E < 60.5), INK, "59.5 keV peak", "--")):
            ax.hist(S[sel], bins=np.linspace(0, 2, 41), density=True, histtype="step", color=col, lw=1.8, ls=ls, label=lab)
        ax.axvline(SXY_CUT, color=CUTC, lw=1.2, ls=":")
        ax.set_xlabel("sigma_xy [pix]"); ax.set_ylabel("fraction of clusters (normalised)")
        ax.set_title(f"EXT{best}: width of line vs continuum clusters", loc="left", fontsize=10, color=INK)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
        # (c) 59.5 keV peak position vs width
        ax = fig.add_axes([0.73, 0.07, 0.25, top - 0.07])
        for e, mk in zip(shown, ("o", "s")):
            pts = [(x, r[0] / 59.54, r[1] / 59.54) for x, r in pos[e] if r]
            if pts:
                x, yv, ye = map(np.array, zip(*pts))
                ax.errorbar(x, yv, yerr=ye, fmt=mk + "-", color=INK, mfc=INK if mk == "o" else "white", ms=7, lw=1.2,
                            capsize=2, label=f"EXT{e}")
        ax.axhline(1, color=MUTED, ls="--", lw=0.9); ax.axvline(SXY_CUT, color=CUTC, lw=1.2, ls=":")
        ax.set_xlabel("sigma_xy [pix]"); ax.set_ylabel("59.5 keV peak / true")
        ax.set_title("59.5 keV peak vs cluster width", loc="left", fontsize=10, color=INK); ax.legend(fontsize=8, frameon=False)
        book.save(fig)

        # ---------------------------------------------------------------- 9. width cut: results
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        for ax, e in zip(axes[0], shown):
            bins = np.arange(8, 32, 0.25)
            ax.hist(cutE[e][0], bins=bins, histtype="step", color=ORIG, lw=1.4, label="without correction")
            ax.hist(cutE[e][1], bins=bins, histtype="step", color=CORR, lw=1.4, label="with correction")
            mark_lines(ax, LINES[:4])
            ax.set_title(f"EXT{e}, sigma_xy < {SXY_CUT}  ({len(cutE[e][1])} of {len(allE[e][1])} clusters)", loc="left",
                         fontsize=10, color=INK)
            ax.set_xlabel("cluster energy [keV]"); ax.set_ylabel("clusters / 0.25 keV")
        axes[0, 0].legend(fontsize=8, frameon=False, loc="center right")
        # (c) peak / background before and after the cut
        ax = axes[1, 0]
        groups, vals = [], []
        for e in shown:
            for name, win, side in PB:
                r = []
                for E_ in (allE[e][1], cutE[e][1]):
                    pk, bg = peak_bg(E_, win, side); r.append(pk / max(bg, 1))
                groups.append(f"EXT{e}\n{name}"); vals.append(r)
        x = np.arange(len(groups)); w = 0.38
        for k, (col, lab) in enumerate(((ALLC, "all clusters"), (CUTC, f"sigma_xy < {SXY_CUT}"))):
            v = [r[k] for r in vals]
            bars = ax.bar(x + (k - 0.5) * (w + 0.02), v, width=w, color=col, label=lab)
            for b, val in zip(bars, v):
                ax.text(b.get_x() + b.get_width() / 2, val + 0.08, f"{val:.1f}", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8); ax.grid(axis="x", visible=False)
        ax.set_ylabel("peak / sideband counts (1 keV windows)")
        ax.set_title("peak-to-background, gain-corrected", loc="left", fontsize=10, color=INK)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        # (d) energy scale with the cut
        ax = axes[1, 1]
        for e, mk in zip(shown, ("o", "s")):
            for k, col in ((0, ORIG), (1, CORR)):
                pts = [(E0, cfits[(e, n, k)][0] / E0, cfits[(e, n, k)][1] / E0) for n, E0 in LINES if cfits[(e, n, k)]]
                if pts:
                    xs, ys, es = map(np.array, zip(*pts))
                    ax.errorbar(xs, ys, yerr=es, fmt=mk + "-", color=col, mfc=col if mk == "o" else "white", ms=7, lw=1.2,
                                capsize=3, label=f"EXT{e} {'with' if k else 'without'} correction")
        ax.axhline(1, color=MUTED, ls="--", lw=0.9); ax.set_xlim(10, 65)
        ax.set_xlabel("true line energy [keV]"); ax.set_ylabel("measured / true")
        ax.set_title(f"energy scale from the lines that fit cleanly after the cut", loc="left", fontsize=10, color=INK)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
        page_title(fig, 9, f"The low-energy lines after the width cut (sigma_xy < {SXY_CUT})",
                   "top: spectra without / with the gain correction;  bottom: peak over neighbouring continuum, and fitted "
                   "line position / true energy")
        fig.tight_layout(rect=[0, 0, 1, 0.91]); book.save(fig)

    print("->", OUT)
    for e in exts:
        print(f"EXT{e}: " + "  ".join(f"{clean(n)}: {fmt_ratio(fits[(e, n, 0)], E0)}->{fmt_ratio(fits[(e, n, 1)], E0)}"
                                      for n, E0 in LINES))
        for k in (0, 1):
            r = fits[(e, G595, k)]
            print(f"   59.5 sigma {'with' if k else 'without'}: {r[2]:.3f}" if r else "   -")
    print(f"width cut sigma_xy < {SXY_CUT}:")
    for e in exts:
        print(f"EXT{e}: " + "  ".join(f"{clean(n)}: {fmt_ratio(cfits[(e, n, 0)], E0)}->{fmt_ratio(cfits[(e, n, 1)], E0)}"
                                      for n, E0 in LINES))


if __name__ == "__main__":
    main()
