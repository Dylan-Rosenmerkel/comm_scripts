"""Multi-page PDF: what the gain-vs-charge correction does and how Fe-55 / cluster energies change.

Purpose: an 11-page report for collaborators on the gain-vs-charge correction (run_WADERs
--gain-correction / gain_vs_charge.py) over the Production_Modules_Data: the measured g(n)
curves, Fe-55 K-alpha spectra of single CCDs and whole modules without / with the correction,
binned images, every dataset at a glance, and the limitations.

Inputs (defaults in brackets, see -h):
    --universal  folder of the "universal" test: universal_results.json (per-CCD K-alpha fits
                 and curve summaries) and WADERs_analysis_<tag>/{root,root_gaincorr,gain_vs_charge}/
                 [REQUIRED: it was a temporary folder of an earlier session and may no longer exist]
    --dm2426     WADERs analysis of the DM24/DM25/DM26 Image 5 set (pages 6-8)
                 [.../2025-09-13-DM24_DM25_DM26_Low_Temp/WADERs_analysis_Image_5_files]
Outputs:
    --out        the PDF                                     [comm_scripts/docs/gain_correction_report.pdf]
    --pages-dir  every page also as page_NN.png (70 dpi)     [pages/ next to this script]

Dataset tags look like I5_DM07-28_0919: image type (Image 5), module folder, date (MMDD).

Run:  python make_gain_correction_report.py --universal FOLDER     (needs PyROOT, matplotlib)

How to read this file
    module_of, pretty     tag/board -> module name; tag -> readable label
    energies              Fe-55 cluster energies of one board/extension (front-side clusters by default)
    text_page             a page of headed text blocks
    spectrum_panel        one K-alpha/K-beta spectrum panel without / with the correction
    row                   the universal_results.json entry of one CCD
    _Both                 writes each page to the PDF and as a PNG
    page_intro ... page_limits   one function per page (group), in page order
    main                  options, load the results, write the pages
"""
import argparse, glob, json, os, re, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import ROOT
ROOT.gROOT.SetBatch(True)

U = "/private/tmp/claude-501/-Users-dylanrosenmerkel/a98994a6-5dc6-442a-abe8-a65a49102c77/scratchpad/universal"
DM2426 = ("/Users/dylanrosenmerkel/Documents/LSM_Module_Testing_Analysis/Production_Modules_Data/"
          "2025-09-13-DM24_DM25_DM26_Low_Temp/WADERs_analysis_Image_5_files")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gain_correction_report.pdf")  # docs/
KA, KB = 5.899, 6.490                     # Mn K-alpha / K-beta (Fe-55) [keV]
ORIG, CORR = "#8c8c8c", "#1f5fbf"         # colours: without / with the correction
NAMES = {"110": 0, "109": 1, "106": 2}    # board -> position of its module in a FOLDERS entry
FOLDERS = {"DM04-06": "DM04/DM05/DM06", "DM07-28": "DM07/DM27/DM28", "DM10-25": "DM10/DM24/DM25",
           "DM11-14": "DM11/DM13/DM14", "DM15-17": "DM15/DM16/DM17", "PD06-08": "PD06/PD07/PD08",
           "DM01-03": "DM01/DM02/DM03"}


def module_of(tag, board):
    """Module name (e.g. 'DM27') read out by board ('110', '109' or '106') in dataset tag.

    Uses the module-folder part of the tag (FOLDERS) and the board order of NAMES; the DM24-26
    set has its own mapping; otherwise returns 'board <board>'.
    """
    for k, v in FOLDERS.items():
        if k in tag:
            return v.split("/")[NAMES[board]]
    if "DM2426" in tag:
        return {"110": "DM24", "109": "DM25", "106": "DM26"}[board]
    return f"board {board}"


def pretty(tag):
    """Readable label of a dataset tag: 'I5_DM07-28_0919' -> 'Image 5, DM07/DM27/DM28, 09/19' (else the tag)."""
    m = re.match(r"I(\d)_(.*)_(\d{4})$", tag)
    if not m:
        return tag
    typ, mods, date = m.groups()
    mods = FOLDERS.get(mods, mods)
    return f"Image {typ}, {mods}, {date[:2]}/{date[2:]}"


def energies(d, board, ext, front=True):
    """Cluster energies [keV] of board/extension ext from the cluster files in folder d.

    front=True keeps only compact clusters, sqrt((sx^2 + sy^2)/2) < 1 pixel: X-rays that convert
    near the front surface diffuse little, so their charge (and energy) is collected best.
    Returns a float array (empty if no file or no cluster).
    """
    E = []
    for f in glob.glob(f"{d}/panaSKImg_clustersRec_*_{board}_*_ext{ext}.root"):
        tf = ROOT.TFile.Open(f)
        t = tf.Get("clustersRec")
        if isinstance(t, ROOT.TTree):
            for i in range(t.GetEntries()):
                t.GetEntry(i)
                e = np.array(list(t.Energy))
                if front:
                    sx, sy = np.array(list(t.fSTD_X)), np.array(list(t.fSTD_Y))
                    e = e[np.sqrt(0.5 * (sx ** 2 + sy ** 2)) < 1]
                E.append(e)
        tf.Close()
    return np.concatenate(E) if E else np.array([])


def text_page(pdf, title, blocks):
    """Add a text-only page: title, then blocks of (heading or '', [paragraphs]), wrapped at 118 characters."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.06, 0.93, title, fontsize=20, weight="bold")
    y = 0.86
    for head, body in blocks:
        if head:
            fig.text(0.06, y, head, fontsize=13, weight="bold", color="#1f3f7f")
            y -= 0.035
        for para in body:
            for line in textwrap.wrap(para, 118):
                fig.text(0.08, y, line, fontsize=10.5)
                y -= 0.026
            y -= 0.008
        y -= 0.012
    pdf.savefig(fig)
    plt.close(fig)


def spectrum_panel(ax, Eo, Ec, title, ka_o=None, ka_c=None, corrected=True):
    """Draw the 4.6-7.4 keV spectrum without (Eo) and with (Ec) the correction on ax (40 eV bins).

    ka_o / ka_c: fitted K-alpha peak [keV] without / with (from universal_results.json), shown
    with their offset from 5.899 keV; corrected=False adds that the CCD was left unchanged.
    """
    bins = np.arange(4.6, 7.4, 0.04)
    ax.hist(Eo, bins=bins, histtype="step", color=ORIG, lw=1.6, label="without correction")
    ax.hist(Ec, bins=bins, histtype="step", color=CORR, lw=1.6, label="with correction")
    ax.axvline(KA, color="k", ls="--", lw=0.9, label="true K$\\alpha$ 5.899 keV")
    ax.axvline(KB, color="k", ls=":", lw=0.9, label="true K$\\beta$ 6.490 keV")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)          # room for the text box
    lines = []
    if ka_o is not None:
        lines.append(f"K$\\alpha$ peak without: {ka_o:.3f} keV ({(ka_o / KA - 1) * 100:+.1f}%)")
    if ka_c is not None:
        lines.append(f"K$\\alpha$ peak with:     {ka_c:.3f} keV ({(ka_c / KA - 1) * 100:+.1f}%)")
    if not corrected:
        lines.append("(no reliable curve: left unchanged)")
    ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=8.5, va="top", family="monospace",
            bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.9))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("cluster energy [keV]")
    ax.set_ylabel("clusters per 0.04 keV")


def row(rows, tag, board, ext):
    """The universal_results.json row of one CCD (dataset tag, board, 'EXT<n>'), or None.

    A row holds 'orig' / 'corr' (K-alpha fit without / with, [peak keV, ...] or empty),
    'corrected' (whether a reliable curve was applied) and 'typ' ('I5', 'I6', ...).
    """
    return next((r for r in rows if r["tag"] == tag and r["board"] == board and r["ext"] == ext), None)


class _Both:
    """PdfPages that also writes each page as <pages_dir>/page_NN.png (for checking)."""
    def __init__(self, pp, pages_dir):
        self.pp, self.n, self.pages_dir = pp, 0, pages_dir
        os.makedirs(pages_dir, exist_ok=True)
    def savefig(self, fig):
        """Add fig as the next PDF page and write it as page_NN.png (70 dpi)."""
        self.n += 1
        self.pp.savefig(fig)
        fig.savefig(os.path.join(self.pages_dir, f"page_{self.n:02d}.png"), dpi=70)


def page_intro(pdf, rows):
    """Page 1: the problem, what the correction does, how to use it, and the Image-5 K-alpha
    summary (mean and CCD-to-CCD spread of peak / 5.899 keV, without -> with)."""
    # ------------------------------------------------------------------ page 1
    i5 = [r for r in rows if r["typ"] == "I5" and r["orig"] and r["corr"] and r["corrected"]]
    o = np.array([r["orig"][0] / KA for r in i5]); c = np.array([r["corr"][0] / KA for r in i5])
    text_page(pdf, "Gain-vs-charge correction for cluster energies", [
        ("What the problem is", [
            "The gain (ADU per electron) is normally measured from the spacing of the 0e- and 1e- peaks and then "
            "used for every pixel. But the amplifier gain falls slowly as the charge in a pixel grows: typically "
            "~3% lower at 1000 e- (0.2-8% across 111 Image-5 CCDs). X-ray clusters have pixels with hundreds to "
            "thousands of electrons, so their energies come out a few percent too low. On top of that, the 0e-/1e- "
            "gain fit of each image can be off by 1-4%, so different CCDs disagree."]),
        ("What the correction does", [
            "1. Measure g(n), the gain at n electrons per pixel, from the images themselves: a skipper image's pixel "
            "values form a comb of peaks, one per electron count; the local spacing of that comb at 50, 300, 800 e- ... "
            "is the gain at that charge (measured in clean rows, all images of the same module/CCD together). "
            "Fit: g(n) = a + b*n, extrapolated up to 3x the highest measured charge, constant beyond.",
            "2. Convert every cluster pixel back to ADU, then to electrons with that curve (instead of one gain), "
            "and re-sum the cluster energy. Written as a second set of cluster files (root_gaincorr/); the originals "
            "are kept.",
            "3. CCDs whose curve cannot be trusted (comb visible over < 4 charge bins or < 400 e-, or a rising gain) "
            "are left unchanged. No source peak is used anywhere."]),
        ("How to use it", [
            "run_WADERs images.txt --gain-correction      (or the standalone tool: gain_vs_charge.py measure / apply)"]),
        ("Result over the Production_Modules_Data (tool frozen, nothing tuned on these data)", [
            f"Image 5, Fe-55 K-alpha of the {len(i5)} corrected CCDs in 10 datasets: average {o.mean():.4f} -> "
            f"{c.mean():.4f} of the true energy; spread between CCDs {o.std(ddof=1) * 100:.1f}% -> {c.std(ddof=1) * 100:.2f}%.",
            "Binned Images 6/7: average ~0.98-0.99 -> 0.998-1.002, spread roughly halved.",
            "Limits: needs multi-skip images with visible electron peaks (1-skip images are skipped); ~20% of CCDs get "
            "no reliable curve and stay as they were; windowed (no-overscan) images improve less; no correction for "
            "the Am-241 run (no resolved peaks)."]),
        ("Pages", ["2: measured g(n) curves.  3-5: Fe-55 spectra of single CCDs with/without.  6-8: all CCDs of a "
                   "module overlaid (alignment).  9: binned image.  10: every dataset, per CCD.  11: limitations."]),
    ])


def page_curves(pdf, u, res):
    """Page 2: measured g(n) / low-charge gain with the linear fit for 3 modules, and the
    distribution of the gain drop at 1000 e- over all reliable Image-5 curves."""
    # ------------------------------------------------------------------ page 2: g(n) curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    picks = [("I5_DM07-28_0919", "#1b9e77"), ("I5_PD06-08_1108", "#d95f02"), ("I5_DM15-17_0903", "#7570b3")]
    for tag, col in picks:
        for j in sorted(glob.glob(f"{u}/WADERs_analysis_{tag}/gain_vs_charge/module_*_gain_vs_charge.json"))[:1]:
            board = re.search(r"module_(\d+)_", j).group(1)
            for k, (ext, r) in enumerate(sorted(json.load(open(j))["extensions"].items())):
                if not r.get("ok"):
                    continue
                pts = [p for p in r["points"] if p["significant"]]
                n = np.array([p["n"] for p in pts]); g = np.array([p["g"] for p in pts]); m = r["model"]
                grid = np.geomspace(20, 3000, 200)
                lab = f"{module_of(tag, board)} {ext}" if k < 4 else None
                axes[0].plot(n, g / m["a"], "o", color=col, ms=4, alpha=0.8, label=lab)
                axes[0].plot(grid, (m["a"] + m["b"] * np.minimum(grid, m["n_cap"])) / m["a"], "-", color=col, alpha=0.5)
    axes[0].set_xscale("log"); axes[0].set_xlabel("electrons in the pixel"); axes[0].set_ylabel("gain / low-charge gain")
    axes[0].set_title("Measured gain vs charge (points) and fit (lines)\n3 modules x 4 CCDs, Image 5", fontsize=11)
    axes[0].axhline(1, color="k", lw=0.5); axes[0].legend(fontsize=7, ncol=2); axes[0].grid(alpha=0.3)
    drops = [cc["drop"] * 100 for cc in res["curves"] if cc["ok"] and cc["tag"].startswith("I5")]
    axes[1].hist(drops, bins=np.arange(0, 9, 0.5), color=CORR, alpha=0.8)
    axes[1].set_xlabel("gain drop at 1000 e- [%]"); axes[1].set_ylabel("number of CCDs")
    axes[1].set_title(f"How much lower the gain is at 1000 e-\n{len(drops)} CCDs, 10 Image-5 datasets "
                      f"(median {np.median(drops):.1f}%)", fontsize=11)
    fig.suptitle("Page 2 - the effect being corrected: the gain falls with the charge in a pixel", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf.savefig(fig); plt.close(fig)


def pages_single_ccd(pdf, u, rows):
    """Pages 3-5: Fe-55 spectra (front-side clusters) of 12 single CCDs without / with the correction."""
    # ------------------------------------------------------------------ pages 3-5: single CCD spectra
    examples = [
        [("I5_DM07-28_0919", "106", "EXT3"), ("I5_DM07-28_0919", "110", "EXT1"),
         ("I5_PD06-08_1108", "109", "EXT2"), ("I5_PD06-08_1108", "106", "EXT4")],
        [("I5_DM11-14_0830", "110", "EXT2"), ("I5_DM15-17_0903", "106", "EXT1"),
         ("I5_DM04-06_1031", "109", "EXT3"), ("I5_DM10-25_1104", "110", "EXT1")],
        [("I5_DM01-03_0218", "110", "EXT2"), ("I5_DM01-03_0218", "109", "EXT4"),
         ("I5_DM01-03_1113", "110", "EXT1"), ("I5_DM01-03_1113", "109", "EXT2")],
    ]
    notes = ["Typical CCDs: the K-alpha peak moves from ~2% low to the right energy and gets slightly sharper.",
             "More modules and dates: same behaviour.",
             "Top: CCDs that were badly off - the uncorrected peak is shifted or split in two because the per-image "
             "fitted gain was wrong or differed between images - come back to one peak at 5.90 keV. "
             "Bottom: CCDs without a reliable curve are left unchanged (grey and blue lie on top of each other)."]
    for p, (page, note) in enumerate(zip(examples, notes)):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        for ax, (tag, b, e) in zip(axes.flat, page):
            d = f"{u}/WADERs_analysis_{tag}"
            r = row(rows, tag, b, e)
            spectrum_panel(ax, energies(f"{d}/root", b, int(e[3:])), energies(f"{d}/root_gaincorr", b, int(e[3:])),
                           f"{pretty(tag)}  -  {module_of(tag, b)} {e}",
                           r["orig"][0] if r and r["orig"] else None, r["corr"][0] if r and r["corr"] else None,
                           corrected=bool(r and r["corrected"]))
        axes[0, 0].legend(fontsize=8, loc="upper right")
        fig.suptitle(f"Page {3 + p} - Fe-55 spectrum of single CCDs (front-side clusters)\n" + "\n".join(textwrap.wrap(note, 110)),
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.92]); pdf.savefig(fig); plt.close(fig)


def pages_alignment(pdf, u, dm2426):
    """Pages 6-8: K-alpha region of all 12 CCDs of three module sets overlaid, each scaled to its
    maximum; the spread of the peak bins of the corrected CCDs shows how well they agree.
    CCDs without a reliable curve (from gain_vs_charge/module_*_gain_vs_charge.json) are dashed grey."""
    # ------------------------------------------------------------------ pages 6-8: module alignment
    align = [(f"{dm2426}", "Image 5, DM24/DM25/DM26, 09/13 (the first test set, 18 images)", "DM2426"),
             (f"{u}/WADERs_analysis_I5_DM07-28_0919", "Image 5, DM07/DM27/DM28, 09/19", "I5_DM07-28_0919"),
             (f"{u}/WADERs_analysis_I5_PD06-08_1108", "Image 5, PD06/PD07/PD08, 11/08", "I5_PD06-08_1108")]
    for p, (d, title, tag) in enumerate(align):
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), sharey=True)
        bins = np.arange(5.3, 6.8, 0.03)
        peaks = {0: [], 1: []}
        okmap = {}
        for j in glob.glob(f"{d}/gain_vs_charge/module_*_gain_vs_charge.json"):
            bb = re.search(r"module_(\d+)_", j).group(1)
            for ext, rr in json.load(open(j))["extensions"].items():
                okmap[(bb, int(ext[3:]))] = bool(rr.get("ok"))
        for b in ("110", "109", "106"):
            for e in (1, 2, 3, 4):
                ok_ccd = okmap.get((b, e), False)
                for k, sub in enumerate(("root", "root_gaincorr")):
                    E = energies(f"{d}/{sub}", b, e)
                    E = E[(E > 5.3) & (E < 6.8)]
                    if E.size < 100:
                        continue
                    h, _ = np.histogram(E, bins=bins)
                    h = h / h.max()
                    if ok_ccd:
                        axes[k].plot(0.5 * (bins[1:] + bins[:-1]), h, "-", lw=1.1, alpha=0.75)
                        peaks[k].append(0.5 * (bins[np.argmax(h)] + bins[np.argmax(h) + 1]))
                    else:
                        axes[k].plot(0.5 * (bins[1:] + bins[:-1]), h, "--", color="0.55", lw=1.0)
        for k, lab in enumerate(("without correction", "with correction")):
            axes[k].axvline(KA, color="k", ls="--", lw=1)
            pk = np.array(peaks[k])
            axes[k].set_title(f"{lab}\npeak of each corrected CCD: {pk.min():.2f}-{pk.max():.2f} keV "
                              f"(spread {pk.std(ddof=1):.3f} keV)", fontsize=11)
            axes[k].set_xlabel("cluster energy [keV]")
        axes[0].set_ylabel("clusters (each CCD scaled to its peak)")
        fig.suptitle(f"Page {6 + p} - all 12 CCDs of one module set, K-alpha region (front-side clusters)\n{title}: "
                     f"one line per CCD; black dashed = 5.899 keV;\n"
                     f"grey dashed = CCD without a reliable curve (left unchanged, not in the peak range)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.92]); pdf.savefig(fig); plt.close(fig)


def page_binned(pdf, u, rows):
    """Page 9: binned images (Image 6/7, all clusters), where pixels hold more charge."""
    # ------------------------------------------------------------------ page 9: binned image
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, (tag, b, e) in zip(axes.flat, [("I6_DM07-28_0919", "106", "EXT2"), ("I6_DM07-28_0919", "110", "EXT4"),
                                           ("I6_DM15-17_0903", "109", "EXT1"), ("I7_DM07-28_0919", "109", "EXT3")]):
        d = f"{u}/WADERs_analysis_{tag}"; r = row(rows, tag, b, e)
        spectrum_panel(ax, energies(f"{d}/root", b, int(e[3:]), front=False),
                       energies(f"{d}/root_gaincorr", b, int(e[3:]), front=False),
                       f"{pretty(tag)}  -  {module_of(tag, b)} {e}",
                       r["orig"][0] if r and r["orig"] else None, r["corr"][0] if r and r["corr"] else None,
                       corrected=bool(r and r["corrected"]))
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Page 9 - binned images (Image 6: 10 rows summed, Image 7: 10 columns summed; all clusters)\n"
                 "Pixels hold more charge here, so the correction is larger", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92]); pdf.savefig(fig); plt.close(fig)


def page_all_datasets(pdf, rows):
    """Page 10: K-alpha peak / 5.899 keV of every CCD of every Image 5/6/7 dataset, without and with."""
    # ------------------------------------------------------------------ page 10: every dataset
    fig, ax = plt.subplots(figsize=(11, 8.5))
    tags = [t for t in sorted({r["tag"] for r in rows}) if t.split("_")[0] in ("I5", "I6", "I7")
            and any(r["tag"] == t and r["orig"] and r["corr"] for r in rows)]
    for i, t in enumerate(tags):
        for r in rows:
            if r["tag"] != t or not (r["orig"] and r["corr"]):
                continue
            o_, c_ = r["orig"][0] / KA, r["corr"][0] / KA
            mk = "o" if r["corrected"] else "x"
            ax.plot([i - 0.15], [o_], mk, color=ORIG, ms=6, alpha=0.8)
            ax.plot([i + 0.15], [c_], mk, color=CORR, ms=6, alpha=0.8)
            ax.plot([i - 0.15, i + 0.15], [o_, c_], "-", color="0.8", lw=0.6)
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(tags))); ax.set_xticklabels([pretty(t) for t in tags], rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Fe-55 K-alpha peak / 5.899 keV"); ax.set_ylim(0.9, 1.1)
    ax.plot([], [], "o", color=ORIG, label="without correction"); ax.plot([], [], "o", color=CORR, label="with correction")
    ax.plot([], [], "x", color="0.4", label="CCD without a reliable curve (left unchanged)")
    ax.legend(loc="upper left", fontsize=9); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Page 10 - every CCD of every dataset with a usable Fe-55 peak: without (grey) / with (blue)\n"
                 "Corrected CCDs (circles) move to 1.00 and close up; unchanged CCDs (x) keep their spread", fontsize=12)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def page_limits(pdf, res):
    """Page 11: when the correction cannot help (rejected curves, unsupported images, what remains)."""
    # ------------------------------------------------------------------ page 11: limits
    nok = sum(1 for cc in res["curves"] if cc["ok"]); ntot = len(res["curves"])
    text_page(pdf, "When the correction cannot help", [
        ("CCDs without a reliable curve (~20%)", [
            f"{ntot - nok} of {ntot} CCD curves in this test were rejected: the electron comb faded after 1-3 charge "
            "bins (noisy or charged CCDs), or the measured gain rose with charge. These CCDs are copied unchanged, "
            "so a dataset where many are rejected (e.g. DM01-03, 11/13: 10 of 12) barely improves."]),
        ("Images it does not apply to", [
            "1-skip images (no single-electron peaks), CCDs with no resolved peaks (the Am-241 run of 03/28 here), "
            "and July 2025 tests with almost no Fe-55 signal (nothing to check against)."]),
        ("Windowed readouts (no overscan, July-August 2025)", [
            "Supported since this test, but the improvement is smaller (spread 0.047 -> 0.037 and 0.059 -> 0.050 keV) "
            "and a few CCDs get worse; treat these with care."]),
        ("What remains", [
            "Corrected K-alpha sits ~0.2-0.7% below 5.899 keV: charge in the faint edge of each cluster that the "
            "clustering threshold drops. A gain correction cannot recover it. Back-side (diffuse) clusters lose more."]),
    ])


def main():
    """Options, load universal_results.json, write the 11 pages."""
    ap = argparse.ArgumentParser(description="Multi-page PDF on the gain-vs-charge correction and Fe-55 / cluster energies.")
    ap.add_argument("--universal", required=True,
                    help=f"folder with universal_results.json and WADERs_analysis_<tag>/ (the report was made from {U})")
    ap.add_argument("--dm2426", default=DM2426, help=f"WADERs analysis of the DM24-26 Image 5 set (default: {DM2426})")
    ap.add_argument("--out", default=OUT, help=f"output PDF (default: {OUT})")
    ap.add_argument("--pages-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "pages"),
                    help="folder for the page_NN.png copies of the pages (default: pages/ next to this script)")
    args = ap.parse_args()
    u = args.universal
    res = json.load(open(f"{u}/universal_results.json"))
    rows = res["rows"]

    with PdfPages(args.out) as _pp:
        pdf = _Both(_pp, args.pages_dir)
        page_intro(pdf, rows)
        page_curves(pdf, u, res)
        pages_single_ccd(pdf, u, rows)
        pages_alignment(pdf, u, args.dm2426)
        page_binned(pdf, u, rows)
        page_all_datasets(pdf, rows)
        page_limits(pdf, res)
    print("->", args.out)


if __name__ == "__main__":
    main()
