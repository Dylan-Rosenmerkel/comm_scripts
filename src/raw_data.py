"""Level-0 and level-1 raw skipper-CCD data for run_superWADERs (run_WADERs --engine native).

The DAQ writes one binary file per channel, <name>.RJ45ADCDATA<ch>, next to <name>.meta (JSON: the settings that
the FITS header of an image carries):

  L0  traces (trace_*): raw ADC samples of the video signal (15 MHz) with the pedestal / signal window flags and
      the inline per-skip window sums; psd_*: raw samples without windows (for noise spectra).
      -> decoded and checked (inline sums = sums of the flagged raw samples), plots and an interactive viewer
         (L0/<name>/), a summary per trace. Traces are not images: nothing goes into the image pipeline.
  L1  images (*_1x1_L1_*): the pedestal and signal window sums of EVERY skip of every pixel.
      -> FITS images like the DAQ's averaged images (avg_img_*.fits: header from the .meta, one image extension
         per channel in channel order, pixel = mean over skips of (pedestal - signal) / n in ADU), one per number of
         skips N (the first N skips after --l1-skip-first), written to L1/ and run through the pipeline like any
         image; afterwards noise and gain against N (L1/noise_vs_skips.*) and the fits of every N side by side
         (L1/compare/, utils/compare_dcfit_vs_skips.py).

The file readers are copies of ccd_opt_ml's (src/raw_levels/, utils/sync_from_ccd_opt_ml.py): decode_trace,
trace_regions, plot_trace, view_trace for L0; for L1 the layout of decode_l1 (also skip_noise_L1.py), read here row
by row (a full L1 image holds 4 x skips words per pixel: GBs), keeping only the sums needed for the requested N.
Binned (NxN) level-1 images have another layout and are not read.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = os.path.join(HERE, "raw_levels")
if LEVELS not in sys.path:
    sys.path.insert(0, LEVELS)

RAW_RE = re.compile(r"\.RJ45ADCDATA(\d)$")


def kind(path):
    """'L0' (trace_ / psd_ files), 'L1' (level-1 images), 'raw' (another DAQ binary, e.g. an L2 image: use its
    avg_*.fits) or None (not a DAQ binary)."""
    n = os.path.basename(path)
    if not RAW_RE.search(n):
        return None
    if n.startswith(("trace_", "psd_")) or "_L0_" in n:
        return "L0"
    return "L1" if "_L1_" in n else "raw"


def base_of(path):
    """<dir>/<name> of a DAQ binary (without .RJ45ADCDATA<ch>)."""
    return RAW_RE.sub("", path)


def read_meta(base):
    """The .meta JSON next to the binaries ({} if missing)."""
    p = base + ".meta"
    return json.load(open(p)) if os.path.isfile(p) else {}


# =============================================================================
# L1 -> FITS images with N skips
# =============================================================================
def default_skips(nskip):
    """1, 2, 5, 10, 20, 50, ... up to nskip, and nskip itself."""
    out, d = [], 1
    while d <= nskip:
        out += [v * d for v in (1, 2, 5) if v * d <= nskip]
        d *= 10
    return sorted(set(out + [nskip]))


def parse_skips(text, nskip, skip_first):
    """--l1-skips: 'auto' (default_skips) or a comma list; 'all' = every skip after skip_first. Returns the N that
    fit into the image (sorted) and the ones that do not."""
    avail = nskip - skip_first
    if text in (None, "", "auto"):
        want = default_skips(avail)
    else:
        want = sorted({avail if v.strip() == "all" else int(v) for v in text.split(",") if v.strip()})
    return [n for n in want if 0 < n <= avail], [n for n in want if n > avail or n <= 0]


def l1_skip_means(path, ncol, nskip, skip_first, ns):
    """Mean over skips [skip_first, skip_first + N) of every pixel, for every N in ns, from one channel's level-1
    file, read row by row (memory map). Layout and checks as decode_l1.read_l1 (skip headers, EPIX, per-skip sums
    against the trailer's sum; window means with each window's own sample count). Returns ({N: (rows, ncol)
    array [ADU], nan where a pixel fails a check}, complete rows, problems); a row cut by the end of the file is
    left out."""
    import decode_l1 as L1
    w = np.memmap(path, dtype=">u4", mode="r")
    pix = 4 * nskip + 2 + L1.TRAILER
    head = np.asarray(w[:256], np.int64)
    first = np.flatnonzero(head == L1.LINE)
    if not len(first):
        raise ValueError(f"{os.path.basename(path)}: no LINE marker at the start: not a 1x1 level-1 image")
    row_len = 1 + ncol * pix
    out = {n: [] for n in ns}
    problems, r = [], 0
    while True:
        o = int(first[0]) + r * row_len
        if o + row_len > len(w):
            if o < len(w) - 1:
                k = (len(w) - o - 1) // pix
                problems.append(f"row {r}: file ends after {k} of {ncol} pixels (row left out)")
            break
        if int(w[o]) != L1.LINE:
            problems.append(f"row {r}: no LINE marker; stopped")
            break
        p = np.asarray(w[o + 1:o + row_len], np.int64).reshape(ncol, pix)
        hdr = p[:, 0:4 * nskip:2]
        counts = hdr & 0xFFFFF
        n = int(np.median(counts))
        good = (p[:, -1] == L1.EPIX) & ((hdr >> 28) == 5).all(1) & (np.abs(counts - n) <= 1).all(1)
        ped, sig = L1.s31(p[:, 1:4 * nskip:4]), L1.s31(p[:, 3:4 * nskip:4])
        tot = L1.s31(p[:, 4 * nskip + 3]) * (1 << 31) + (p[:, 4 * nskip + 4] & 0x7FFFFFFF)
        good &= tot == (sig - ped).sum(axis=1)
        if not good.all():
            problems.append(f"row {r}: {int((~good).sum())} of {ncol} pixels fail the layout checks (set to nan)")
        v = ped / np.maximum(counts[:, 0::2], 1) - sig / np.maximum(counts[:, 1::2], 1)     # (ncol, nskip) ADU
        cs = np.cumsum(v[:, skip_first:], axis=1)
        for m in ns:
            row = cs[:, m - 1] / m
            row[~good] = np.nan
            out[m].append(row)
        r += 1
    return {m: np.array(v) for m, v in out.items()}, r, problems


def fits_header(meta):
    """Primary FITS header from a .meta JSON: the keywords the DAQ writes into its images (upper case; longer names
    as HIERARCH)."""
    from astropy.io import fits
    h = fits.Header()
    for k, v in meta.items():
        key = k.upper()
        if isinstance(v, (list, dict)) or v is None:
            continue
        try:
            if len(key) > 8 or not re.fullmatch(r"[A-Z0-9_-]+", key):
                h[f"HIERARCH {k}"] = v
            else:
                h[key] = v
        except (ValueError, TypeError):
            pass
    return h


def l1_to_fits(base, chans, ns_text, skip_first, outdir):
    """One level-1 image (its channel files chans {ch: path}) -> outdir/avg_<name with N skips>.fits for every N of
    --l1-skips. Missing channels become all-zero extensions (dropped later as 'no CCD signal'), so the extension
    numbers keep matching the channels. Returns ([(N, fits path)], summary lines)."""
    import warnings
    from astropy.io import fits
    from astropy.io.fits.verify import VerifyWarning
    import decode_l1 as L1
    warnings.simplefilter("ignore", VerifyWarning)        # long .meta values: the comment is cut, not the value
    stem = os.path.basename(base)
    lines = [f"  {stem}: level-1 image, channels {', '.join(str(c) for c in sorted(chans))}"]
    geo = L1.geometry_from_name(stem)
    if geo is None:
        lines.append("    not read: only unbinned (_1x1_L1_) level-1 images are supported (binned ones have another "
                     "layout)")
        return [], lines
    nrow, ncol, nskip = geo
    ns, dropped = parse_skips(ns_text, nskip, skip_first)
    if dropped:
        lines.append(f"    skip counts not possible ({nskip} skips, first {skip_first} left out): {dropped}")
    if not ns:
        return [], lines
    means, rows, probs = {}, {}, []
    for ch, p in sorted(chans.items()):
        m, nr, pr = l1_skip_means(p, ncol, nskip, skip_first, ns)
        means[ch], rows[ch] = m, nr
        probs += [f"ch{ch} {x}" for x in pr]
    nr = min(rows.values())
    lines += [f"    {x}" for x in probs[:12]] + ([f"    ... {len(probs) - 12} more"] if len(probs) > 12 else [])
    if nr == 0:
        lines.append(f"    no complete row (of {nrow}) in the files: no image made")
        return [], lines
    lines.append(f"    {nr} of {nrow} rows complete, {nskip} skips; images with N = {', '.join(map(str, ns))} skips "
                 f"(skips {skip_first}..{skip_first}+N-1)")
    meta = read_meta(base)
    os.makedirs(outdir, exist_ok=True)
    made = []
    for n in ns:
        hdr = fits_header(meta)
        hdr["NDCM"] = n
        hdr["NROW"] = nr
        hdr["L1SRC"] = (stem[:60], "level-1 image this was made from")
        hdr["L1SKIP0"] = (skip_first, "first skip used (0-based)")
        hdr["L1NSKIP"] = (n, "skips averaged")
        hdr["L1ALLSK"] = (nskip, "skips in the level-1 image")
        hdus = [fits.PrimaryHDU(header=hdr)]
        for ch in range(max(4, max(chans) + 1)):
            img = means[ch][n][:nr] if ch in means else np.zeros((nr, ncol))
            if np.isnan(img).any():                        # a pixel that failed the checks: its row's median
                med = np.nanmedian(img, axis=1, keepdims=True)
                img = np.where(np.isnan(img), np.nan_to_num(med), img)
            hdus.append(fits.ImageHDU(img.astype(np.float32), name=str(ch)))
        name = "avg_" + re.sub(r"^(\w*?)(\d+)x(\d+)x(\d+)_", lambda mm: f"{mm.group(1)}{nr}x{mm.group(3)}x{n}_", stem, 1)
        path = os.path.join(outdir, name + ".fits")
        fits.HDUList(hdus).writeto(path, overwrite=True)
        made.append((n, path))
    return made, lines


def convert_l1(paths, ns_text, skip_first, outdir):
    """Every level-1 image among paths (channel files grouped by name) -> FITS images. Returns ([fits paths],
    summary lines)."""
    groups = {}
    for p in paths:
        groups.setdefault(base_of(p), {})[int(RAW_RE.search(p).group(1))] = p
    made, lines = [], []
    for base, chans in sorted(groups.items()):
        try:
            m, ls = l1_to_fits(base, chans, ns_text, skip_first, outdir)
        except Exception as e:                          # report and go on with the other images
            m, ls = [], [f"  {os.path.basename(base)}: FAILED ({type(e).__name__}: {e})"]
        made += [p for _, p in m]
        lines += ls
    return made, lines


def skips_of(stem):
    """N of an image name ..._<rows>x<cols>x<N>_..."""
    m = re.search(r"_?(\d+)x(\d+)x(\d+)_", stem)
    return int(m.group(3)) if m else None


def l1_noise_summary(rows, out_dir, fits_paths):
    """After the pipeline: noise and gain of every CCD against N for the images made from level-1 data.
    rows: the fit_quality rows (image, ext, gain, noise [e-], ...). Writes out_dir/noise_vs_skips.csv / .png;
    returns summary lines."""
    stems = {os.path.basename(p)[:-5] for p in fits_paths}
    sel = [q for q in rows if q["image"] in stems]
    if not sel:
        return []
    src = lambda s: re.sub(r"_(\d+)x(\d+)x(\d+)_", "_", s, 1)       # the level-1 image a FITS came from
    table = {}
    for q in sel:
        table.setdefault((src(q["image"]), q["ext"]), []).append((skips_of(q["image"]), q["noise"], q["gain"],
                                                                  q["chi2ndf"]))
    import csv
    with open(os.path.join(out_dir, "noise_vs_skips.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["level1_image", "ext", "skips", "noise_e", "noise_adu", "gain", "chi2ndf"])
        for (s, e), v in sorted(table.items()):
            for n, noise, g, c in sorted(v):
                wr.writerow([s, e, n, round(noise, 5), round(noise * g, 5), round(g, 5), round(c, 3)])
    lines = ["noise vs number of skips (images made from level-1 data; "
             f"{os.path.join(out_dir, 'noise_vs_skips.csv')}, .png):"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["text.usetex"] = False
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for (s, e), v in sorted(table.items()):
        v = sorted(v)
        n = np.array([a[0] for a in v], float)
        y = np.array([a[1] for a in v])
        ax.plot(n, y, "o-", ms=4, lw=1.3, label=f"{e}" + ("" if len({k[0] for k in table}) == 1 else f" {s[-20:]}"))
        ref = n >= max(10, n.min())
        if ref.any():                                   # white-noise line through the points with >= 10 skips
            k0 = np.exp(np.mean(np.log(y[ref] * np.sqrt(n[ref]))))
            ax.plot(n, k0 / np.sqrt(n), ":", lw=0.9, color=ax.lines[-1].get_color())
        lines.append(f"  {s} {e}: " + ", ".join(f"N={int(a[0])} {a[1]:.3f} e-" for a in v))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("skips N")
    ax.set_ylabel("noise [e-]")
    ax.set_title("noise vs skips from level-1 data (dotted: 1/sqrt(N))", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "noise_vs_skips.png"), dpi=120)
    plt.close(fig)
    return lines


def compare_skips(run_out, fits_paths, out_dir, python=sys.executable):
    """The panaSKImg-style plots of the images made from one level-1 image, side by side by N
    (utils/compare_dcfit_vs_skips.py: DC fit, baseline, sigma). Returns summary lines."""
    tool = os.path.join(os.path.dirname(HERE), "utils", "compare_dcfit_vs_skips.py")
    groups = {}
    for p in fits_paths:
        stem = os.path.basename(p)[:-5]
        groups.setdefault(re.sub(r"_(\d+)x(\d+)x(\d+)_", "_", stem, 1), []).append(stem)
    lines = []
    for src, stems in sorted(groups.items()):
        stage = os.path.join(out_dir, "compare_input", src)
        for stem in stems:
            d = os.path.join(stage, str(skips_of(stem)))
            os.makedirs(d, exist_ok=True)
            for f in glob.glob(os.path.join(run_out, f"{stem}_*.pdf")):
                shutil.copy(f, d)
        dest = os.path.join(out_dir, "compare", src)
        for plot in ("FitDarkCurrentProcess_DCfit", "PedestalSubtractionProcess_Baseline",
                     "PedestalSubtractionProcess_Sigma"):
            subprocess.run([python, tool, stage, "-o", dest, "--plot", plot], capture_output=True)
        n = len(glob.glob(os.path.join(dest, "*.pdf")))
        lines.append(f"  {src}: fits of every N side by side: {n} PDFs in {dest}")
    return lines


# =============================================================================
# L0: traces
# =============================================================================
def l0_report(path, out_dir, html=True, python=sys.executable):
    """Decode one trace, check it and plot it into out_dir/<name>/. Pixel traces (with skip windows):
    plot_trace.py's overview (raw cycles, per-skip window means, folded cycle, noise spectrum), view_trace's
    interactive viewer (.html), and the numbers of decode_trace. psd_ traces (no windows): the raw samples and their
    noise spectrum. Returns (summary lines, dict of the numbers)."""
    from decode_trace import SAMPLE_RATE_HZ, decode_trace
    name = os.path.basename(path)
    t = decode_trace(path)
    if len(t.values) == 0 and not len(t.l1_sum):        # only the keyword words: channel not recorded
        return [], dict(file=path, empty=True)
    d = os.path.join(out_dir, name.replace(".", "_"))
    os.makedirs(d, exist_ok=True)
    problems = t.quality_problems()
    info = dict(file=path, samples=int(len(t.values)), duration_ms=len(t.values) / SAMPLE_RATE_HZ * 1e3,
                windows=bool(t.rd.any() or t.ru.any()), quality="OK" if not problems else "; ".join(problems))
    if info["windows"]:
        cycles, period, cpix = t.skip_cycles()
        sk = t.skips()
        info.update(skip_cycles=int(len(cycles)), cycle_us=period / SAMPLE_RATE_HZ * 1e6,
                    pixels=int(len(np.unique(sk["pixel"]))), skips_per_pixel=int(len(sk["cds"]) // max(len(np.unique(sk["pixel"])), 1)),
                    inline_vs_raw=int(t.check_l1()), cds_rms_adu=float(np.std(sk["cds"])))
        r = subprocess.run([python, os.path.join(LEVELS, "plot_trace.py"), path, os.path.join(d, "overview.png")],
                           capture_output=True, text=True)
        if r.returncode:
            info["plot_error"] = (r.stderr.strip().splitlines() or ["?"])[-1]
        if html:
            try:
                from view_trace import build_html
                with open(os.path.join(d, "viewer.html"), "w") as fh:
                    fh.write(build_html(path))
            except Exception as e:                        # the overview still exists
                info["viewer_error"] = f"{type(e).__name__}: {e}"
        line = (f"  {name}: {info['samples']} samples ({info['duration_ms']:.1f} ms), {info['pixels']} pixels x "
                f"{info['skips_per_pixel']} skips, skip cycle {info['cycle_us']:.1f} us, inline vs raw sums "
                f"{'agree' if info['inline_vs_raw'] == 0 else 'differ by ' + str(info['inline_vs_raw'])}, quality "
                f"{info['quality']}; CDS rms per skip {info['cds_rms_adu']:.1f} ADU (uncalibrated)")
    else:
        problems = [x for x in problems if "window" not in x]   # a noise trace has no windows by design
        info["quality"] = "OK" if not problems else "; ".join(problems)
        psd_plot(t, os.path.join(d, "overview.png"), name)
        info["rms_adu"] = float(np.std(t.values))
        line = (f"  {name}: {info['samples']} samples ({info['duration_ms']:.1f} ms), no skip windows (noise "
                f"trace), rms {info['rms_adu']:.1f} ADU, quality {info['quality']}")
    with open(os.path.join(d, "summary.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    return [line + f"  -> {d}/"], info


def psd_plot(t, out, title):
    """Raw samples of a trace without windows: the first 2 ms, the whole trace (min/max per block) and the
    averaged noise spectrum (plot_trace.averaged_psd)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from decode_trace import SAMPLE_RATE_HZ
    from plot_trace import averaged_psd
    plt.rcParams["text.usetex"] = False
    v = t.values.astype(float)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    n2 = min(len(v), int(2e-3 * SAMPLE_RATE_HZ))
    ax[0].plot(np.arange(n2) / SAMPLE_RATE_HZ * 1e3, v[:n2], lw=0.5, color="0.25")
    ax[0].set(title="first 2 ms", xlabel="time [ms]", ylabel="ADC value [ADU]")
    blk = max(len(v) // 2000, 1)
    nb = len(v) // blk
    vb = v[:nb * blk].reshape(nb, blk)
    tb = np.arange(nb) * blk / SAMPLE_RATE_HZ * 1e3
    ax[1].fill_between(tb, vb.min(1), vb.max(1), color="0.6", lw=0)
    ax[1].plot(tb, vb.mean(1), lw=0.6, color="0.1")
    ax[1].set(title="whole trace (min / mean / max per block)", xlabel="time [ms]", ylabel="ADC value [ADU]")
    f, a = averaged_psd(v - v.mean(), segment=min(8192, len(v)))
    ax[2].loglog(f[1:], a[1:], lw=0.8, color="tab:purple")
    ax[2].set(title="noise spectrum", xlabel="frequency [Hz]", ylabel="ADU / sqrt(Hz)")
    ax[2].grid(True, which="both", alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def report_l0(paths, out_dir, html=True):
    """l0_report on every trace; returns summary lines."""
    lines = ["level-0 traces (decoded, checked and plotted; src/raw_levels/ = ccd_opt_ml's decoder and plots):"]
    empty = []
    for p in sorted(paths):
        try:
            ls, info = l0_report(p, out_dir, html)
            if info.get("empty"):
                empty.append(os.path.basename(p))
        except Exception as e:                          # report and go on with the other traces
            ls = [f"  {os.path.basename(p)}: FAILED ({type(e).__name__}: {e})"]
        lines += ls
    if empty:
        lines.append(f"  no samples (channel not recorded in that acquisition; a trace of ch<k> is in .RJ45ADCDATA<k>): "
                     f"{len(empty)} file(s): " + ", ".join(empty[:6]) + (" ..." if len(empty) > 6 else ""))
    return lines
