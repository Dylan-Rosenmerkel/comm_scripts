"""Commissioning packet report (run_superWADERs --packet): one interactive HTML page per module from the files of a
packet, plus an index.

A packet is a list of files in any order (.txt), of these types (recognised from the file, not the order):
  trace    trace_*.RJ45ADCDATA<ch>      L0 trace of one channel: raw video samples with the window flags
  psd      psd_*.RJ45ADCDATA<ch>        L0 noise trace of one channel (raw samples, no windows)
  image    avg_*.fits(.bz2/.fz)         averaged image of all channels (e.g. 5 min 1x1 1 skip, 5 min 10x10 500 skips)
  image    *_L1_*.RJ45ADCDATA<ch>       level-1 image (the 4 channel files + .meta): converted to a FITS image with all
                                        its skips (raw_data.convert_l1), then as above
DAQ binaries of averaged (L2) images are not read (give their avg_*.fits). A channel file without samples (the
channels a trace did not record) is left out.

Every CCD is identified by its ACM (board) id, extension number (= channel + 1), CCD name (the channel's name in the
header / .meta, e.g. CCD_D) and module name (from --module-list, see read_module_list; else ACM<id>). The report of
a module (packet/<module>_ACM<id>.html) has the tabs for which the packet has data:
  Overview, Trace, PSD, Image summary (fits per image; no clustering), one image viewer per image,
and check boxes for its CCDs (all of them or single ones): traces are shown one below the other, PSDs and profiles
overlaid, numbers side by side, images one CCD at a time with a slider (as ds9 frames).
Each tab has a comment box (kept in the browser; 'Save comments' / 'Load comments' as a JSON file) and a log, and
the page has 'Export PDF' (all tabs, comments included, through the browser's print). The page is self-contained:
data are gzip + base64 in the page, plots and the image viewer are drawn by its own script (no network needed).

    from packet_report import build
    lines = build(files, out_dir, module_list=None, autoconfig_args=())
"""
import base64
import gzip
import hashlib
import html
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # superWADERs/
for _p in ("commissioning", "clustering", "engine"):          # superWADERs' own modules first on the path
    _p = os.path.join(SW_DIR, _p)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
if os.path.join(HERE, "raw_levels") not in sys.path:
    sys.path.insert(0, os.path.join(HERE, "raw_levels"))
import raw_data as RD          # noqa: E402

FS = 15e6                       # ADC samples per second (decode_trace.SAMPLE_RATE_HZ)
MAX_VIEW_PIXELS = 4_000_000     # larger images are shown binned 2x2 (repeatedly) in the viewer
FITS_EXT = (".fits", ".fits.bz2", ".fits.gz", ".fz")


# =============================================================================
# encoding of arrays for the page
# =============================================================================
def enc(a, dtype="f4"):
    """{'dtype', 'b64'} of an array: gzip of little-endian float32 ('f4'), float64 ('f8'), uint8 ('u1') or int32
    differences ('i4d', for integer samples: compresses well, summed again in the page)."""
    a = np.asarray(a)
    if dtype == "i4d":
        a = a.astype(np.int64)
        raw = np.diff(a, prepend=0).astype("<i4").tobytes()
    elif dtype == "u1":
        raw = a.astype(np.uint8).tobytes()
    else:
        raw = a.astype("<" + dtype).tobytes()
    return {"dtype": dtype, "b64": base64.b64encode(gzip.compress(raw, 6, mtime=0)).decode()}


def logbin(f, a, n=4000):
    """Spectrum (f, a) reduced to <= n log-spaced bins (mean of a per bin) for display."""
    keep = f > 0
    f, a = f[keep], a[keep]
    if f.size <= n:
        return f, a
    edges = np.geomspace(f[0], f[-1] * 1.000001, n + 1)
    idx = np.searchsorted(edges, f, side="right") - 1
    cnt = np.bincount(idx, minlength=n)
    ok = cnt > 0
    fm = np.bincount(idx, f, minlength=n)[ok] / cnt[ok]
    am = np.bincount(idx, a, minlength=n)[ok] / cnt[ok]
    return fm, am


# =============================================================================
# identity: ACM, channel, module names
# =============================================================================
def acm_of(name):
    """ACM (board) id from a DAQ file name (..._<ACM>_<YYYYMMDD>_<hhmmss>_...), or None."""
    m = re.search(r"_(\d{2,4})_\d{8}_\d{6}", os.path.basename(name))
    return int(m.group(1)) if m else None


def read_module_list(path):
    """The correlation list of module names: one line per ACM (or per CCD), comma, tab or space separated; '#'
    comments and a header line are ignored. Columns: ACM id, module name [, extension or channel name, CCD name].
    Examples:   101, DM24        |   ACM101 PA03        |   101, DM24, EXT2, CCD_C
    A line with an extension (EXT<n> / <n>) or channel name (CCD_A ...) names that CCD only. Returns
    {(acm, None): module, (acm, ext or ccd name): module}."""
    out = {}
    if not path:
        return out
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        tok = [t for t in re.split(r"[,\t;]+|\s+", line) if t]
        m = re.fullmatch(r"(?:ACM)?(\d+)", tok[0], re.I)
        if not m or len(tok) < 2:
            continue                                     # header or unreadable line
        acm, module = int(m.group(1)), tok[1]
        if len(tok) >= 3:
            e = re.fullmatch(r"(?:EXT)?(\d)", tok[2], re.I)
            out[(acm, int(e.group(1)) if e else tok[2].upper())] = module
        else:
            out[(acm, None)] = module
    return out


def module_for(mods, acm, ext, ccd_name):
    return (mods.get((acm, ext)) or mods.get((acm, (ccd_name or "").upper())) or mods.get((acm, None))
            or f"ACM{acm}")


def meta_or_header(path):
    """Settings of a file: the .meta of a DAQ binary, the primary header of a FITS image (upper-case keys)."""
    if RD.kind(path):
        return {k.upper(): v for k, v in RD.read_meta(RD.base_of(path)).items()}
    from astropy.io import fits
    with fits.open(path) as h:
        return {k.upper(): v for k, v in h[0].header.items()}


# =============================================================================
# per file: trace, psd, image
# =============================================================================
def trace_data(path, meta):
    """Trace tab data of one L0 trace (decode_trace / plot_trace numbers and arrays)."""
    from decode_trace import decode_trace
    from plot_trace import averaged_psd
    t = decode_trace(path)
    if len(t.values) == 0:
        return None
    cycles, period, cpix = t.skip_cycles()
    sk = t.skips()
    npix = len(np.unique(sk["pixel"]))
    probs = t.quality_problems()
    numbers = [["file", os.path.basename(path)], ["samples", f"{len(t.values)} ({len(t.values) / FS * 1e3:.1f} ms at 15 MHz)"],
               ["skip cycles", f"{len(cycles)} of {period} samples ({period / FS * 1e6:.1f} us)"],
               ["pixels x skips", f"{npix} x {len(sk['cds']) // max(npix, 1)}"],
               ["inline vs raw window sums", "agree" if t.check_l1() == 0 else f"differ by {t.check_l1()}", t.check_l1() != 0],
               ["quality", "OK" if not probs else "; ".join(probs), bool(probs)],
               ["CDS rms per skip", f"{np.std(sk['cds']):.1f} ADU (uncalibrated, includes transients)"]]
    out = dict(file=os.path.basename(path), numbers=numbers, dt_us=1e6 / FS, values=enc(t.values, "i4d"),
               rd=enc(t.rd, "u1"), ru=enc(t.ru, "u1"),
               skips=dict(pedestal=enc(sk["pedestal"]), signal=enc(sk["signal"]), cds=enc(sk["cds"])))
    if len(cycles) > 2:
        f = cycles[:, :].astype(float)
        f -= f[:, :min(150, f.shape[1])].mean(axis=1, keepdims=True)
        out["folded"] = dict(mean=enc(f.mean(axis=0)), std=enc(f.std(axis=0)))
        res = (f - f.mean(axis=0)).ravel()
        fr, am = averaged_psd(res, segment=period * 8)
        fr, am = logbin(np.asarray(fr), np.asarray(am))
        out["spectrum"] = dict(f=enc(fr, "f8"), a=enc(am))
    log = [f"trace {path}", f"ACM {meta.get('ACMID')}, routine {meta.get('ROUTINE')}, sequencer {meta.get('SEQNAME')}",
           f"DATEINI {meta.get('DATEINI')}  DATEEND {meta.get('DATEEND')}",
           f"delayInt {meta.get('DELAYINT')} cycles, NDCM {meta.get('NDCM')}, NROW {meta.get('NROW')}, NCOL {meta.get('NCOL')}",
           f"undecoded words {len(t.other)}"] + [f"quality: {p}" for p in probs]
    return out, log


def psd_data(path, meta):
    """PSD tab data of one noise trace: samples, periodogram, Welch spectrum, sigma_ADU and sigma_CDS.
    sigma_CDS: rms of (mean of one window - mean of the next) for consecutive windows of the CDS integration length
    (delayInt from the .meta -> samples, as decode_trace.window_samples; 150 samples if missing), i.e. the noise a
    CDS of that window length would see from this signal (no dead time between the windows)."""
    from decode_trace import decode_trace, window_samples
    from scipy import signal
    t = decode_trace(path)
    if len(t.values) == 0:
        return None
    v = t.values.astype(float)
    x = v - v.mean()
    n = len(x)
    X = np.fft.rfft(x)
    fp = np.fft.rfftfreq(n, 1 / FS)
    ap = np.sqrt(2 * np.abs(X) ** 2 / (FS * n))
    fp, ap = logbin(fp, ap)
    fw, pw = signal.welch(x, FS, nperseg=min(n, 1 << 15))
    aw = np.sqrt(pw)
    try:
        nw = int(round(window_samples(float(meta.get("DELAYINT")))))
    except (TypeError, ValueError):
        nw = 150
    m = n // nw
    means = x[:m * nw].reshape(m, nw).mean(axis=1)
    cds = means[1::2][:len(means[0::2])] - means[0::2][:len(means[1::2])]
    out = dict(label=os.path.basename(path).split(".RJ45")[0], dt_us=1e6 / FS, values=enc(t.values, "i4d"),
               pf=enc(fp, "f8"), pa=enc(ap), wf=enc(fw, "f8"), wa=enc(aw), sigma_adu=float(np.std(x)),
               sigma_cds=float(np.std(cds)), n=n, nw=nw)
    log = [f"psd {path}: {n} samples ({n / FS * 1e3:.1f} ms), Welch segments of {min(n, 1 << 15)} samples",
           f"sigma_ADU {np.std(x):.2f}, sigma_CDS {np.std(cds):.3f} ADU (windows of {nw} samples; delayInt "
           f"{meta.get('DELAYINT')})", f"DATEINI {meta.get('DATEINI')}, routine {meta.get('ROUTINE')}"]
    return out, log


def image_role(meta, path):
    """'<N>-skip <bin> image' from the header."""
    nd, pb, sb = meta.get("NDCM"), meta.get("NPBIN"), meta.get("NSBIN")
    if nd is None:
        m = re.search(r"_(\d+)x(\d+)x(\d+)_(\d+)x(\d+)_", os.path.basename(path))
        if m:
            nd, pb, sb = m.group(3), m.group(4), m.group(5)
    return f"{nd}-skip {pb}x{sb} image"


def image_data(path, autoconfig_args, s=None):
    """{ext number: (summary tab data, viewer tab data, log lines)} of one averaged image: native pedestal and
    fits per CCD (native_pipeline.process_image, no cluster files; or s, its result with keep_arrays and all_ext
    when already made), histograms, profiles, the pixels."""
    import native_pipeline as NP
    from native_plots import plot_range
    meta = meta_or_header(path)
    role = image_role(meta, path)
    if s is None:
        s = NP.process_image(path, os.devnull, outputs=[], write=False, keep_arrays=True, all_ext=True, verbose=False,
                             autoconfig_args=autoconfig_args)
    img = NP.read_image(path, autoconfig_args)
    raw = dict(img["exts"])
    out = {}
    for ext, c in s["ccds"].items():
        n = int(ext[3:])
        log = [f"image {path}", f"{role}; NROW {meta.get('NROW')}, NCOL {meta.get('NCOL')}, EXPOSURE {meta.get('EXPOSURE')} s, "
               f"DATEINI {meta.get('DATEINI')}  DATEEND {meta.get('DATEEND')}",
               f"geometry: prescan {img['geo']['n_cols_prescan']} columns, active up to column {img['geo']['ccd_cols']}"]
        if "ps" not in c:                    # not fitted (e.g. no CCD signal): still show the image and its numbers
            r_ = raw[n] - np.median(raw[n], axis=1, keepdims=True)
            mad = 1.4826 * np.median(np.abs(r_))
            log.append(f"{ext}: not processed ({c.get('skipped')}); shown: raw minus the median of each row")
            numbers = [["image", os.path.basename(path)], ["type", role], ["status", f"not fitted: {c.get('skipped')}", True],
                       ["noise (MAD of raw - row median)", f"{mad:.3f} ADU"],
                       ["pixels > 5 MAD", f"{np.mean(r_ > 5 * mad) * 100:.3f} %"], ["size", f"{r_.shape[0]} rows x {r_.shape[1]} columns"]]
            summ = dict(role=role, file=os.path.basename(path), numbers=numbers, unit="ADU",
                        mu_row=enc(np.median(raw[n], axis=1)), sig_row=enc(1.4826 * np.median(np.abs(r_), axis=1)),
                        col_med=enc(np.median(r_, axis=0)), row_med=enc(np.median(r_, axis=1)))
            out[n] = (summ, dict(role=role, file=os.path.basename(path), unit="ADU (raw - row median)", w=int(r_.shape[1]),
                                 h=int(r_.shape[0]), decimated=0, pixels=enc(r_)), log)
            continue
        ps, g = c["ps"], c["gain"]
        active, _ = NP.regions(ps.shape, img["geo"])
        st = c["fit_settings"]
        fit = c["fits"]["final"]
        resolved = c["resolved"] and c["noise_e"] < 0.35
        unit = "ADU"
        x1 = plot_range(st, {"resolved": resolved, "gain": g}, 5)
        data = ps[NP.fit_mask(ps.shape, active, st)]
        nb = max(int(round((x1 - st["x_min"]) / st["binning_size"])), 10)
        if resolved:                            # the dark-current fit's histogram and model
            hy, he = np.histogram(data[(data > st["x_min"]) & (data < x1)], nb, (st["x_min"], x1))
            mx = np.linspace(st["x_min"], x1, 1500)
            p = [fit[k][0] for k in ("norm", "mu", "sigma", "lam", "gain")]
            npk = max(st["n_peaks"], int(math.ceil((x1 - p[1]) / max(p[4], 1e-6))) + 1)
            my = NP.comb_model(mx, p, npk)
        else:                                   # no electron peaks: no comb fit; the pixel distribution only
            lo_, hi_ = np.percentile(data, [0.5, 99.5])
            hy, he = np.histogram(data[(data > lo_) & (data < hi_)], 400, (lo_, hi_))
            mx = my = None
        hx = he[:-1]
        e = ps / g
        lam_row = None
        if resolved:
            fr0 = np.array([np.mean(e[r][active[r]] < 0.5) if active[r].any() else np.nan for r in range(e.shape[0])])
            with np.errstate(divide="ignore", invalid="ignore"):
                lam_row = -np.log(fr0)
        row_med = np.array([np.median(ps[r][active[r]]) if active[r].any() else np.nan for r in range(ps.shape[0])])
        held = c.get("held")
        act_px = ps[active]
        if resolved:
            numbers = [["image", os.path.basename(path)], ["type", role],
                       ["gain", f"{g:.4f} +- {c['gain_err']:.4f} ADU/e-" + (" (held: many-peak fit)" if held else "")],
                       ["noise", f"{c['noise_e']:.4f} e- = {fit['sigma'][0]:.3f} ADU", c["noise_e"] > 0.3],
                       ["dark charge (fit)", f"{c['dc_e']:.4f} e-/pix"],
                       ["dark charge (0e-/1e- counts)", f"{c['lam01']:.4f} e-/pix" if c.get("lam01") is not None else "-"],
                       ["chi2 / ndf", f"{c['chi2ndf']:.2f}"], ["electron peaks resolved", "yes"]]
        else:                                   # nothing to fit: what the pixels show
            numbers = [["image", os.path.basename(path)], ["type", role],
                       ["electron peaks resolved", "no: no gain or dark-current fit (single electrons not visible)"],
                       ["read noise (pedestal sigma, mean over rows)", f"{np.nanmean(c['sig_row']):.3f} ADU"],
                       ["active area: median level", f"{np.median(act_px):.2f} ADU (~{np.median(act_px) / g:.1f} e- with the nominal gain {g:.3f})"],
                       ["active area: spread (MAD)", f"{1.4826 * np.median(np.abs(act_px - np.median(act_px))):.2f} ADU"],
                       ["pixels > 10 x noise", f"{np.mean(act_px > 10 * np.nanmean(c['sig_row'])) * 100:.2f} %"]]
        numbers += [
                   ["pedestal", c["pedestal"]["description"]]] + ([["gain choice", c["gain_note"]]] if resolved else []) + [
                   ["pedestal sigma (mean over rows)", f"{np.nanmean(c['sig_row']):.3f} ADU"],
                   ["size", f"{ps.shape[0]} rows x {ps.shape[1]} columns"]]
        summ = dict(role=role, file=os.path.basename(path), numbers=numbers, unit=unit,
                    hist=dict(x=enc(hx, "f8"), y=enc(hy.astype(float)), mx=enc(mx, "f8") if mx is not None else None,
                              my=enc(my) if my is not None else None, fitted=bool(resolved)),
                    mu_row=enc(c["mu_row"]), sig_row=enc(c["sig_row"]), lam_row=enc(lam_row) if lam_row is not None else None,
                    col_med=enc(np.median(ps, axis=0)), row_med=enc(row_med))
        view, dec = ps, 1
        while view.size > MAX_VIEW_PIXELS:
            h2, w2 = view.shape[0] // 2 * 2, view.shape[1] // 2 * 2
            view = view[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))
            dec *= 2
        viewer = dict(role=role, file=os.path.basename(path), unit=unit, w=int(view.shape[1]), h=int(view.shape[0]),
                      decimated=dec if dec > 1 else 0, pixels=enc(view))
        log += [f"{ext}: gain {g:.4f} ADU/e-, noise {c['noise_e']:.4f} e-, DC {c['dc_e']:.4f} e-/pix, chi2/ndf {c['chi2ndf']:.2f}",
                f"{ext}: {c['gain_note']}", f"{ext}: pedestal {c['pedestal']['description']}",
                f"{ext}: fit window [{st['x_min']}, {st['x_max']}] ADU, {st['n_peaks']} peaks, binning {st['binning_size']} ADU"]
        if c.get("pedestal_note"):
            log.append(f"{ext}: NOTE {c['pedestal_note']}")
        out[n] = (summ, viewer, log)
    return out, meta


# =============================================================================
# the packet
# =============================================================================
def classify(paths):
    """{'trace': [...], 'psd': [...], 'image': [...], 'l1': [...], 'other': [...]} (any order in, by file type)."""
    out = {k: [] for k in ("trace", "psd", "image", "l1", "other")}
    for p in paths:
        n = os.path.basename(p)
        k = RD.kind(p)
        if k == "L0":
            out["psd" if n.startswith("psd_") else "trace"].append(p)
        elif k == "L1":
            out["l1"].append(p)
        elif k is None and n.endswith(FITS_EXT):
            out["image"].append(p)
        else:
            out["other"].append(p)
    return out


def file_items(path, kind):
    """The packet items of one level-0 trace or PSD trace: [dict(kind, acm, ext, ccd, file, res, log)] (one channel;
    [] when the channel was not recorded; an item with 'error' when it failed)."""
    meta = meta_or_header(path)
    ch = int(RD.RAW_RE.search(path).group(1))
    acm = meta.get("ACMID") or acm_of(path)
    try:
        res = (trace_data if kind == "trace" else psd_data)(path, meta)
    except Exception as e:                          # report and go on
        return [dict(kind=kind, file=os.path.basename(path), error=f"{type(e).__name__}: {e}")]
    if res is None:
        return []
    return [dict(kind=kind, acm=int(acm or 0), ext=ch + 1, ccd=meta.get(f"CH{ch}"), file=os.path.basename(path),
                 res=res[0], log=res[1])]


def image_numbers(s, meta):
    """{ext number: numbers of the packet overview} of one image's native summary s."""
    nsk = int(meta.get("NDCM", meta.get("NSKIPS", 1)) or 1)
    out = {}
    for ext, c in s["ccds"].items():
        if "gain" not in c:
            out[int(ext[3:])] = dict(nskips=nsk, status=c.get("skipped") or "not processed")
            continue
        resolved = bool(c.get("resolved")) and c["noise_e"] < 0.35
        out[int(ext[3:])] = dict(nskips=nsk, resolved=resolved, gain=c["gain"], noise_e=c["noise_e"],
                                 noise_adu=(c.get("sigma0_e") or 0) * c["gain"], dc_e=c["dc_e"] if resolved else None,
                                 role=image_role(meta, ""))
    return out


def image_items(path, meta, per_ext, numbers, char=None):
    """The packet items of one averaged image from image_data's {ext number: (summary, viewer, log)} and
    image_numbers(): one per CCD. char: the front end's record of the image (quiet channels)."""
    acm = meta.get("ACMID") or acm_of(path)
    out = []
    for n, (summ, view, log) in per_ext.items():
        ch = (((char or {}).get("ccds") or {}).get(f"EXT{n}") or {}).get("channel", {}).get("value")
        out.append(dict(kind="image", acm=int(acm or 0), ext=n, ccd=meta.get(f"CH{n - 1}"), file=os.path.basename(path),
                        res=[summ, view, log], num=dict(numbers.get(n, {}), channel=ch)))
    return out


def build(paths, out_dir, module_list=None, autoconfig_args=(), log=print, mods_of=None):
    """Reports of every CCD in the packet files paths -> out_dir/packet/ (without a campaign: everything made here;
    images without the characterisation front end). mods_of(acm, ext, ccd) -> module name (default: from the
    module_list file). Returns summary lines."""
    mods = read_module_list(module_list)
    if mods_of is None:
        mods_of = lambda acm, ext, ccd: module_for(mods, acm, ext, ccd)
    cls = classify(paths)
    items, pre = [], []
    if cls["l1"]:
        made, ls = RD.convert_l1(cls["l1"], "all", 0, os.path.join(out_dir, "L1"))
        cls["image"] += made
        pre += ls
    for kind in ("trace", "psd"):
        for p in sorted(cls[kind]):
            log(f"  {kind} {os.path.basename(p)}")
            items += file_items(p, kind)
    for p in sorted(cls["image"]):
        log(f"  image {os.path.basename(p)}")
        try:
            import native_pipeline as NP
            s = NP.process_image(p, os.devnull, outputs=[], write=False, keep_arrays=True, all_ext=True, verbose=False,
                                 autoconfig_args=autoconfig_args)
            per_ext, meta = image_data(p, autoconfig_args, s)
        except Exception as e:                      # report and go on
            items.append(dict(kind="image", file=os.path.basename(p), error=f"{type(e).__name__}: {e}"))
            continue
        items += image_items(p, meta, per_ext, image_numbers(s, meta))
    lines, _ = render(items, out_dir, mods_of, n_files=len(paths), classes=cls)
    return lines[:1] + pre + lines[1:]


def render(items, out_dir, mods_of, n_files=None, classes=None):
    """packet/: one page per module and the overview rows, from every packet item (file_items, image_items) of the
    campaign. Returns (summary lines, overview rows: one per CCD)."""
    pdir = os.path.join(out_dir, "packet")
    os.makedirs(pdir, exist_ok=True)
    nk = lambda k: sum(1 for it in items if it.get("kind") == k)
    lines = [f"commissioning packet: {len({it.get('file') for it in items})} file(s) with data: {nk('trace')} trace, "
             f"{nk('psd')} psd, {len({it['file'] for it in items if it.get('kind') == 'image'})} image(s)"
             + (f", not used: {len(classes['other'])} ({', '.join(os.path.basename(p) for p in classes['other'][:4])})"
                if classes and classes.get("other") else "")]
    lines += [f"  {it['file']}: FAILED ({it['error']})" for it in items if it.get("error")]
    ccds = {}                                   # (acm, ext) -> dict(traces, psds, images, names, numbers)

    def slot(acm, ext):
        return ccds.setdefault((acm, ext), dict(traces=[], psds=[], images=[], ccd=None, files=[], nums=[], flags=[]))

    for it in items:
        if it.get("error"):
            continue
        s = slot(it["acm"], it["ext"])
        s["ccd"] = s["ccd"] or it.get("ccd")
        if it["kind"] in ("trace", "psd"):
            s["traces" if it["kind"] == "trace" else "psds"].append((it["res"], it["log"]))
            s["files"].append([it["kind"], it["file"]])
            if it["kind"] == "trace":
                for row in it["res"]["numbers"]:
                    if len(row) > 2 and row[2]:
                        s["flags"].append(("trace_sums" if row[0].startswith("inline") else "trace", row[0], row[1], it["file"]))
        else:
            s["images"].append(tuple(it["res"]))
            s["files"].append(["image", it["file"]])
            s["nums"].append(dict(it.get("num") or {}, file=it["file"]))
    pages, ov = {}, []                          # module -> [(acm, ext, ccd, tabs, first image numbers, files)]
    for (acm, ext), s in sorted(ccds.items()):
        ccd = s["ccd"] or f"ch{ext - 1}"
        module = mods_of(acm, ext, ccd)
        rid = f"{module}_ACM{acm}_EXT{ext}_{ccd}"
        tabs = []
        ov_rows = [["module", module], ["ACM", str(acm)], ["extension", f"EXT{ext} (channel {ext - 1})"], ["CCD", ccd],
                   ["traces", str(len(s["traces"]))], ["PSD traces", str(len(s["psds"]))], ["images", str(len(s["images"]))]]
        for summ, _, _ in s["images"]:
            ov_rows += [[f"{summ['role']}", "; ".join(f"{k} {v}" for k, v, *_ in summ["numbers"][2:5])]]
        tabs.append(dict(id="overview", label="Overview", kind="overview", numbers=ov_rows, files=s["files"],
                         log=[f"report {rid}", f"module {module} (module list, --module, folder name; ACM<id> if none)"]))
        for k, (tr, lg) in enumerate(s["traces"]):
            tabs.append(dict(id=f"trace{k}", label="Trace" + (f" {k + 1}" if len(s["traces"]) > 1 else ""), kind="trace",
                             data=tr, log=lg))
        if s["psds"]:
            nums = [[f["label"], f"sigma_ADU {f['sigma_adu']:.2f}, sigma_CDS {f['sigma_cds']:.3f} ADU ({f['nw']}-sample windows), "
                     f"{f['n']} samples"] for f, _ in s["psds"]]
            tabs.append(dict(id="psd", label="PSD", kind="psd", data=dict(files=[f for f, _ in s["psds"]], numbers=nums),
                             log=sum((lg for _, lg in s["psds"]), [])))
        for summ, view, lg in sorted(s["images"], key=lambda x: x[0]["role"]):
            short = summ["role"].replace(" image", "")
            slug = re.sub(r"\W+", "_", short)                 # the same tab id for every CCD of the module
            tabs.append(dict(id=f"summary_{slug}", label=f"{short}: numbers", kind="summary", data=[summ], log=lg))
            if view is not None:
                tabs.append(dict(id=f"image_{slug}", label=f"{short}: image", kind="image", data=view, log=lg))
        first = next((a for a, _, _ in s["images"] if a["numbers"][0][0] == "image"), None)
        pages.setdefault(module, []).append((acm, ext, ccd, tabs, first["numbers"][2:4] if first else [], s["files"]))
        files = sorted({f for _, f in s["files"]})
        ov.append(dict(module=module, acm=acm, ext=f"EXT{ext}", ccd=ccd, values=_packet_values(s), flags=s["flags"],
                       nums=s["nums"], files=files, stamp=f"{len(files)} file(s): {', '.join(files)[:300]}"))
    for module, its in sorted(pages.items()):
        rid, name, ccd_list, tabs = module_page(module, [it[:4] for it in its])
        path = os.path.join(pdir, rid + ".html")
        nfiles = len({f[1] for it in its for f in it[5]})
        write_page(path, rid, f"{name} · {len(its)} CCD(s)",
                   f"commissioning packet report (run_superWADERs --packet), {nfiles} file(s)", tabs, ccd_list, back="index.html",
                   tool="packet", module=module, stamp=f"{nfiles} file(s) "
                   + hashlib.sha1("\n".join(sorted({f[1] for it in its for f in it[5]})).encode()).hexdigest()[:10])
        for r in ov:
            if r["module"] == module:
                r["page"] = rid + ".html"
        lines.append(f"  {rid}: CCDs {', '.join(c['label'] for c in ccd_list)}; {', '.join(t['label'] for t in tabs)}  "
                     f"({os.path.getsize(path) / 1e6:.1f} MB)")
    lines.append(f"  overview: {os.path.join(pdir, 'index.html')}")
    return lines, ov


def _packet_values(s):
    """Numbers of one CCD of the packet overview: PSD sigmas (latest PSD trace), the image with the most skips
    (resolved: gain, read noise, dark charge), the 1-skip image (single-skip noise in ADU)."""
    v = {}
    if s["psds"]:
        f = s["psds"][-1][0]
        v.update(psd_sigma_adu=f["sigma_adu"], psd_sigma_cds_adu=f["sigma_cds"])
    v["traces"], v["psd_traces"], v["images"] = len(s["traces"]), len(s["psds"]), len(s["images"])
    ims = [n for n in s["nums"] if "gain" in n]
    res = [n for n in ims if n.get("resolved")]
    if res:
        b = max(res, key=lambda n: n["nskips"])
        v.update(read_noise_e=b["noise_e"], gain=b["gain"], dc_e_per_pix=b["dc_e"], resolved_image=b["file"],
                 resolved_skips=b["nskips"])
    elif ims:                                   # no resolved image: the noise of the one with the most skips (judged
        b = max(ims, key=lambda n: n["nskips"])  # in overview() when it has enough skips to resolve the peaks)
        v.update(_unresolved_noise_e=b["noise_e"], _unresolved_skips=b["nskips"], _unresolved_image=b["file"])
    one = [n for n in ims if n["nskips"] == 1]
    if one:
        v.update(noise_single_adu=one[-1]["noise_adu"], single_image=one[-1]["file"])
    return v


FIELDS = {"read_noise_e": ("read noise (image with the most skips)", "e-"), "gain": ("gain", "ADU/e-"),
          "dc_e_per_pix": ("dark charge", "e-/pix"), "resolved_image": ("image used", ""), "resolved_skips": ("its skips", ""),
          "noise_single_adu": ("single-skip noise (1-skip image)", "ADU"), "single_image": ("1-skip image", ""),
          "psd_sigma_adu": ("PSD sigma", "ADU"), "psd_sigma_cds_adu": ("PSD sigma CDS", "ADU"),
          "traces": ("traces", ""), "psd_traces": ("PSD traces", ""), "images": ("images", "")}


def overview(camp, rows, crit, path, back, expect_resolved=True):
    """packet/index.html: the module x CCD overview of the packets. Returns the view."""
    import campaign as CA
    sec = CA.section(crit, "packet")
    for r in rows:
        flags = [(k, "warn", f"trace {f}: {lab} {val}", "trace quality" if k == "trace" else "trace sums differ")
                 for k, lab, val, f in r.pop("flags", [])]
        nums = r.pop("nums", [])
        if any(n.get("channel") == "quiet" for n in nums):
            flags.append(("channel", "fail", "no charge seen in an image (active region = overscan): a dead channel or an "
                                             "ultra-low dark-charge CCD", "no charge seen (dead channel?)"))
        ims = [n for n in nums if "gain" in n]
        if ims and not any(n.get("resolved") for n in ims) and max(n["nskips"] for n in ims) >= \
                int(crit.get("expect_resolved_min_skips", 100)):
            flags.append(("unresolved", "warn", "electron peaks not resolved in any image with "
                                                f">= {crit.get('expect_resolved_min_skips', 100)} skips", "peaks not resolved"))
        un = {k: r["values"].pop(k) for k in list(r["values"]) if k.startswith("_unresolved")}
        if un and un["_unresolved_skips"] >= int(crit.get("expect_resolved_min_skips", 100)):
            r["values"].update(read_noise_e=un["_unresolved_noise_e"], resolved_image=un["_unresolved_image"] + " (not resolved; prior gain)",
                               resolved_skips=un["_unresolved_skips"])
        r["verdict"], r["reasons"], r["levels"] = CA.judge(r["values"], sec, flags)
        r["links"] = [["module page", r.pop("page")]] if r.get("page") else []
        files = r.pop("files", [])
        r["history"] = dict(head=["file"], rows=[[f] for f in files]) if files else None
    view = dict(name="commissioning packets", ccds=rows)
    views_out = CA.write_overview(path, dict(
        title="Commissioning packets", sub=CA.sub_line(camp), back=back, criteria=sec,
        fields={k: dict(label=a, unit=b) for k, (a, b) in FIELDS.items()},
        sections=[["Noise", ["read_noise_e", "noise_single_adu", "psd_sigma_adu", "psd_sigma_cds_adu"]],
                  ["Fit", ["gain", "dc_e_per_pix", "resolved_image", "resolved_skips", "single_image"]],
                  ["Files", ["traces", "psd_traces", "images"]]],
        cell=["read_noise_e", "noise_single_adu", "gain"], views=[view], review=dict(tool="packet"),
        columns=["read_noise_e", "gain", "dc_e_per_pix", "noise_single_adu", "psd_sigma_adu", "psd_sigma_cds_adu",
                 "traces", "psd_traces", "images"],
        definitions=[("read noise", "Width of the electron peaks [e-] in the packet's image with the most skips whose peaks "
                                    "are resolved."),
                     ("single-skip noise", "Pedestal noise [ADU] of the 1-skip image (mean over rows)."),
                     ("PSD sigma / sigma CDS", "Noise of the latest PSD trace: rms of the samples, and of the difference "
                                               "of consecutive windows of the CDS integration length [ADU]."),
                     ("module page", "Overview, trace, PSD, the numbers and the image of every packet image "
                                     "(packet/<module>_ACM<id>.html). History in the panel: the files of the CCD.")],
        flag_labels={"trace": "trace quality problem", "trace_sums": "inline vs raw window sums differ",
                     "channel": "no charge seen", "unresolved": "peaks not resolved"}, csv_name="packet_overview"))
    return views_out


def module_page(module, items):
    """One page per module from the pages of its CCDs, items [(acm, ext, ccd name, tabs)]. Returns (page id, title,
    CCD list for the page's check boxes, tabs). A tab of the module holds the data of every CCD that has it:
    tab['per'][ccd key] (and its log in tab['logs'][ccd key])."""
    acms = sorted({a for a, *_ in items})
    ccds = [dict(key=f"ACM{a}_EXT{e}", label=f"EXT{e} · {c}" + (f" · ACM {a}" if len(acms) > 1 else ""), acm=a, ext=f"EXT{e}",
                 ccd=c) for a, e, c, _ in items]
    tabs, by = [], {}
    for a, e, _, ts in items:
        for tb in ts:
            m = by.get(tb["id"])
            if m is None:
                m = by[tb["id"]] = dict(id=tb["id"], label=tb["label"], kind=tb["kind"], per={}, logs={})
                tabs.append(m)
            m["per"][f"ACM{a}_EXT{e}"] = tb["data"] if "data" in tb else dict(numbers=tb.get("numbers"), files=tb.get("files"))
            m["logs"][f"ACM{a}_EXT{e}"] = tb.get("log") or []
    return f"{module}_ACM{'_'.join(map(str, acms))}", f"{module} · ACM {', '.join(map(str, acms))}", ccds, tabs


def flagged(rows):
    """True if a numbers table has a flagged line ([name, value, True])."""
    return any(len(r) > 2 and r[2] for r in rows or [])


def write_page(path, rid, title, ident, tabs, ccds=None, back=None, tool="packet", module=None, stamp=None):
    """One self-contained report page (packet_report_template.html) with the given tabs; a tab gets the red dot when
    one of its numbers is flagged. ccds: the CCDs of a module page (module_page), for tabs with per-CCD data.
    tool, module: what reviews written on the page are about (each tab has a review box); stamp: the data of the page
    (default: a hash of the tabs' data), kept with a review to mark it 'data changed since' on a newer page.
    Returns the file size."""
    import campaign as CA
    module, dataset = CA.split_module(module) if module else (module, "")
    if dataset:                                # a module's data set (--dataset): "<module> [<data set>]" in the title
        title = title.replace(f"{module}{CA.DS_SEP}{dataset}", f"{module} [data set {dataset}]")
    if stamp is None:                       # before the flags: new criteria do not make the reviews' data look changed
        stamp = hashlib.sha1(json.dumps(tabs, sort_keys=True, default=str).encode()).hexdigest()[:12]
    def nums_of(d):
        if isinstance(d, dict):
            return list(d.get("numbers") or [])
        if isinstance(d, list):
            return [r for x in d if isinstance(x, dict) for r in x.get("numbers") or []]
        return []
    for tb in tabs:
        if "per" in tb:
            tb["flagged_ccds"] = [k for k, d in tb["per"].items() if flagged(nums_of(d))]
            tb["flagged"] = bool(tb["flagged_ccds"])
        else:
            tb["flagged"] = flagged(list(tb.get("numbers") or []) + nums_of(tb.get("data")))
    for c in ccds or []:
        c["flagged"] = [tb["label"] for tb in tabs if c["key"] in tb.get("flagged_ccds", [])]
    page = dict(report_id=rid, title=title, ident=ident, tabs=tabs, ccds=ccds, back=back,
                review=CA.review_context(path, tool, module=module, dataset=dataset or None, stamp=stamp))
    body = json.dumps(page, separators=(",", ":")).replace("</", "<\\/")
    body = re.sub(r"(?<=[:,\[])(-?Infinity|NaN)(?=[,\]}])", "null", body)     # JSON has no NaN
    with open(os.path.join(HERE, "packet_report_template.html")) as fh:
        template = fh.read()
    with open(path, "w") as fh:
        fh.write(CA.page_scripts(template, page["review"]["root"]).replace("__TITLE__", html.escape(title)).replace("__DATA__", body))
    return os.path.getsize(path)


def write_index(pdir, rows, title="Commissioning packet: module reports"):
    """<dir>/index.html: one row per report (module)."""
    tr = []
    for module, acm, ext, ccd, fname, tabs, nums, size, flags in rows:
        n = " · ".join(html.escape(f"{k}: {v}") for k, v, *_ in nums)
        fl = (f"<span class='fl'>{html.escape(', '.join(flags))}</span>" if flags else "ok")
        tr.append(f"<tr><td>{html.escape(module)}</td><td>{acm}</td><td>{f'EXT{ext}' if isinstance(ext, int) else html.escape(str(ext))}</td><td>{html.escape(str(ccd))}</td>"
                  f"<td><a href='{html.escape(fname)}'>open</a></td><td>{fl}</td><td>{html.escape(', '.join(tabs))}</td>"
                  f"<td>{n}</td><td>{size / 1e6:.1f} MB</td></tr>")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title.split(':')[0])}</title><style>
:root{{--bg:#fff;--fg:#1d1d1f;--line:#d8d8dc;--muted:#6b6b70}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#16171a;--fg:#e8e8ea;--line:#34353a;--muted:#a0a0a8}}}}
:root[data-theme="dark"]{{--bg:#16171a;--fg:#e8e8ea;--line:#34353a;--muted:#a0a0a8}}
body{{background:var(--bg);color:var(--fg);font:14px -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:16px}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid var(--line);padding:5px 12px 5px 0;text-align:left;vertical-align:top}}
th{{cursor:pointer;user-select:none}}th:hover{{color:#2a6fdb}}a{{color:#2a6fdb}}.fl{{color:#c0392b;font-weight:600}}
input{{font-size:14px;padding:5px 8px;width:min(420px,100%);margin:6px 0 12px;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px}}
.muted{{color:var(--muted);font-size:13px}}
</style></head><body><h1 style="font-size:19px">{html.escape(title)}</h1>
<p class="muted">{len(rows)} report(s). Type to filter (module, ACM, CCD, ...); click a column title to sort. "flags": tabs with a number outside the usual range.</p>
<input id="q" placeholder="filter, e.g. DM24 or 101 or CCD_A">
<table id="t"><thead><tr><th>module</th><th>ACM</th><th>ext</th><th>CCDs</th><th>report</th><th>flags</th><th>tabs</th><th>summary</th><th>size</th></tr></thead>
<tbody>{''.join(tr)}</tbody></table>
<script>
const q=document.getElementById("q"),tb=document.querySelector("#t tbody");
q.oninput=()=>{{const s=q.value.toLowerCase();for(const r of tb.rows)r.style.display=r.textContent.toLowerCase().includes(s)?"":"none";}};
document.querySelectorAll("#t th").forEach((th,k)=>{{let up=true;th.onclick=()=>{{const rs=[...tb.rows];rs.sort((a,b)=>{{const x=a.cells[k].textContent,y=b.cells[k].textContent,nx=parseFloat(x),ny=parseFloat(y);return (isNaN(nx)||isNaN(ny)?x.localeCompare(y):nx-ny)*(up?1:-1);}});up=!up;rs.forEach(r=>tb.appendChild(r));}};}});
</script></body></html>"""
    with open(os.path.join(pdir, "index.html"), "w") as fh:
        fh.write(page)
