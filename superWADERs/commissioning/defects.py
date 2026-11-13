"""Defect maps (run_superWADERs --defect-map): median composites and column defects with the group's Image_4 script,
plus a defect log and an interactive defect map.

The group's Image_4 (D. Venegas-Vargas, LSM_Module_Testing_Analysis/src; copy in lsm_defects/,
utils/sync_from_lsm_analysis.py) runs unchanged: composite median of all images of a module and image type (pedestal
per row from the overscan, column defects by a running median / MAD of the column medians, event rate per pixel per
day): its PNG and log, as when run by hand. Its geometry constants (first overscan column, prescan, first
vertical-overscan row) are set from each image's header (autoconfig's geometry) instead of the full-frame values
written in the script. Images without an overscan: it still runs, with its pedestal (row median) and noise taken
from the active region instead of the overscan (noise then includes the dark charge) and no CTI (n/a).
Added here, per CCD, on the composite median:
  columns      bright / dark columns: Image_4's find_col_traps on the composite (median of the column above / below
               the running median +- 6 MAD), kept only if the column is as bright / dark (Image_4's statistic, at
               L x 6 MAD, L = LEVELS) in >= k of the single images, k as for hot pixels from the images' column
               occupancies (< FALSE_SPOTS chance columns per CCD): a track along a column in one image is not a
               bright column. Image_4's own plot and log list its composite columns unfiltered. Adjacent columns of
               one kind are one defect (its column range, the largest excess)
  hot pixels   composite median minus its local level (running median over 3 rows x 31 columns) above 5 x the robust spread of
               that residual (at least the noise), and persistent, not an event: above its own image's local level
               by more than L x that threshold (or between L and 4 L x it) in at least k of the images, for one of
               the levels L = 1 .. 16 (LEVELS). k comes from the images' occupancies in that window (fraction of
               pixels in it: dark electrons and events): the smallest k for which chance coincidences give
               < FALSE_SPOTS / 18 hot pixels on the CCD (Poisson-binomial tail x active pixels); k >= 2. The windows
               L..4L leave out most event pixels (their charge is broad), so few quiet images are enough for a
               faint hot pixel; images too busy for every window give no hot pixels (logged). Grouped into connected spots; 'seen in' = number of images in
               which the pixel is above the threshold
  cold pixels  the same residual below -5 x the spread, below by as much in >= k images (k as for hot pixels, from
               the images' occupancies below -threshold)
Images are grouped per ACM by size and binning (rows x columns, NPBIN x NSBIN): any images of the same size form
one composite, whatever their type or number of skips (e.g. Image 2, Image 4, Image 5, 1-skip or 500-skip).
Outputs (out_dir/defects/<module>_ACM<id>_<rows>x<cols>_<binning>/): Image_4's PNG / log, defects.csv (the defect log: one line per
defect), composite_EXT<n>.fits, and one interactive page per module (defect_map.html: per CCD the composite with
the defects marked, a clickable defect list; Image_4's figure and log), plus defects/index.html.
"""
import glob
import io
import math
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # superWADERs/
for _p in ("commissioning", "clustering", "engine"):          # superWADERs' own modules first on the path
    _p = os.path.join(SW_DIR, _p)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
try:
    import natsort  # noqa: F401
except ImportError:                                        # Image_4 imports natsorted; a natural sort is enough here
    import types
    _ns = types.ModuleType("natsort")
    _ns.natsorted = lambda seq: sorted(seq, key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))])
    sys.modules["natsort"] = _ns
from lsm_defects import Image_4 as I4   # noqa: E402

_CTI_Y = I4.calculate_cti_y      # restored for images with a vertical overscan
TYPES = [("bright column", "#ff3b30"), ("dark column", "#3a86ff"), ("hot pixel", "#ffbe0b"), ("cold pixel", "#00c2a8")]
FALSE_SPOTS = 0.1          # expected hot pixels per CCD from chance coincidences of events (sets k, find_defects)
LEVELS = tuple(2 ** (i / 2) for i in range(9))      # 1 .. 16 x the threshold, in steps of sqrt(2)


def persistence_needed(occ, n_pix, false_spots=FALSE_SPOTS):
    """Smallest k (>= 2) of the images in which a pixel must be above the threshold so that, for a normal pixel
    (above it in image i with probability occ[i], independently), n_pix x P(K >= k) <= false_spots; None if even
    all images are not enough (too busy, or a single image)."""
    dist = np.array([1.0])
    for p in occ:
        dist = np.convolve(dist, [1.0 - p, p])
    tail = np.cumsum(dist[::-1])[::-1]               # tail[k] = P(K >= k)
    for k in range(2, len(occ) + 1):
        if n_pix * tail[k] <= false_spots:
            return k
    return None


def set_geometry(path, autoconfig_args=()):
    """Set the scripts' geometry constants from one image (autoconfig geometry + header). Returns them."""
    import native_pipeline as NP
    img = NP.read_image(path, autoconfig_args)
    h0, geo = img["h0"], img["geo"]
    nrow = img["exts"][0][1].shape[0]
    try:
        yovs = min(nrow, int(math.ceil(float(h0["PHYSROW"]) / float(h0.get("NPBIN", 1) or 1))))
    except (KeyError, TypeError, ValueError):
        yovs = nrow
    g = dict(overscan_start=int(geo["ccd_cols"]), prescan=int(geo["n_cols_prescan"]), y_overscan_start=int(yovs))
    apply_geometry(g, nrow)
    return g, h0


def apply_geometry(g, nrow):
    """Set Image_4's constants; without >= 2 rows of vertical overscan its parallel CTI (calculate_cti_y) has
    nothing to measure and returns nan (instead of failing)."""
    for k, v in g.items():
        setattr(I4, k, v)
    I4.calculate_cti_y = _CTI_Y if g["y_overscan_start"] < nrow - 2 else (lambda rm: (float("nan"), float("nan")))


def image_type(path, h0):
    m = re.search(r"Image_(\d+)", os.path.basename(path))
    if m:
        return f"Image{m.group(1)}"
    return f"{h0.get('NDCM', '?')}skip_{h0.get('NPBIN', 1)}x{h0.get('NSBIN', 1)}"


def background(img, act, size=(3, 31)):
    """Local level of img: running median over size = (rows, columns) around each pixel (outside the active area:
    the active area's median). Follows the steep rise of the dark charge along the rows and slow changes along a
    row; a defect of a few pixels does not move it."""
    from scipy import ndimage
    x = np.where(act, img, np.nanmedian(img[act]))
    return ndimage.median_filter(x, size=size, mode="nearest")


def find_defects(comp, noise, subs, geo, nsig=5.0):
    """Defects of one CCD on its composite median comp [ADU] (noise [ADU]; subs: the pedestal-subtracted images).
    Returns (mask uint8 with codes 1..4 = TYPES, rows of the log, info: threshold, occupancies, k needed)."""
    from scipy import ndimage
    h, w = comp.shape
    a0, a1 = geo["prescan"], min(geo["overscan_start"], w)
    mask = np.zeros((h, w), np.uint8)
    rows = []
    with redirect_stdout(io.StringIO()):
        high, low, traps, idx = I4.find_col_traps(comp)
    colmed = np.ma.median(np.ma.masked_invalid(comp[:, a0:a1]), axis=0).filled(np.nan)

    def col_z(img):              # Image_4's column statistic of one image: (column median - running median) / running MAD
        m = np.ma.masked_invalid(np.asarray(img, float))
        m[:, :a0] = np.ma.masked
        m[:, a1:] = np.ma.masked
        cm = np.ma.median(m, axis=0)
        med, mad = I4.running_median_absolute_deviation(cm, window_size=I4.running_window)
        z = (cm - med) / np.ma.maximum(mad, 1e-9)
        return np.ma.filled(z.astype(float), np.nan)
    zs = [col_z(s) for s in subs]
    ncol = max(a1 - a0, 1)

    def cols_persistent(sign):   # columns beyond L x Image_4's 6 MAD in >= k images (k from the column occupancies)
        ok, rule = np.zeros(w, bool), []
        for lev in LEVELS:
            hits = [np.nan_to_num(sign * z) > lev * I4.col_threshold for z in zs]
            occ = [float(np.mean(x[a0:a1])) for x in hits]
            k = persistence_needed(occ, ncol, FALSE_SPOTS / len(LEVELS))
            rule.append((lev, k, occ, np.inf))
            if k:
                ok |= np.sum(hits, axis=0) >= k
        return ok, rule
    col_hi, col_rule_hi = cols_persistent(1)
    col_lo, col_rule_lo = cols_persistent(-1)
    cand = [int(c) for c in idx if a0 <= c < a1]
    n_cand = len(cand)           # Image_4's columns, kept only if bright / dark in enough of the single images
    kept = [c for c in cand if (col_hi[c] if float(colmed[c - a0]) > float(high[c]) else col_lo[c])]
    run = []                     # adjacent flagged columns of one kind: one defect (column range), the largest excess
    for c in kept + [None]:
        if c is not None:
            v = float(colmed[c - a0])
            bright = v > float(high[c])
            mask[:, c] = 1 if bright else 2
            ex = v - float((high[c] + low[c]) / 2)
            if run and c == run[-1][0] + 1 and bright == run[-1][1]:
                run.append((c, bright, ex))
                continue
        if run:
            pk = max(run, key=lambda r: abs(r[2]))
            rows.append(("bright column" if pk[1] else "dark column", int(pk[0]), -1, h * len(run), pk[2],
                         f"columns {run[0][0]}-{run[-1][0]}" if len(run) > 1 else "composite"))
        run = [(c, bright, ex)] if c is not None else []
    act = np.zeros((h, w), bool)
    act[:, a0:a1] = True
    if True:
        # persistent local excess: the composite minus its local level (running median: follows dark-charge or
        # light gradients) above 5 x the spread of that residual over the active area, and above the local level of
        # its image by as much in >= k images, k from the images' occupancies (persistence_needed): not events
        bg = background(comp, act)
        res = comp - bg
        spread = 1.4826 * float(np.nanmedian(np.abs(res[act] - np.nanmedian(res[act]))))
        thr = nsig * max(spread, noise if noise and np.isfinite(noise) else 0.0, 1e-6)
        r_img = [s - background(s, act) for s in subs]
        npix = int(act.sum())

        def persistent(sign):            # pixels beyond L x thr in >= k_L images for some level L; the rule per level
            ok, n_at, rule = np.zeros((h, w), bool), np.zeros((h, w), np.int32), []
            for lev, top in [(lv, np.inf) for lv in LEVELS] + [(lv, 4 * lv) for lv in LEVELS]:
                hits = [(sign * r > lev * thr) & (sign * r <= top * thr) for r in r_img]
                occ = [float(np.mean(x[act])) for x in hits]
                k = persistence_needed(occ, npix, FALSE_SPOTS / (2 * len(LEVELS)))
                rule.append((lev, k, occ, top))
                if k:
                    n = np.sum(hits, axis=0)
                    new = (n >= k) & ~ok
                    n_at[new] = n[new]
                    ok |= new
            return ok, n_at, rule
        hot_ok, n_hot, rule_hi = persistent(1)
        cold_ok, n_cold, rule_lo = persistent(-1)
        info = dict(col_rule=col_rule_hi, col_cand=n_cand, col_kept=len(kept), thr=thr, rule_hi=rule_hi, rule_lo=rule_lo, occ=rule_hi[0][2], need=any(r[1] for r in rule_hi),
                    need_lo=any(r[1] for r in rule_lo))
        for code, sel in ((3, act & (res > thr) & hot_ok & (mask == 0)),
                          (4, act & (res < -thr) & cold_ok & (mask == 0))):
            lab, n = ndimage.label(sel, structure=np.ones((3, 3), bool))
            for k, sl in enumerate(ndimage.find_objects(lab), start=1):
                pix = lab[sl] == k
                rr, cc = np.nonzero(pix)
                r0, c0 = int(rr[0] + sl[0].start), int(cc[0] + sl[1].start)
                vals = comp[sl][pix]
                pk = int(np.argmax(np.abs(vals)))
                rp, cp = int(rr[pk] + sl[0].start), int(cc[pk] + sl[1].start)
                mask[sl][pix] = code
                rows.append((TYPES[code - 1][0], cp, rp, int(pix.sum()), float(vals[pk]),
                             f"{int((n_hot if code == 3 else n_cold)[rp, cp])} of {len(subs)} images"))
    return mask, rows, info


class _ActiveRegionPedestal:
    """For images without an overscan: Image_4's overscan pedestal and noise taken from the active region
    (columns prescan..), its overscan_start set to the image width, CTI X / Y n/a. Restored on exit."""
    def __init__(self, ncol, prescan):
        self.ncol, self.pre = ncol, prescan

    def __enter__(self):
        self.saved = {m: dict(subtract_pedestal=m.subtract_pedestal, robust_noise_from_overscan=m.robust_noise_from_overscan,
                              calculate_cti_x=m.calculate_cti_x, calculate_cti_y=m.calculate_cti_y,
                              overscan_start=m.overscan_start) for m in (I4,)}
        pre = self.pre
        for m in (I4,):
            orig_noise = self.saved[m]["robust_noise_from_overscan"]
            m.overscan_start = self.ncol
            m.robust_noise_from_overscan = lambda sub, _start, clip_sigma=5.0, max_iter=2, f=orig_noise: f(sub, pre, clip_sigma, max_iter)
            m.calculate_cti_x = m.calculate_cti_y = lambda med: (float("nan"), float("nan"))
        def ped(data, clip_sigma=5.0, max_iter=2):
            act = np.asarray(data, float)[:, pre:]
            row_med = np.nanmedian(act, axis=1)
            for _ in range(max_iter):                          # as the scripts: clipped row median (here: active region)
                r = act - row_med[:, None]
                mad = max(1.4826 * float(np.nanmedian(np.abs(r))), 1e-8)
                new = np.nanmedian(np.where(np.abs(r) <= clip_sigma * mad, act, np.nan), axis=1)
                row_med = np.where(np.isfinite(new), new, row_med)
            return np.asarray(data, float) - row_med[:, None]
        I4.subtract_pedestal = ped
        return self

    def __exit__(self, *exc):
        for m, d in self.saved.items():
            for k, v in d.items():
                setattr(m, k, v)


def run(images, out_dir, mods_of, autoconfig_args=(), log=print, nsig=5.0, redo=None, write_index=True):
    """--defect-map on the images (all images of the campaign). redo: the new image paths - only the groups that
    contain one are made again (the others keep their folder and group.json); None = every group. Each group writes
    defects/<group>/group.json (per CCD: defect counts by type, noise, whether the persistence rules could be met),
    read by overview(). Returns summary lines."""
    import packet_report as PR
    from astropy.io import fits
    base = os.path.join(out_dir, "defects")
    os.makedirs(base, exist_ok=True)
    groups = {}
    for p in images:
        geo, h0 = set_geometry(p, autoconfig_args)
        acm = h0.get("ACMID") or PR.acm_of(p)
        hd = fits.getheader(p, 1)                         # images of the same size and binning go together,
        size = f"{hd['NAXIS2']}x{hd['NAXIS1']}_{h0.get('NPBIN', 1) or 1}x{h0.get('NSBIN', 1) or 1}"   # whatever their type
        groups.setdefault((int(acm) if acm else 0, size), []).append((p, geo, h0))
    lines = [f"defect maps (--defect-map; images grouped by ACM, size and binning; group script Image_4 + "
             f"defect log; {base}/index.html):"]
    index = []
    for (acm, itype), items in sorted(groups.items()):
        files = [p for p, _, _ in items]
        kinds = sorted({image_type(p, h) for p, _, h in items})
        geo, h0 = items[0][1], items[0][2]
        from astropy.io import fits as _f
        apply_geometry(geo, _f.getdata(files[0], 1).shape[0])
        module = mods_of(acm, None, None)
        name = f"{module}_ACM{acm}_{itype}" if not module.startswith("ACM") else f"ACM{acm}_{itype}"
        if redo is not None and not any(p in redo for p in files) and os.path.exists(os.path.join(base, name, "group.json")):
            lines.append(f"  {name}: {len(files)} image(s), no new image: kept")
            continue
        d = os.path.join(base, name)
        os.makedirs(d, exist_ok=True)
        log(f"  defects {name}: {len(files)} image(s)")
        out = io.StringIO()
        nc = fits.getdata(files[0], 1).shape[1]
        has_ovs = geo["overscan_start"] < nc - 2
        script_note = None if has_ovs else ("no overscan in these images: Image_4 ran with its pedestal (row "
                                            "median) and noise from the active region (noise includes the dark charge); "
                                            "CTI X / Y n/a")
        from contextlib import nullcontext
        try:
            t_img = float(h0.get("EXPOSURE", 0) or 0)
            try:
                from datetime import datetime
                t_img += (datetime.fromisoformat(str(h0["DATEEND"])) - datetime.fromisoformat(str(h0["DATEINI"]))).total_seconds()
            except (KeyError, ValueError):
                pass
            argv = ["Image_4", "--files", *files, "--output_dir", d, "--module", name, "--minutes-per-image",
                    f"{max(t_img / 60, 1e-3):.4f}", "--binning-x", str(h0.get("NSBIN", 1) or 1), "--binning-y", str(h0.get("NPBIN", 1) or 1)]
            old = sys.argv
            sys.argv = argv
            with redirect_stdout(out), (nullcontext() if has_ovs else _ActiveRegionPedestal(nc, geo["prescan"])):
                I4.main()
            sys.argv = old
        except Exception as e:                             # the scripts failed: report, the defect log still follows
            sys.argv = old if "old" in dir() else sys.argv
            script_note = f"Image_4 not run: {type(e).__name__}: {e}"
        with fits.open(files[0]) as hh:                    # CCD extensions of this ACM (usually 4)
            nch = sum(1 for h in hh[1:] if h.data is not None and np.ndim(h.data) == 2)
        try:
            with redirect_stdout(out):
                subs = [[] for _ in range(nch)]
                for p in files:
                    with fits.open(p) as hd:
                        for ch in range(nch):
                            raw = np.array(hd[ch + 1].data, float)
                            if has_ovs:
                                subs[ch].append(I4.subtract_pedestal(raw))
                            else:                          # no overscan: median of the active region per row
                                subs[ch].append(raw - np.median(raw[:, geo["prescan"]:], axis=1, keepdims=True))
                comps = [np.median(np.stack(s), axis=0) for s in subs]
                if has_ovs:
                    noises = [round(float(np.median([I4.robust_noise_from_overscan(x, geo["overscan_start"]) for x in s])), 1) for s in subs]
                else:
                    noises = [round(float(np.median([1.4826 * np.median(np.abs(x[:, geo["prescan"]:] - np.median(x[:, geo["prescan"]:])))
                                                     for x in s])), 2) for s in subs]
        except Exception as e:                             # report and go on with the other groups
            lines.append(f"  {name}: FAILED ({type(e).__name__}: {e})")
            continue
        tabs = [dict(id="map", label="Defect map", kind="defects", per={}, logs={})]     # one viewer, the CCDs as frames
        ccd_list = []
        csv_rows = []
        group_ccds = []
        for ch in range(nch):
            ext = ch + 1
            ccd = h0.get(f"CH{ch}") or f"ch{ch}"
            mask, rows, finfo = find_defects(comps[ch], float(noises[ch]), subs[ch], geo, nsig)
            occ_txt = ", ".join(f"{100 * o:.2g} %" for o in finfo["occ"])
            def rule_txt(rule, s, sgn):  # the loosest level for each number of images, plus the best band
                first = {}
                for lev, k, _, top in rule:
                    if k and not np.isfinite(top) and k not in first:
                        first[k] = lev
                out = ", ".join(f">= {k} images {s} {sgn}{lev * finfo['thr']:.2g} ADU" for k, lev in sorted(first.items(), reverse=True))
                band = next(((lev, top, k) for lev, k, _, top in rule if k and np.isfinite(top)), None)
                if band:
                    out += (f"; or >= {band[2]} images between {sgn}{band[0] * finfo['thr']:.2g} and "
                            f"{sgn}{band[1] * finfo['thr']:.2g} ADU")
                return out
            persist = (f"{rule_txt(finfo['rule_hi'], 'above', '')} (of {len(files)}; < {FALSE_SPOTS} chance hot pixels per CCD)" if finfo["need"] else
                       f"none identifiable: {len(files)} image(s) too {'few' if len(files) < 2 else 'busy'} (pixels above "
                       f"{finfo['thr']:.3g} ADU per image: {occ_txt}); chance coincidences of events would look the same")
            first_c = {}
            for lev, k, _, _ in finfo["col_rule"]:
                if k and k not in first_c:
                    first_c[k] = lev
            col_txt = (f"{finfo['col_kept']} of Image_4's {finfo['col_cand']} composite columns confirmed: bright / dark in "
                       + (", ".join(f">= {k} images beyond {6 * lev:.3g} MAD" for k, lev in sorted(first_c.items(), reverse=True))
                          if first_c else "none identifiable (images too few or too busy)")
                       + f" (of {len(files)}; a track along a column in one image is not a defect)")
            persist_lo = (f"{rule_txt(finfo['rule_lo'], 'below', '-')} (of {len(files)})" if finfo["need_lo"]
                          else "none identifiable (images too few or too busy)")
            fits.PrimaryHDU(comps[ch].astype(np.float32)).writeto(os.path.join(d, f"composite_EXT{ext}.fits"), overwrite=True)
            for r in rows:
                csv_rows.append([module, acm, ext, ccd, itype, *r])
            view, vmask, dec = comps[ch], mask, 1
            while view.size > PR.MAX_VIEW_PIXELS:
                h2, w2 = view.shape[0] // 2 * 2, view.shape[1] // 2 * 2
                view = view[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))
                vmask = vmask[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).max(axis=(1, 3))
                dec *= 2
            counts = {t: sum(1 for r in rows if r[0] == t) for t, _ in TYPES}
            group_ccds.append(dict(ext=f"EXT{ext}", ccd=ccd, counts=counts, noise_adu=float(noises[ch]),
                                   hot_rule=bool(finfo["need"]), cold_rule=bool(finfo["need_lo"]),
                                   columns_confirmed=finfo.get("col_kept"), columns_candidates=finfo.get("col_cand")))
            shown = sorted(rows, key=lambda r: -abs(r[4]))[:1500]
            key = f"ACM{acm}_EXT{ext}"
            ccd_list.append(dict(key=key, label=f"EXT{ext} · {ccd}", acm=acm, ext=f"EXT{ext}", ccd=ccd))
            tabs[0]["logs"][key] = [
                f"composite median of {len(files)} image(s) of {itype} (rows x columns _ binning); types: {', '.join(kinds)}; "
                f"noise {noises[ch]} ADU",
                f"geometry: prescan {geo['prescan']}, first overscan column {geo['overscan_start']}, vertical overscan from row {geo['y_overscan_start']}",
                f"hot pixels: {persist}; occupancy per image (pixels above the threshold, mostly events): {occ_txt}; "
                f"k chosen for < {FALSE_SPOTS} chance hot pixels on the CCD"] \
                + [f"{t}: {n}" for t, n in counts.items()] + ([f"view binned {dec}x{dec}; the defect list uses full-resolution positions"] if dec > 1 else [])
            tabs[0]["per"][key] = dict(
                          title=f"EXT{ext} {ccd}: composite median of {len(files)} image(s)", unit="ADU", w=int(view.shape[1]),
                          h=int(view.shape[0]), pixels=PR.enc(view), mask=PR.enc(vmask, "u1"),
                          types=[dict(name=t, color=c, n=counts[t]) for t, c in TYPES],
                          defects=[[r[0], int(r[1] // dec), int(r[2] // dec) if r[2] >= 0 else -1, r[3], r[4], r[5]] for r in shown],
                          defects_more=max(0, len(rows) - len(shown)),
                          numbers=[["images", str(len(files))], [f"noise ({'overscan' if has_ovs else 'active region'})", f"{noises[ch]} ADU"],
                                   ["column rule", col_txt], ["hot-pixel rule", persist, not finfo["need"]], ["cold-pixel rule", persist_lo]]
                          + [[t, str(n), t.endswith("column") and n > 0] for t, n in counts.items()])
        figs = []
        for png in sorted(glob.glob(os.path.join(d, "*.png"))):
            import base64
            figs.append(dict(caption=os.path.basename(png), png=base64.b64encode(open(png, "rb").read()).decode()))
        logs = []
        for lg in sorted(glob.glob(os.path.join(d, "*.log"))):
            logs += [f"--- {os.path.basename(lg)}"] + open(lg).read().splitlines()
        if figs or logs:
            tabs.append(dict(id="scripts", label="Image_4 plot", kind="figure",
                             data=dict(title="The group's Image_4: composite median of all images, column defects"
                                       + (f" ({script_note})" if script_note else ""), figures=figs), log=logs))
        if script_note:
            for lg in list(tabs[0]["logs"].values()) + [tb["log"] for tb in tabs[1:]]:
                lg.insert(0, script_note)
        import csv
        with open(os.path.join(d, "defects.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["module", "acm", "ext", "ccd", "image_type", "type", "column", "row (-1: whole column)", "size_pixels",
                        "value_adu", "seen_in"])
            w.writerows(csv_rows)
        size = PR.write_page(os.path.join(d, "defect_map.html"), name, f"{module} · ACM {acm} · {itype} · defect map",
                             f"defect map (run_superWADERs --defect-map), {len(files)} image(s)", tabs, ccd_list,
                             back="../index.html", tool="defects", module=module)
        nums = [[f"EXT{c + 1}", ", ".join(f"{t}: {sum(1 for r in csv_rows if r[2] == c + 1 and r[5] == t)}" for t, _ in TYPES)] for c in range(nch)]
        import json
        from datetime import datetime as _dt
        with open(os.path.join(d, "group.json"), "w") as fh:
            json.dump(dict(name=name, module=module, acm=int(acm) if acm else 0, image_type=itype, kinds=kinds,
                           images=[os.path.basename(p) for p in files], ccds=group_ccds, page=f"{name}/defect_map.html",
                           note=script_note, made=_dt.now().isoformat(timespec="seconds")), fh, indent=1)
        index.append((module, acm, itype, "4 CCDs", os.path.join(name, "defect_map.html"), [t["label"] for t in tabs], nums, size, []))
        lines.append(f"  {name}: {len(files)} image(s) ({', '.join(kinds)}), {len(csv_rows)} defects (defects.csv, defect_map.html"
                     + (", Image_4 PNG + log)" if not script_note else f"); {script_note}"))
    if write_index:
        PR.write_index(base, index, "Defect maps")
    return lines


FIELDS = {"bright_columns": ("bright columns", ""), "dark_columns": ("dark columns", ""), "hot_pixels": ("hot-pixel spots", ""),
          "cold_pixels": ("cold-pixel spots", ""), "images": ("images in the composite", ""), "noise_adu": ("noise", "ADU"),
          "group": ("group (ACM, size, binning)", ""), "columns_candidates": ("Image_4 composite columns", ""),
          "columns_confirmed": ("of them confirmed in the single images", "")}


def overview(camp, crit, path, back):
    """defects/index.html from every defects/<group>/group.json: one view per image size (group type), one row per
    CCD. Returns the views."""
    import json
    import campaign as CA
    sec = CA.section(crit, "defects")
    views = {}
    for gj in sorted(glob.glob(os.path.join(os.path.dirname(path), "*", "group.json"))):
        g = json.load(open(gj))
        for c in g["ccds"]:
            n = c["counts"]
            v = dict(bright_columns=n.get("bright column", 0), dark_columns=n.get("dark column", 0),
                     hot_pixels=n.get("hot pixel", 0), cold_pixels=n.get("cold pixel", 0), images=len(g["images"]),
                     noise_adu=c.get("noise_adu"), group=g["name"], columns_candidates=c.get("columns_candidates"),
                     columns_confirmed=c.get("columns_confirmed"))
            flags = []
            if not c.get("hot_rule"):
                flags.append(("not_identifiable", "warn", f"hot pixels not identifiable: {len(g['images'])} image(s) too "
                                                          f"{'few' if len(g['images']) < 2 else 'busy'} for the persistence rule",
                              "hot pixels not identifiable (too few / busy images)"))
            verdict, reasons, levels = CA.judge(v, sec, flags)
            views.setdefault(g["image_type"], []).append(dict(
                module=g["module"], acm=g["acm"], ext=c["ext"], ccd=c["ccd"], verdict=verdict, reasons=reasons, levels=levels,
                values=v, when=g.get("made", ""), links=[["defect map", g["page"]]],
                stamp=f"{g['name']}: {len(g['images'])} image(s), last {max(g['images'], default='')}"))
    out = [dict(name=k, ccds=v, note=f"images of {k}") for k, v in sorted(views.items())]
    views_out = CA.write_overview(path, dict(
        title="Defect maps", sub=CA.sub_line(camp), back=back, criteria=sec,
        fields={k: dict(label=a, unit=b) for k, (a, b) in FIELDS.items()},
        sections=[["Defects", ["bright_columns", "dark_columns", "hot_pixels", "cold_pixels"]],
                  ["Composite", ["images", "noise_adu", "columns_candidates", "columns_confirmed", "group"]]],
        cell=["bright_columns", "dark_columns", "hot_pixels", "cold_pixels"], views=out, review=dict(tool="defects"),
        columns=["bright_columns", "dark_columns", "hot_pixels", "cold_pixels", "images", "noise_adu", "group"],
        definitions=[("composite", "Median of all images of one ACM with the same size and binning (any type / skips)."),
                     ("columns", "Bright / dark columns: Image_4's rule on the composite, kept when also bright / dark in "
                                 "enough single images; adjacent columns are one defect."),
                     ("hot / cold pixels", "Spots above / below the local level by > --defect-sigma x the spread, present "
                                           "in enough images that chance coincidences are unlikely (< 0.1 per CCD)."),
                     ("defect map", "The composite with the defects marked, the defect list, Image_4's figure and log "
                                    "(defects/<group>/defect_map.html); defects.csv per group.")],
        flag_labels={"not_identifiable": "hot pixels not identifiable"}, csv_name="defects_overview"))
    return views_out
