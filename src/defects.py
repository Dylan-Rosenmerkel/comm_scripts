"""Defect maps (run_superWADERs --defect-map): median composites and column defects with the group's scripts, plus
a defect log and an interactive defect map.

The group's scripts (D. Venegas-Vargas, LSM_Module_Testing_Analysis/src; copies in src/lsm_defects/,
utils/sync_from_lsm_analysis.py) run unchanged:
  Image_4  composite median of all images of a module and image type (pedestal per row from the overscan,
           column defects by a running median / MAD of the column medians, event rate per pixel per day): its PNG
           and log, as when run by hand;
  Image_2  per image: column defects / traps and CTI estimates ('Serial Register Defect Maps'): PNG and log.
Their geometry constants (first overscan column, prescan, first vertical-overscan row) are set from each image's
header (autoconfig's geometry) instead of the full-frame values written in the scripts.
Added here, per CCD, on the composite median:
  columns      bright / dark columns: Image_4's find_col_traps (median of the column above / below the running
               median +- 6 MAD), with the column's excess over the running median
  hot pixels   composite median minus its smooth local level (32 x 32 block medians) above 5 x the robust spread of
               that residual (at least the overscan noise), and above its own image's local level by as much in at
               least half of the images (all of them with < 3 images): persistent, not an event; grouped into
               connected spots; 'seen in' = number of images in which the pixel is above that threshold
  cold pixels  the same residual below -5 x the spread
Images are grouped per ACM by size and binning (rows x columns, NPBIN x NSBIN): any images of the same size form
one composite, whatever their type or number of skips (e.g. Image 2, Image 4, Image 5, 1-skip or 500-skip).
Outputs (out_dir/defects/<module>_ACM<id>_<rows>x<cols>_<binning>/): the scripts' PNG / log files, defects.csv (the defect log: one line per
defect), composite_EXT<n>.fits, and one interactive page per module (defect_map.html: per CCD the composite with
the defects marked, a clickable defect list; the scripts' figures and logs), plus defects/index.html.
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
if HERE not in sys.path:
    sys.path.insert(0, HERE)
try:
    import natsort  # noqa: F401
except ImportError:                                        # Image_4 imports natsorted; a natural sort is enough here
    import types
    _ns = types.ModuleType("natsort")
    _ns.natsorted = lambda seq: sorted(seq, key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))])
    sys.modules["natsort"] = _ns
from lsm_defects import Image_2 as I2   # noqa: E402
from lsm_defects import Image_4 as I4   # noqa: E402

_CTI_Y = {I2: I2.calculate_cti_y, I4: I4.calculate_cti_y}      # restored for images with a vertical overscan
TYPES = [("bright column", "#ff3b30"), ("dark column", "#3a86ff"), ("hot pixel", "#ffbe0b"), ("cold pixel", "#00c2a8")]


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
    """Set the scripts' constants; without >= 2 rows of vertical overscan their parallel CTI (calculate_cti_y) has
    nothing to measure and returns nan (instead of failing)."""
    for mod in (I2, I4):
        for k, v in g.items():
            setattr(mod, k, v)
        mod.calculate_cti_y = _CTI_Y[mod] if g["y_overscan_start"] < nrow - 2 else (lambda rm: (float("nan"), float("nan")))


def image_type(path, h0):
    m = re.search(r"Image_(\d+)", os.path.basename(path))
    if m:
        return f"Image{m.group(1)}"
    return f"{h0.get('NDCM', '?')}skip_{h0.get('NPBIN', 1)}x{h0.get('NSBIN', 1)}"


def background(img, act, block=32):
    """Smooth local level of img over the active area: median of block x block cells, interpolated back."""
    from scipy import ndimage
    h, w = img.shape
    x = np.where(act, img, np.nan)
    nb_r, nb_c = max(1, h // block), max(1, w // block)
    cells = np.full((nb_r, nb_c), np.nan)
    for i in range(nb_r):
        for j in range(nb_c):
            v = x[i * h // nb_r:(i + 1) * h // nb_r, j * w // nb_c:(j + 1) * w // nb_c]
            if np.isfinite(v).any():
                cells[i, j] = np.nanmedian(v)
    cells = np.where(np.isfinite(cells), cells, np.nanmedian(cells))
    return ndimage.zoom(cells, (h / nb_r, w / nb_c), order=1, mode="nearest")[:h, :w]


def find_defects(comp, noise, subs, geo, nsig=5.0):
    """Defects of one CCD on its composite median comp [ADU] (noise [ADU]; subs: the pedestal-subtracted images).
    Returns (mask uint8 with codes 1..4 = TYPES, rows of the log)."""
    from scipy import ndimage
    h, w = comp.shape
    a0, a1 = geo["prescan"], min(geo["overscan_start"], w)
    mask = np.zeros((h, w), np.uint8)
    rows = []
    with redirect_stdout(io.StringIO()):
        high, low, traps, idx = I4.find_col_traps(comp)
    colmed = np.ma.median(np.ma.masked_invalid(comp[:, a0:a1]), axis=0).filled(np.nan)
    for c in idx:
        if not (a0 <= c < a1):
            continue
        v = float(colmed[c - a0])
        bright = v > float(high[c])
        mask[:, c] = 1 if bright else 2
        rows.append(("bright column" if bright else "dark column", int(c), -1, h, v - float((high[c] + low[c]) / 2), "composite"))
    act = np.zeros((h, w), bool)
    act[:, a0:a1] = True
    if True:
        # persistent local excess: the composite minus its smooth local level (block medians: follows dark-charge or
        # light gradients) above 5 x the spread of that residual over the active area, and above the local level of
        # its image by as much in at least half of the images (all of them with < 3 images): not an event
        bg = background(comp, act)
        res = comp - bg
        spread = 1.4826 * float(np.nanmedian(np.abs(res[act] - np.nanmedian(res[act]))))
        thr = nsig * max(spread, noise if noise and np.isfinite(noise) else 0.0, 1e-6)
        n_hot = np.zeros((h, w), np.int32)
        for s in subs:
            n_hot += (s - background(s, act)) > thr
        need = len(subs) if len(subs) < 3 else int(math.ceil(0.5 * len(subs)))
        for code, sel in ((3, act & (res > thr) & (n_hot >= need) & (mask == 0)),
                          (4, act & (res < -thr) & (mask == 0))):
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
                             f"{int(n_hot[rp, cp])} of {len(subs)} images" if code == 3 else "composite"))
    return mask, rows


def run(images, out_dir, mods_of, autoconfig_args=(), log=print, nsig=5.0):
    """--defect-map on the run's images. Returns summary lines."""
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
    lines = [f"defect maps (--defect-map; images grouped by ACM, size and binning; group scripts Image_4 / Image_2 + "
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
        d = os.path.join(base, name)
        os.makedirs(d, exist_ok=True)
        log(f"  defects {name}: {len(files)} image(s)")
        out = io.StringIO()
        nc = fits.getdata(files[0], 1).shape[1]
        has_ovs = geo["overscan_start"] < nc - 2
        script_note = None
        try:
            if not has_ovs:
                raise RuntimeError("no overscan in these images: the group scripts need one (pedestal and noise from "
                                   "the overscan); skipped, composite made with the active-region row median")
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
            with redirect_stdout(out):
                I4.main()
                for p in files:
                    I2.process_fz_and_plot_all_channels(p, f"{name}_{Path(p).name.split('.')[0]}", output_dir=Path(d))
            sys.argv = old
        except Exception as e:                             # the scripts failed: report, the defect log still follows
            sys.argv = old if "old" in dir() else sys.argv
            script_note = f"group scripts not run: {type(e).__name__}: {e}"
        try:
            with redirect_stdout(out):
                subs = [[] for _ in range(4)]
                for p in files:
                    with fits.open(p) as hd:
                        for ch in range(4):
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
        for ch in range(4):
            ext = ch + 1
            ccd = h0.get(f"CH{ch}") or f"ch{ch}"
            mask, rows = find_defects(comps[ch], float(noises[ch]), subs[ch], geo, nsig)
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
            shown = sorted(rows, key=lambda r: -abs(r[4]))[:1500]
            key = f"ACM{acm}_EXT{ext}"
            ccd_list.append(dict(key=key, label=f"EXT{ext} · {ccd}"))
            tabs[0]["logs"][key] = [
                f"composite median of {len(files)} image(s) of {itype} (rows x columns _ binning); types: {', '.join(kinds)}; "
                f"noise {noises[ch]} ADU",
                f"geometry: prescan {geo['prescan']}, first overscan column {geo['overscan_start']}, vertical overscan from row {geo['y_overscan_start']}"] \
                + [f"{t}: {n}" for t, n in counts.items()] + ([f"view binned {dec}x{dec}; the defect list uses full-resolution positions"] if dec > 1 else [])
            tabs[0]["per"][key] = dict(
                          title=f"EXT{ext} {ccd}: composite median of {len(files)} image(s)", unit="ADU", w=int(view.shape[1]),
                          h=int(view.shape[0]), pixels=PR.enc(view), mask=PR.enc(vmask, "u1"),
                          types=[dict(name=t, color=c, n=counts[t]) for t, c in TYPES],
                          defects=[[r[0], int(r[1] // dec), int(r[2] // dec) if r[2] >= 0 else -1, r[3], r[4], r[5]] for r in shown],
                          defects_more=max(0, len(rows) - len(shown)),
                          numbers=[["images", str(len(files))], ["noise (overscan)", f"{noises[ch]} ADU"]]
                          + [[t, str(n), t.endswith("column") and n > 0] for t, n in counts.items()])
        figs = []
        for png in sorted(glob.glob(os.path.join(d, "*.png"))):
            import base64
            figs.append(dict(caption=os.path.basename(png), png=base64.b64encode(open(png, "rb").read()).decode()))
        logs = []
        for lg in sorted(glob.glob(os.path.join(d, "*.log"))):
            logs += [f"--- {os.path.basename(lg)}"] + open(lg).read().splitlines()
        if figs or logs:
            tabs.append(dict(id="scripts", label="Group scripts", kind="figure",
                             data=dict(title="Image_4 (composite) and Image_2 (per image): their figures", figures=figs), log=logs))
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
                             f"defect map (run_superWADERs --defect-map), {len(files)} image(s)", tabs, ccd_list)
        nums = [[f"EXT{c + 1}", ", ".join(f"{t}: {sum(1 for r in csv_rows if r[2] == c + 1 and r[5] == t)}" for t, _ in TYPES)] for c in range(4)]
        index.append((module, acm, itype, "4 CCDs", os.path.join(name, "defect_map.html"), [t["label"] for t in tabs], nums, size, []))
        lines.append(f"  {name}: {len(files)} image(s) ({', '.join(kinds)}), {len(csv_rows)} defects (defects.csv, defect_map.html"
                     + (", Image_4/Image_2 PNG + log)" if not script_note else f"); {script_note}"))
    PR.write_index(base, index, "Defect maps")
    return lines
