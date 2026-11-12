"""Image-based commissioning metrics of every CCD of one averaged image (computed once per image, stored in the
campaign record, judged against the criteria when the overview pages are written).

Per CCD (pedestal = overscan median per row, or the active region's when there is no overscan; e- with the CCD's
fitted gain):
  baseline    overscan level per row: robust rms of row-to-row changes, the largest step [ADU and in units of that
              rms, so that it means the same at 1 skip and at 1000], the drift over the image [ADU]; overscan noise [ADU]
  columns     dark charge per column from the fraction of 0e- pixels (robust to events); hot columns = more than
              5 robust sigma and HOT_COL_MIN_E above a running median over 31 columns (gradients and glow are not
              columns); glow index = dark charge of the last 10 % of the active columns (next to the amplifier) / the
              median
  saturation  pixels at the extension's maximum value (counted when >= 3)
Per image (the CCDs of one module are read at the same time):
  crosstalk   victim pixel ADU vs source charge [e-] where the source has > XT_SOURCE_E and the victim no event of
              its own (> 20 e-): slope with an intercept (the victim's dark-charge level and pedestal residual are
              not crosstalk), in e- per e-, with its error
These read the image file once more (the engine's arrays are not reused here; see docs/superWADERs_efficiency_notes.md).
"""
import math

import numpy as np

HOT_COL_MIN_E = 0.05          # a hot column has at least this much extra dark charge per pixel [e-]
XT_SOURCE_E = 200.0           # source pixels used for crosstalk [e-]
XT_MIN_NSIG = 5.0             # a crosstalk coefficient is significant at >= this many sigma


def regions(h0, shape):
    """(first active column, first overscan column) from the header (PRESCAN, PHYSCOL - NCOLPRE + PRESCAN; binned)."""
    ncol = shape[1]
    nsbin = int(h0.get("NSBIN", 1) or 1)
    pre = int(h0.get("PRESCAN", 0) or 0) // nsbin
    if all(k in h0 for k in ("PHYSCOL", "NCOLPRE", "PRESCAN")):
        ovs = int(round((h0["PHYSCOL"] - h0["NCOLPRE"] + h0["PRESCAN"]) / nsbin))
    else:
        ovs = ncol
    return min(pre, ncol - 1), min(max(ovs, pre + 1), ncol)


def image_metrics(path, gains, geo=None):
    """({ext: metrics}, {'EXTa->EXTb': [coefficient, error]}) of one image. gains: {ext: gain [ADU/e-] or None}
    (CCDs without a gain get only the ADU metrics). geo: (first active column, first overscan column) if known."""
    from astropy.io import fits
    from scipy.ndimage import median_filter
    out, charge, gain_of = {}, {}, {}
    with fits.open(path) as hd:
        h0 = hd[0].header
        exts = [(i, np.asarray(h.data, float)) for i, h in enumerate(hd)
                if i > 0 and h.data is not None and np.ndim(h.data) == 2]
    for n, raw in exts:
        e = f"EXT{n}"
        a0, o0 = geo if geo else regions(h0, raw.shape)
        has_ovs = o0 < raw.shape[1] - 2
        base = np.median(raw[:, o0:], axis=1) if has_ovs else np.median(raw[:, a0:o0], axis=1)
        ps = raw - base[:, None]
        m = {}
        d = np.diff(base)
        if d.size:
            m["baseline_row_rms_adu"] = float(1.4826 * np.median(np.abs(d - np.median(d))))
            m["baseline_max_step_adu"] = float(np.max(np.abs(d)))
            rms = 1.4826 * np.median(np.abs(d - np.median(d)))
            if rms > 0:                               # the largest step in units of the row-to-row scatter
                m["baseline_step_nsigma"] = float(np.max(np.abs(d - np.median(d))) / rms)
        if base.size > 2:
            m["baseline_drift_adu"] = float(np.polyfit(np.arange(base.size), base, 1)[0] * base.size)
        if has_ovs:
            ov = ps[:, o0:]
            m["overscan_noise_adu"] = float(1.4826 * np.median(np.abs(ov - np.median(ov))))
        top = raw.max()
        nsat = int(np.sum(raw >= top))
        m["saturated_pixels"] = nsat if nsat >= 3 else 0
        g = gains.get(e)
        if g:
            q = ps[:, a0:o0] / g
            charge[e], gain_of[e] = q, g
            f0 = np.mean(q < 0.5, axis=0)
            with np.errstate(divide="ignore"):
                lam_c = -np.log(np.clip(f0, 1e-6, 1))
            med = float(np.median(lam_c))
            trend = median_filter(lam_c, size=31, mode="nearest")
            resid = lam_c - trend
            spread = 1.4826 * np.median(np.abs(resid)) + math.sqrt(max(med, 1e-6) / q.shape[0])
            hot = (resid > 5 * spread) & (resid > HOT_COL_MIN_E) if med > 1e-4 else np.zeros(lam_c.size, bool)
            m["hot_columns"] = int(hot.sum())
            m["hot_column_list"] = [int(c + a0) for c in np.flatnonzero(hot)][:20]
            m["usable_columns_pct"] = float(100 * (1 - hot.mean()))
            k = max(int(0.1 * lam_c.size), 3)
            if med > 1e-4:
                m["glow_index"] = float(np.median(lam_c[-k:]) / med)
        out[e] = m
    xt = {}
    for s_e, qs in charge.items():
        sel = qs > XT_SOURCE_E
        if sel.sum() < 30:
            continue
        for v_e, qv_e in charge.items():
            if v_e == s_e or qv_e.shape != qs.shape:
                continue
            use = sel & ~(qv_e > 20)
            if use.sum() < 30:
                continue
            x, y = qs[use], qv_e[use] * gain_of[v_e]
            xm, ym = x.mean(), y.mean()
            sxx = float(np.sum((x - xm) ** 2))
            if sxx <= 0:
                continue
            c = float(np.sum((x - xm) * (y - ym)) / sxx)
            err = float(np.std(y - ym - c * (x - xm)) / math.sqrt(sxx))
            xt[f"{s_e}->{v_e}"] = [c / gain_of[v_e], err / gain_of[v_e]]
    return out, xt
