"""One image -> its campaign record, in a worker process (run_superWADERs: -j images at a time).

image_job((path, out_dir, process_image keyword arguments, options)) runs native_pipeline.process_image (fits,
clustering chain, cluster files as asked) and makes the record kept in the campaign (campaign.py), so that every
report can be written again later without fitting the image again:
    stem, path, acm, setting, meta      header (json-safe), ACM, image setting (skips, binning)
    summary                             the native summary without its arrays
    char                                the front end's record of the image (characterise.py), if any
    metrics, crosstalk                  ccd_metrics.image_metrics (baseline, columns, glow, saturation, crosstalk)
    fit_cards                           {ext number: the dark-current fit card of the pages (histogram, model, profiles)}
    packet                              options['packet']: {ext number: [fit card, viewer, log]} for the packet pages
    packet_numbers                      options['packet']: packet_report.image_numbers
    sr                                  options['sr'] = (threshold, min length): {ext: sr_images.image_part item}
Returns (path, record or None, error text or None).
"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
SW = os.path.dirname(HERE)
for _d in ("commissioning", "clustering", "engine"):
    _d = os.path.join(SW, _d)
    if _d not in sys.path:
        sys.path.insert(0, _d)

ARRAYS = ("ps", "mu_row", "sig_row", "e_img")


def strip(s):
    """The native summary without arrays (also the per-output cluster tables)."""
    out = dict(s, ccds={})
    for e, c in s["ccds"].items():
        c = {k: v for k, v in c.items() if k not in ARRAYS}
        for k, v in list(c.items()):
            if isinstance(v, dict) and "table" in v:
                c[k] = {kk: vv for kk, vv in v.items() if kk != "table"}
        out["ccds"][e] = c
    return out


def header(path):
    from astropy.io import fits
    import campaign as CA
    h = fits.getheader(path)
    return CA.jsonable({k: v for k, v in h.items() if k and k not in ("COMMENT", "HISTORY", "")
                        and isinstance(v, (str, int, float, bool))})


def make_record(path, s, kw, opts):
    import campaign as CA
    import ccd_metrics as CM
    import naming
    import native_pipeline as NP
    import packet_report as PR
    args = kw.get("autoconfig_args", ())
    meta = header(path)
    stem = s["stem"]
    acm = meta.get("ACMID") or naming.acm_id(stem)
    rec = dict(stem=stem, path=os.path.abspath(path), acm=int(acm) if acm else 0, meta=meta, setting=CA.setting_of(meta),
               summary=None, char=kw.get("characterisation"), version=getattr(NP, "VERSION", ""))
    img = NP.read_image(path, args)
    geo = img["geo"]
    # dark-current fit cards (and the packet viewer) from the arrays of this run
    try:
        per_ext, _ = PR.image_data(path, args, s)
    except Exception as e:                            # the record still holds everything else
        per_ext = {}
        rec["fit_cards_error"] = f"{type(e).__name__}: {e}"
    rec["fit_cards"] = {str(n): summ for n, (summ, _, _) in per_ext.items()}
    if opts.get("packet"):
        rec["packet"] = {str(n): [summ, view, log] for n, (summ, view, log) in per_ext.items()}
        rec["packet_numbers"] = {str(k): v for k, v in PR.image_numbers(s, meta).items()}
    gains = {e: c.get("gain") for e, c in s["ccds"].items() if "gain" in c}
    try:
        rec["metrics"], rec["crosstalk"] = CM.image_metrics(path, gains, (geo["n_cols_prescan"], geo["ccd_cols"]))
    except Exception as e:
        rec["metrics"], rec["crosstalk"] = {}, {}
        rec["metrics_error"] = f"{type(e).__name__}: {e}"
    if opts.get("sr"):
        import sr_images as SR
        thr, ml = opts["sr"]
        rec["sr"] = SR.image_part(path, s, geo, meta, thr, ml, {n: summ for n, (summ, _, _) in per_ext.items()})
        rec["sr_settings"] = [thr, ml]
    rec["summary"] = strip(s)
    return rec


def image_job(task):
    path, out, kw, opts = task
    import native_pipeline as NP
    s, err = NP._job((path, out, dict(kw, keep_arrays=True)))
    if err:
        return path, None, err
    try:
        return path, make_record(path, s, kw, opts), None
    except Exception as e:                            # the cluster files are written; the record failed
        return path, None, f"{path}: record: {type(e).__name__}: {e}\n{traceback.format_exc()}"
