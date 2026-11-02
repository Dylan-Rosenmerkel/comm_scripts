#!/usr/bin/env python3
"""A_GMX clusters written as panaSKImg cluster files (same trees, branches and conventions as the WADERs ClusterFinder).

A_GMX (clustering_tests/adaptive.py, engine 'gmx') runs on the calibrated image that panaSKImg saves for each CCD
(<stem>_EXT<n>_waders.fits.bz2, electrons). Its clusters are turned into pysimdamicm `Cluster` objects - the class the
ClusterFinder uses - so every branch (positions, widths, fitted STD, PCA, projections, seed, track endpoints,
pixel lists, ...) is computed by panaSKImg's own code, and written with pysimdamicm's `OutputDataManager` into
    panaSKImg_clustersRec_<stem>_ext<n>.root   (trees clustersRec, info, process_config)
`info` and `process_config` are copied from the image's panaSKImg file; clustersRec gets the usual branches plus
    agmx_cls, agmx_blend, agmx_ped, agmx_peakfrac, agmx_chi2, agmx_sig, agmx_regime   (vector per cluster;
  -999 = not computed, e.g. agmx_chi2 of clusters the mixture did not fit)
and the file gets a TNamed `agmx_config` (JSON: version, parameters, class and regime codes, crowded_mode: setting,
core fill, whether crowded mode ran and what it found - clustering_tests/crowded.py).
For electron-counting images (read noise < 0.35 e-) the few-electron candidate catalogue (clustering_tests/
fe_candidates.py: every 2-20 e- group, masks, background expectation per class) is written to
OUTDIR/fe_candidates/<stem>_ext<n>_fe_candidates.csv / .json (run_AGMX --no-fe-candidates to skip).

Conventions (checked against panaSKImg files): pixels_x / pixels_y are 1-based indices of the calibrated image;
pixels_E [keV] = the charge [e-] the cluster takes from the pixel x e2eV (shared pixels are split between clusters,
so Energy = sum(pixels_E) is the A_GMX energy); Energy_PS / Energy_AVG = sums of the full pedestal-subtracted /
raw-average pixel values [ADU] of the cluster's pixels, as panaSKImg; sigma_seed_eV, qmin_cluster_seed, readout
times, ccd, nfile, RUNID are those of the image's panaSKImg file.

    python src/agmx_root.py RUN_DIR [-o OUTDIR] [--curve gain_vs_charge.json or gain_vs_charge/] [--anchor comb] [-j N]
       (or bin/run_AGMX, or run_WADERs --clusterer agmx/both)
       RUN_DIR: a run_WADERs output folder (needs root/, json/<stem>/final/, the *_waders.fits.bz2 images)
       --curve: g(n) curve(s) (gain_vs_charge) -> clusters of the gain-corrected image
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CT = os.path.join(HERE, "..", "clustering_tests")
for p in (HERE, CT):
    if p not in sys.path:
        sys.path.insert(0, p)
import naming  # noqa: E402

AGMX_VERSION = "2026-10-28"     # crowded mode (clustering_tests/crowded.py)
CLS_CODES = {"x-ray": 1, "x-ray diffuse": 2, "low-E": 3, "muon": 4, "worm": 5, "alpha": 6, "big deposit": 7,
             "artifact-row": 8, "artifact-column": 9, "extended": 10, "glow": 11}
REGIME_CODES = {"sparse": 0, "crowded": 1, "unmodelled": 2, "serial": 3, "line": 4}     # serial: serial-register streak, line: muon line component (crowded mode)
NOT_COMPUTED = -999.0     # agmx_* flag that could not be computed (a real ped / chi2 / blend is never this low)


class _Time:
    def __init__(self, t): self.t = float(t)
    def timestamp(self): return self.t


class _Hits:
    """the attributes of panaSKImg's image object that pysimdamicm's Cluster reads"""
    __attr_to_clusterize__ = "_pixels"


def read_reference(ref_root, ext, final_json=None):
    """per-image constants from the image's panaSKImg cluster file (from its clusters; if it has none, from its info
    tree: ccd = idccd, nfile, readout times, sigma_seed_eV = EXT<n>_pedestal_sigma_row_mean_ADU / gain x e2eV and
    qmin_cluster_seed = seed threshold (ClusterFinder spr_threshold of the final JSON, default 30) x sigma_seed)"""
    import ROOT
    tf = ROOT.TFile.Open(ref_root)
    try:
        return _read_reference(tf, ext, final_json)
    finally:
        tf.Close()


def _read_reference(tf, ext, final_json):
    info = tf.Get("info"); info.GetEntry(0)
    pc = tf.Get("process_config"); pc.GetEntry(0)
    t = tf.Get("clustersRec")                   # absent when panaSKImg wrote no cluster tree for this CCD
    if t:
        t.GetEntry(0)
    runid = int(t.RUNID) if t else int(getattr(info, "RUNID", 0))
    r = dict(RUNID=runid, gain=float(getattr(info, f"ext{ext}_gainext")), e2eV=float(pc.e2eV))
    mu = getattr(info, f"EXT{ext}_pedestal_mu_row_ADU")
    r["mu_row"] = np.array(list(mu[0]), float) if len(mu) else None
    n = int(t.Nclusters) if t else 0
    for k in ("sigma_seed_eV", "qmin_cluster_seed", "readout_start", "readout_end"):
        r[k] = float(getattr(t, k)[0]) if n else None
    for k in ("ccd", "nfile"):
        r[k] = int(getattr(t, k)[0]) if n else None
    if not n:                                   # no WADERs clusters on this CCD: the same values from 'info'
        g = lambda b: getattr(info, b) if info.GetBranch(b) else None
        r["ccd"] = int(g("idccd")) if g("idccd") is not None else int(ext)
        r["nfile"] = int(g("nfile")) if g("nfile") is not None else 0
        for k in ("readout_start", "readout_end"):
            r[k] = float(g(k)) if g(k) is not None else None
        sm = g(f"EXT{ext}_pedestal_sigma_row_mean_ADU")
        if sm is not None:
            r["sigma_seed_eV"] = float(sm) / r["gain"] * r["e2eV"]
            k_seed = 30.0
            if final_json and os.path.exists(final_json):
                thr = json.load(open(final_json)).get("process", {}).get("ClusterFinder", {}).get("spr_threshold", {})
                v = thr.get(f"EXT{ext}") if isinstance(thr, dict) else None
                if isinstance(v, list) and len(v) >= 3 and v[2] is not None:
                    k_seed = float(v[2])
            r["qmin_cluster_seed"] = k_seed * r["sigma_seed_eV"] * 1e-3      # keV, as in the files
    return r


def valid_mask(shape, final_json, nsbin=1):
    """active area as in the clustering study: rows from id_row_start + 3, columns from n_cols_prescan + 2 to the
    first overscan column - 2, 2-px border (from the final panaSKImg JSON of the CCD). ccd_cols is in UNBINNED
    columns (panaSKImg: overscan starts at ccd_cols / NSBIN of the image)"""
    with open(final_json) as fh:
        im = json.load(fh)["input"]["image"]
    H, W = shape
    a1 = min(int(im["ccd_cols"] // max(int(nsbin), 1)), W)
    v = np.zeros(shape, bool)
    v[im["id_row_start"] + 3:H - 2, im["n_cols_prescan"] + 2:a1 - 2] = True
    return v


def build_clusters(clusters, ref, adu, gain, H, W):
    """pysimdamicm Cluster objects from A_GMX clusters (with '_pix'), properties computed by pysimdamicm"""
    from pysimdamicm.io import data_formats as DF
    DF.u.e2eV = ref["e2eV"]
    DF.u.calibration = gain
    DF.u.ADC2e = gain
    base = _Hits()
    base.event, base.isdata, base.image_header = ref["RUNID"], True, {}
    if ref["ccd"] is not None: base.ccd = ref["ccd"]
    if ref["nfile"] is not None: base.n_image = ref["nfile"]
    if ref["sigma_seed_eV"] is not None: base.sigma0 = ref["sigma_seed_eV"]
    if ref["qmin_cluster_seed"] is not None: base.q_min_cls_seed = ref["qmin_cluster_seed"] * DF.u.keV
    if ref["readout_start"] is not None:
        base.start_readout = True; base.start = _Time(ref["readout_start"]); base.end = _Time(ref["readout_end"])
    base.image_mean_compressed_pedestal_subtracted = adu
    if ref["mu_row"] is not None and ref["mu_row"].size == H:
        base.image_mean_compressed = adu + ref["mu_row"][:, None]
    out = []
    for i, c in enumerate(clusters, start=1):
        ys, xs, q = c["_pix"]
        if len(ys) == 0:
            continue
        h = _Hits(); h.__dict__.update(base.__dict__)
        h.x_pixels, h.y_pixels = np.asarray(xs, int), np.asarray(ys, int)
        h.z_pixels = -np.ones(len(ys)); h.Edep_pixels = np.asarray(q, float) * DF.u.e2eV
        cl = DF.Cluster(i, h, np.ones(len(ys), bool))
        try:
            cl.get_cluster_properties(get_fitted_STD=True)
        except Exception:
            continue
        cl.cluster_id = len(out) + 1
        cl.agmx_cls = int(CLS_CODES.get(c.get("cls"), 0))
        for k in ("blend", "ped", "peakfrac", "chi2", "sig"):
            v = c.get(k, np.nan)
            setattr(cl, f"agmx_{k}", float(v) if v is not None and np.isfinite(v) else NOT_COMPUTED)
        cl.agmx_regime = int(REGIME_CODES.get(c.get("regime"), -1))
        out.append(cl)
    closest(out)
    return out


def closest(cls):
    """closest_cls_* as panaSKImg's ClusterFinder.get_closest_cluster: the nearest other seeded cluster (centre
    distance, |dX|, |dY|, 1-based index), with panaSKImg's own rule for excluding 'itself' (PosX != PosX_i AND
    PosY != PosY_i). That rule also drops a neighbour whose PosX or PosY is exactly equal (same row/column and the
    same floating-point value), so for same-row/column neighbours the result depends on the last bits of the
    positions - in panaSKImg files as here."""
    if len(cls) <= 1:
        for c in cls:
            for k in ("closest_cls_dist", "closest_cls_dist_X", "closest_cls_dist_Y", "closest_cls_id"):
                setattr(c, k, float(-1.0))
        return
    PosX = np.array([c.PosX for c in cls]); PosY = np.array([c.PosY for c in cls])
    hasseed = np.array([getattr(c, "has_seed", 1) for c in cls], bool)
    for i, c in enumerate(cls):
        if getattr(c, "has_seed", 1):
            d = np.sqrt((PosX - c.PosX) ** 2 + (PosY - c.PosY) ** 2)
            m = np.logical_and(np.logical_and(PosX != c.PosX, PosY != c.PosY), hasseed)
            d = np.where(m, d, np.inf); j = int(np.argmin(d))
            c.closest_cls_dist, c.closest_cls_dist_X = float(d[j]), float(abs(PosX[j] - c.PosX))
            c.closest_cls_dist_Y, c.closest_cls_id = float(abs(PosY[j] - c.PosY)), float(j + 1)
        else:
            c.closest_cls_dist = c.closest_cls_dist_X = c.closest_cls_dist_Y = c.closest_cls_id = float(-1.0)


def write_root(out_path, cls, ref, ref_root, params):
    import ROOT
    from pysimdamicm.io.data_formats import OutputDataManager
    om = OutputDataManager(out_path)
    if cls:
        om.fill_tree("clustersRec", ref["RUNID"], cls, isdata=True)
    src = ROOT.TFile.Open(ref_root)
    om.tfile.cd()
    clones = [src.Get(n).CloneTree(-1) for n in ("info", "process_config") if src.Get(n)]   # noqa: F841 (kept alive until written)
    ROOT.TNamed("agmx_config", json.dumps(params)).Write()
    om.close()
    src.Close()
    return len(cls)


def process(waders_fits, ref_root, final_json, out_path, ext, curve=None, agmx_kw=None, anchor="comb", fe_candidates=None):
    """one CCD of one image -> one panaSKImg-format cluster file; returns the number of clusters.
    curve: g(n) model of this CCD (gain_vs_charge) -> the image is converted ADU -> electrons with it before
    clustering; anchor as gain_vs_charge.apply ('comb': the comb's g(n) throughout, 'fit': g(n) scaled to the
    image's fitted gain at low charge)"""
    from astropy.io import fits
    import adaptive as AD
    with fits.open(waders_fits) as hdul:
        cal = hdul[0].data.astype(float)
        nsbin = int(hdul[0].header.get("NSBIN", 1) or 1)
    ref = read_reference(ref_root, ext, final_json)
    adu = cal * ref["gain"]
    if curve is not None:
        import gain_vs_charge as GVC
        Q = GVC.electrons_with_anchor(adu, curve, ref["gain"], anchor).reshape(cal.shape)
    else:
        Q = cal
    valid = valid_mask(Q.shape, final_json, nsbin)
    Q = np.where(valid, Q, 0.0)
    kw = dict(agmx_kw or {})
    try:
        bg = AD.AM.background_map(Q, valid)
    except Exception:
        bg = None
    clusters, info = AD.adaptive_cluster(Q, valid, bg, engine="gmx", return_pixels=True, return_info=True, **kw)
    clusters = [c for c in clusters if np.isfinite(c.get("E", np.nan)) and "_pix" in c]
    if fe_candidates is None:
        fe_candidates = os.environ.get("AGMX_FE_CANDIDATES", "1") != "0"
    if fe_candidates and bg is not None and info.get("regime") != "unmodelled" and info.get("noise", 9) < 0.35:
        # electron-counting image: the few-electron candidate catalogue (fe_candidates.py) next to the cluster files
        import fe_candidates as FE
        cands, summ = FE.candidates(Q, valid, bg, clusters)
        summ.update(source_image=os.path.basename(waders_fits), ext=int(ext), gain_corrected=curve is not None)
        stem = os.path.basename(out_path)[len("panaSKImg_clustersRec_"):-len(".root")]
        FE.write(cands, summ, os.path.join(os.path.dirname(out_path), "fe_candidates", stem))
    cls = build_clusters(clusters, ref, adu, ref["gain"], *Q.shape)
    params = dict(method="A_GMX (clustering_tests/adaptive.py engine='gmx')", version=AGMX_VERSION,
                  gain_corrected=curve is not None, gain_anchor=anchor if curve is not None else None, parameters=kw, cls_codes=CLS_CODES, regime_codes=REGIME_CODES, not_computed=NOT_COMPUTED,
                  crowded_mode=crowded_summary(info),
                  source_image=os.path.basename(waders_fits), reference=os.path.basename(ref_root),
                  recommended_selection="agmx_blend < 0.02 and agmx_ped < 10 (and fSTD/sigma cut of the analysis)")
    return write_root(out_path, cls, ref, ref_root, params)


def crowded_summary(info):
    """crowded-mode record for agmx_config (adaptive.adaptive_cluster info): the setting, the core fill that decided
    it, and what ran"""
    import adaptive as AD
    r = dict(setting=AD.CROWDED, core_fill=info.get("core_fill"), on=bool(info.get("crowded_mode", False)))
    if r["on"]:
        bo = info.get("bg_override") or {}
        bc = info.get("bg_columns") or {}
        r.update(hot_columns=int(bc.get("columns", 0)), glow=int(info.get("n_glow", 0)))
        r.update(shapes=bool(info.get("crowded_shapes", False)), lines=int(info.get("n_lines", 0)), topology_splits=int(info.get("n_topology_splits", 0)))
        r.update(bg_override=bool(bo.get("applied")), bad_rows=int(bo.get("bad_rows", 0)), cells_changed=int(bo.get("cells_changed", 0)),
                 straight_tracks=int(info.get("n_hints", 0)), streaks=int(info.get("n_streaks", 0)), linked=int(info.get("n_linked", 0)))
    return r


def tasks_for_run(run_dir, root_dir="root"):
    """(waders fits, reference root, final json, ext) for every CCD of a run_WADERs output folder"""
    out = []
    for ref in sorted(glob.glob(os.path.join(run_dir, root_dir, "panaSKImg_clustersRec_*_ext*.root"))):
        b = os.path.basename(ref)[len("panaSKImg_clustersRec_"):-len(".root")]
        stem, ext = b.rsplit("_ext", 1)
        if not ext.isdigit():
            continue
        fits_f = next(iter(glob.glob(os.path.join(run_dir, f"{stem}_EXT{ext}_waders.fits*"))), None)
        fj = os.path.join(run_dir, "json", stem, "final", f"{stem}_EXT{ext}.json")
        if fits_f and os.path.exists(fj):
            out.append((fits_f, ref, fj, int(ext), stem))
    return out


def curve_file(curve, stem):
    """the g(n) file for an image: curve is a gain_vs_charge JSON (used for every image) or a run_WADERs
    gain_vs_charge/ folder, where 'image_<stem>' (per-image curve) is preferred if per_image, then the curve of the
    image's gain period ('module_<ACM id>_period<k>', run_WADERs --gain-periods auto; the one whose image list holds
    this image), else 'module_<ACM id>' (pooled curve)"""
    if curve is None or os.path.isfile(curve):
        return curve
    path, per_image = (curve[:-len(":per-image")], True) if curve.endswith(":per-image") else (curve, False)
    acm = naming.acm_id(stem)
    if per_image and os.path.isfile(os.path.join(path, f"image_{stem}_gain_vs_charge.json")):
        return os.path.join(path, f"image_{stem}_gain_vs_charge.json")
    for f in sorted(glob.glob(os.path.join(path, f"module_{acm or stem}_period*_gain_vs_charge.json"))):
        with open(f) as fh:
            if any(naming.image_stem(i) == stem for i in json.load(fh).get("images", [])):
                return f
    f = os.path.join(path, f"module_{acm or stem}_gain_vs_charge.json")
    return f if os.path.isfile(f) else None


def _job(a):
    """one CCD; returns (stem, ext, n clusters, error or None, note)"""
    fits_f, ref, fj, ext, stem, outdir, curve, anchor = a
    out_path = os.path.join(outdir, os.path.basename(ref))
    note = ""
    try:
        cpath = curve_file(curve, stem)
        import gain_vs_charge as GVC
        model = GVC.load_model(cpath, ext) if cpath else None
        if curve is not None and model is None:
            # as gain_vs_charge.apply, which copies such CCDs unchanged: the panaSKImg calibration is kept
            note = " (no reliable g(n) curve for this CCD: fitted gain, as in root_gaincorr)"
        n = process(fits_f, ref, fj, out_path, ext, curve=model, anchor=anchor)
        return stem, ext, n, None, note
    except Exception as e:
        return stem, ext, 0, f"{type(e).__name__}: {e}", note


def run(run_dir, outdir, curve=None, anchor="comb", jobs=4, log=print):
    """A_GMX on every CCD of a run_WADERs output folder -> outdir/panaSKImg_clustersRec_<stem>_ext<n>.root and
    outdir/root_files.txt. curve: None, a g(n) JSON, or a gain_vs_charge/ folder (append ':per-image' for the
    per-image curves). Returns (files, failed [(stem, ext, reason)])."""
    os.makedirs(outdir, exist_ok=True)
    tasks = [t + (outdir, curve, anchor) for t in tasks_for_run(run_dir)]
    failed = []
    if tasks:
        from multiprocessing import get_context
        with get_context("spawn").Pool(max(1, jobs), maxtasksperchild=2) as p:
            for stem, ext, n, err, note in p.imap_unordered(_job, tasks):
                log(f"  {stem} EXT{ext}: " + (f"{n} clusters{note}" if err is None else f"FAILED ({err})"))
                if err is not None:
                    failed.append((stem, ext, err))
    files = sorted(glob.glob(os.path.join(outdir, "panaSKImg_clustersRec_*.root")))
    with open(os.path.join(outdir, "root_files.txt"), "w") as fh:
        fh.write("".join(f + "\n" for f in files))
    return files, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--outdir", default=None, help="default: RUN_DIR/root_agmx (root_agmx_gaincorr with --curve)")
    ap.add_argument("--curve", default=None, help="g(n): a gain_vs_charge JSON, or a run_WADERs gain_vs_charge/ folder "
                    "(module curve per image; FOLDER:per-image for the per-image curves)")
    ap.add_argument("--anchor", choices=("comb", "fit"), default="comb", help="with --curve: as run_WADERs --gain-anchor")
    ap.add_argument("-j", "--jobs", type=int, default=4)
    a = ap.parse_args()
    outdir = a.outdir or os.path.join(a.run_dir, "root_agmx_gaincorr" if a.curve else "root_agmx")
    if not tasks_for_run(a.run_dir):
        sys.exit(f"agmx_root: no CCDs found in {a.run_dir} (needs root/, json/<stem>/final/ and the *_waders.fits images)")
    files, failed = run(a.run_dir, outdir, a.curve, a.anchor, a.jobs, log=lambda m: print(m, flush=True))
    print(f"agmx_root: {len(files)} cluster files in {outdir}" + (f"; {len(failed)} CCD(s) failed" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
