"""Regression of the regional background (scratch): current vs regional A_GMX background on many real CCD images.
jobs: one image per module of every gn_lowcharge_study run (all CCDs), the 29 crowded Am-241 images (EXT4), L2 ACM101.
Per job -> reg/<tag>_<variant>.npz: cluster table + background diagnostics (per-row pedestal integer of the current
fit vs the regional one, spread of the cell dark charge along rows)."""
import glob, os, re, sys, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as HX
AM, AD, AR, GV = HX.AM, HX.AD, HX.AR, HX.GV
D = os.path.expanduser("~/Documents/comm_scripts_data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reg")

def load_run(run, stem, ext):
    from astropy.io import fits
    with fits.open(f"{run}/{stem}_EXT{ext}_waders.fits.bz2") as h: cal = h[0].data.astype(float); nsbin = int(h[0].header.get("NSBIN", 1) or 1)
    fj = glob.glob(f"{run}/json/{stem}/final/{stem}_EXT{ext}.json")[0]
    ref = AR.read_reference(f"{run}/root/panaSKImg_clustersRec_{stem}_ext{ext}.root", ext, fj)
    acm = re.search(r"_(1\d\d)_\d{8}_\d{6}", stem).group(1)
    cj = glob.glob(f"{run}/gain_vs_charge/module_{acm}_gain_vs_charge.json")
    curve = GV.load_model(cj[0], ext) if cj else None
    adu = cal * ref["gain"]
    Q = GV.electrons_with_anchor(adu, curve, ref["gain"], "comb").reshape(cal.shape) if curve else cal
    valid = AR.valid_mask(Q.shape, fj, nsbin)
    return np.where(valid, Q, 0.0), valid, curve is not None

def jobs():
    J = []
    for run in sorted(glob.glob(f"{D}/gn_lowcharge_study/WADERs_analysis_*")):
        stems = sorted({os.path.basename(f).split("_EXT")[0] for f in glob.glob(f"{run}/*_waders.fits.bz2")})
        first = {}
        for s in stems:
            acm = re.search(r"_(1\d\d)_\d{8}_\d{6}", s).group(1); first.setdefault(acm, s)
        for s in first.values():
            for f in sorted(glob.glob(f"{run}/{s}_EXT*_waders.fits.bz2")):
                e = int(re.search(r"_EXT(\d)_waders", f).group(1))
                if glob.glob(f"{run}/root/panaSKImg_clustersRec_{s}_ext{e}.root"):
                    J.append(("run", run, s, e, f"{os.path.basename(run)[15:40]}_{s[-26:]}_ext{e}"))
    sys.path.insert(0, os.path.expanduser("~/Documents/Software/comm_scripts/clustering_tests"))
    import am241_29img_recluster as RC
    for s in sorted({os.path.basename(f).split("_EXT")[0] for f in glob.glob(os.path.join(RC.G, "*_EXT4_waders.fits.bz2"))}):
        J.append(("am", None, s, 4, f"Am241_{s[-22:]}_ext4"))
    L2 = f"{D}/L2_101_20260922/WADERs_analysis_L2_101_20260922"
    for s in HX.STEM.values():
        for e in (1, 2, 4): J.append(("run", L2, s, e, f"L2_{s[-18:]}_ext{e}"))
    return J

def diag(Q, valid, bg, bgr):
    o_int_cur = np.round(bg.o[:, 0] - (bg.o[:, 0] - np.round(bg.o[:, 0])))
    d_o = bg.o[:, 0] - bgr.o[:, 0]
    lam_c = bgr.rows["regional"]
    rowspread = np.nanmax(lam_c, 1) - np.nanmin(lam_c, 1)
    return dict(ped_shift_max=float(np.nanmax(np.abs(d_o))), ped_shift_frac=float(np.mean(np.abs(d_o) > 0.5)),
                lam_med=float(np.median(bg.lam[valid])), lam_reg_med=float(np.median(bgr.lam[valid])),
                lam_rowspread_max=float(np.nanmax(rowspread)), lam_rowspread_med=float(np.nanmedian(rowspread)),
                mean_diff=float(np.mean((bg.mean - bgr.mean)[valid])), mean_absdiff=float(np.mean(np.abs(bg.mean - bgr.mean)[valid])),
                I0=-999 if bgr.rows.get("I0") is None else int(bgr.rows["I0"]), noise=float(np.median(bg.s[valid])))

def run_job(j):
    import bgv3, xsplit, streaks
    kind, run, stem, ext, tag = j
    os.makedirs(OUT, exist_ok=True)
    if all(os.path.exists(f"{OUT}/{tag}_{v}.npz") for v in ("base", "new")): return
    t0 = time.time()
    try:
        if kind == "am":
            import am241_29img_recluster as RC
            Q, valid = RC.load(stem, ext); gc = True
        else:
            Q, valid, gc = load_run(run, stem, ext)
        bg = AM.background_map(Q, valid)
        b3, rep = bgv3.regional_v3(Q, valid, bg)
        for v in ("base", "new"):
            fn = f"{OUT}/{tag}_{v}.npz"
            if os.path.exists(fn): continue
            t1 = time.time()
            if v == "new":
                xsplit.install(HX.GX); streaks.install(AD)
                cl, info = AD.adaptive_cluster(Q, valid, b3, engine="gmx", return_info=True, return_pixels=True)
                streaks.uninstall(AD); xsplit.uninstall(HX.GX)
                cl = [c for c in cl if np.isfinite(c.get("E", np.nan))]
                cl, nl = HX.link_tracks([c for c in cl if "_pix" in c]); cl = cl + []
            else:
                assert not getattr(HX.GX, "_xsplit", False) and not getattr(AD, "_streaks", False)
                cl, info = AD.adaptive_cluster(Q, valid, bg, engine="gmx", return_info=True)
                cl = [c for c in cl if np.isfinite(c.get("E", np.nan))]; nl = 0
            tab = {k: np.array([c.get(k, np.nan) for c in cl], float) for k in ("E", "sxy", "x", "y", "blend", "ped")}
            np.savez_compressed(fn, cls=np.array([c.get("cls", "") for c in cl]), regime=info.get("regime", ""), gaincorr=gc,
                                sec=time.time() - t1, rep=json.dumps(rep), n_linked=nl, n_streaks=info.get("n_streaks", 0), **tab)
        print(f"{tag}: v3 applied {rep['applied']} (bad rows {rep['bad_rows']}, cells {rep['cells_changed']}/{rep['ncells']}), {time.time() - t0:.0f} s", flush=True)
    except Exception as e:
        import traceback; print(f"{tag}: FAILED {e!r} {traceback.format_exc()[-300:]}", flush=True)

if __name__ == "__main__":
    J = jobs()
    hi = [j for j in J if "High_Temp" in j[4]]; J = [j for j in J if "High_Temp" not in j[4]] + hi    # slow ones last
    print(len(J), "jobs", flush=True)
    from multiprocessing import get_context
    with get_context("spawn").Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 10, maxtasksperchild=4) as p:
        list(p.imap_unordered(run_job, J, chunksize=1))
    print("ALLDONE", flush=True)
