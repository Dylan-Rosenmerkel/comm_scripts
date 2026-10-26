"""Re-cluster the 29 AMtest Am-241 images (EXT4) with MFD+G and the MFD+G hybrid merges.
Images: runs_29img/gaincorr WADERs calibrated images; electrons with the per-gain-period (epoch) g(n) curves
(epoch1 = images 21-40, epoch2 = images 41-50), as for the WADERs 'gaincorr' run. Output: clusters_<stem>_ext<N>.npz"""
import ast, glob, json, os, sys, argparse
import numpy as np
from astropy.io import fits
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.expanduser("~/Documents/comm_scripts_data/session_2026-10-23_clustering")
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import agent_methods as AM
import methods2 as M2
import mhc as MH
import gain_vs_charge as GVC
import autoconfig_panaSKImg as A
AMT = os.path.expanduser("~/Documents/JH1_Testing/AMTEst")
G = os.path.join(AMT, "runs_29img", "gaincorr", "WADERs_analysis_AMtest_29")
EPOCH = {1: os.path.join(G, "gain_vs_charge_epochs", "epoch1_img21-40_gain_vs_charge.json"),
         2: os.path.join(G, "gain_vs_charge_epochs", "epoch2_img41-50_gain_vs_charge.json")}
BASELINE = os.environ.get("BASELINE", "0") == "1"
OCC = float(os.environ.get("OCC", "0.3"))       # max local fraction of occupied (> 1.5 e-) pixels, 31 x 31 px box
OUT = os.path.join(HERE, f"clusters_occ{OCC:g}")


def model_for(num, ext):
    e = json.load(open(EPOCH[1 if num <= 40 else 2]))["extensions"][f"EXT{ext}"]["model"]
    return ast.literal_eval(e) if isinstance(e, str) else e


def load(stem, ext):
    cal = fits.open(os.path.join(G, f"{stem}_EXT{ext}_waders.fits.bz2"))[0].data.astype(float)
    import ROOT
    tf = ROOT.TFile.Open(os.path.join(G, "root", f"panaSKImg_clustersRec_{stem}_ext{ext}.root"))
    info = tf.Get("info"); info.GetEntry(0); gfit = float(getattr(info, f"ext{ext}_gainext")); tf.Close()
    num = int(stem.split("_")[-1])
    adu = cal * gfit
    if BASELINE:
        # the saved WADERs image still holds the slow baseline drift that panaSKImg removes internally: subtract a
        # running median along each row (5 x 61 px; ~80 % of the pixels hold 0 e-, so the median sits on the 0-e peak)
        from scipy import ndimage
        adu = adu - ndimage.median_filter(adu, size=(5, 61), mode="nearest")
    Q = GVC.electrons_of_adu(adu, model_for(num, ext)).reshape(cal.shape)
    h = fits.open(os.path.join(AMT, stem + ".fz"))
    ref = json.loads(A.REFERENCE_JSON)
    _, cfg, _, _ = A.geometry(h[0].header, h, ref["input"]["image"], argparse.Namespace(ccd_cols=None))
    valid = np.zeros_like(Q, bool)
    valid[cfg["id_row_start"] + 3:Q.shape[0] - 2, cfg["n_cols_prescan"] + 2:min(cfg["ccd_cols"], Q.shape[1]) - 2] = True
    return np.where(valid, Q, 0.0), valid


def job(a):
    stem, ext = a
    fn = os.path.join(OUT, f"{stem}_ext{ext}.npz")
    if os.path.exists(fn): return
    Q, valid = load(stem, ext)
    from scipy import ndimage
    # occupancy mask: where the CCD is covered by a continuous sheet of charge there is nothing to cluster
    occ = ndimage.uniform_filter(((Q > 1.5) & valid).astype(float), 31, mode="constant") / np.maximum(
        ndimage.uniform_filter(valid.astype(float), 31, mode="constant"), 1e-3)
    valid = valid & (occ < OCC)
    Q = np.where(valid, Q, 0.0)
    bg = AM.background_map(Q, valid)
    cl = AM.mf_detect(Q, valid, bg)[0]
    res = {"MFD+G": cl,
           "hybrid v1": M2.mfdg_hybrid(Q, valid, bg, cl=[dict(c) for c in cl], ext_big=14),
           "hybrid v2": M2.mfdg_hybrid(Q, valid, bg, cl=[dict(c) for c in cl], f_big=0.2, ext_big=14),
           "MHC": MH.mhc(Q, valid, bg)}
    out = dict(valid_frac=float(valid.sum()), frac_gt20=float((Q[valid] > 20).mean()), lam=float(np.median(bg.lam[valid])), s=float(np.median(bg.s[valid])))
    for k, v in res.items():
        v = [c for c in v if np.isfinite(c.get("E", np.nan))]
        key = k.replace(" ", "_").replace("+", "")
        out[f"{key}_E"] = np.array([c["E"] for c in v]); out[f"{key}_sxy"] = np.array([c["sxy"] for c in v])
        out[f"{key}_n"] = np.array([c.get("n_merged", 1) for c in v])
    np.savez(fn, **out)
    print(stem[-12:], ext, f"lam {out['lam']:.3f} s {out['s']:.2f} frac>20e {out['frac_gt20']:.4f}", {k: len(v) for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    stems = sorted({os.path.basename(f).split("_EXT")[0] for f in glob.glob(os.path.join(G, "*_EXT4_waders.fits.bz2"))})
    exts = [int(e) for e in os.environ.get("EXTS", "4").split(",")]
    with Pool(int(os.environ.get("NPROC", 8))) as p:
        p.map(job, [(s, e) for s in stems for e in exts], chunksize=1)
