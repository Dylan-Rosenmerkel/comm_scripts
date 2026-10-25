"""Compare clustering methods on real Fe-55 clusters and with injected clusters of known charge."""
import argparse, glob, json, os, sys
import numpy as np
from astropy.io import fits
from scipy import optimize
import ROOT
ROOT.gROOT.SetBatch(True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import gain_vs_charge as GVC
import autoconfig_panaSKImg as A
import cluster_methods as CM

DATA = "/Users/dylanrosenmerkel/Documents/LSM_Module_Testing_Analysis/Production_Modules_Data/2025-08-27-DM07_DM09_DM10_Low_Temp"
RUN = os.path.join(DATA, "gain_curve_study_frontside", "WADERs_analysis_Image5_DM09_DM10")
CURVES = json.load(open(os.path.expanduser("~/Documents/comm_scripts_data/session_2026-10-23_clustering/coherent_test/coherent_vs_incoherent.json")))
ST = {0: "20250827_182132_0", 1: "20250827_212852_1", 2: "20250828_003611_2"}
ACM = {"DM09": "109", "DM10": "106"}


def wader_sigma(stem, ext):
    """sigma panaSKImg used for its cluster thresholds (median per-row pedestal width), in e-, from its log."""
    import re
    log = open(os.path.join(RUN, "logs", f"{stem}_EXT{ext}_panaSKImg.log")).read()
    m = re.search(r"Sigma for SPR: ([0-9.eE+-]+) ADUs \(([0-9.eE+-]+) ADU/e-", log)
    return float(m.group(1)) / float(m.group(2))


def load(mod, img, ext):
    stem = f"avg_Image_5_Low_Temp_{ACM[mod]}_{ST[img]}"
    cal = fits.open(os.path.join(RUN, f"{stem}_EXT{ext}_waders.fits.bz2"))[0].data.astype(float)
    tf = ROOT.TFile.Open(os.path.join(RUN, "root", f"panaSKImg_clustersRec_{stem}_ext{ext}.root"))
    info = tf.Get("info"); info.GetEntry(0); gfit = float(getattr(info, f"ext{ext}_gainext")); tf.Close()
    model = CURVES[mod]["coherent"][f"EXT{ext}"]["model"]
    Q = GVC.electrons_of_adu(cal * gfit, model).reshape(cal.shape)
    h = fits.open(os.path.join(DATA, stem + ".fz"))
    ref = json.loads(A.REFERENCE_JSON)
    _, cfg, _, _ = A.geometry(h[0].header, h, ref["input"]["image"], argparse.Namespace(ccd_cols=None))
    valid = np.zeros_like(Q, bool)
    r0, c0, c1 = cfg["id_row_start"] + 3, cfg["n_cols_prescan"] + 2, min(cfg["ccd_cols"], Q.shape[1]) - 2
    valid[r0:Q.shape[0] - 2, c0:c1] = True
    act = Q[valid]
    z = act[np.abs(act) < 0.5]
    sigma = 1.4826 * np.median(np.abs(z - np.median(z)))
    quiet = act[act < 2.5]
    p0 = np.mean(np.abs(quiet) < 0.5) / max(np.mean(quiet < 2.5), 1e-9)
    lam = -np.log(max(p0, 1e-6))
    # dark charge per row (it grows along the readout): -ln P(0 e-) of the quiet pixels of each row,
    # smoothed over 15 rows; the scalar 'lam' is kept for printing
    lam_row = np.full(Q.shape[0], lam)
    for r in range(Q.shape[0]):
        v = Q[r][valid[r]]
        v = v[v < 2.5]
        if v.size > 50:
            lam_row[r] = -np.log(max(np.mean(np.abs(v) < 0.5), 1e-6))
    lam_row = np.convolve(np.pad(lam_row, 7, mode="edge"), np.ones(15) / 15, mode="valid")
    lam = lam_row
    Q = np.where(valid, Q, 0.0)
    return Q, valid, sigma, lam, wader_sigma(stem, ext)


def fit_ka(e):
    e = np.asarray([v for v in e if np.isfinite(v)]); e = e[(e > 4.5) & (e < 7.0)]
    if e.size < 30:
        return None
    lo, hi, r = 4.5, 7.0, 6.490 / 5.899
    def nll(p):
        mu, s, fb, fbg = p
        g = lambda m: np.exp(-0.5 * ((e - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        v = (1 - fbg) * ((1 - fb) * g(mu) + fb * g(mu * r)) + fbg / (hi - lo)
        return np.inf if np.any(v <= 0) else -np.sum(np.log(v))
    sel = e[(e > 5.0) & (e < 6.2)]
    res = optimize.minimize(nll, [np.median(sel), 0.08, 0.15, 0.05], method="L-BFGS-B",
                            bounds=[(5.0, 6.3), (0.02, 0.5), (0.0, 0.5), (1e-4, 0.9)])
    h = 1e-3
    c = (nll(res.x + [h, 0, 0, 0]) - 2 * nll(res.x) + nll(res.x - [h, 0, 0, 0])) / h ** 2
    return float(res.x[0]), float(1 / np.sqrt(c)) if c > 0 else np.nan, float(res.x[1]), int(e.size)


def match(found, y, x, r=2.0):
    best, bd = None, r
    for c in found:
        d = np.hypot(c["y"] - y, c["x"] - x)
        if d < bd:
            best, bd = c, d
    return best


def real_data(mod, img, ext):
    Q, valid, sigma, lam, sw = load(mod, img, ext)
    base, _ = CM.wader(Q, sw, 10, 30, 3.0, 8.5)          # WADERs' own thresholds (its sigma)
    out = dict(sigma=sigma, lam=float(np.mean(lam)), sigma_wader=sw, methods={})
    side = {"front": [c for c in base if c["sxy"] < 1], "back": [c for c in base if c["sxy"] >= 1]}
    for name, func in (
            ("WADERs (10s, 30s)", lambda cl: [c["E"] for c in cl]),
            ("fixed: pix 3 e-, seed 10 e-", None),
            ("topo S10 N2 rings0", None), ("topo S10 N2 rings1", None), ("topo S10 N2 rings2", None),
            ("topo S10 N1 rings1", None),
            ("aperture k3", lambda cl: CM.aperture(Q, lam, cl, k=3.0, valid=valid)),
            ("gauss fit", lambda cl: CM.gfit(Q, lam, sigma, cl, valid=valid)[0])):
        res = {}
        if name.startswith("topo") or name.startswith("fixed"):
            if name.startswith("fixed"):
                tc = CM.topo(Q, lam, 10.0, 3.0, 0, 3.0, 8.5)
            else:
                S, N, R = float(name.split()[1][1:]), float(name.split()[2][1:]), int(name.split()[3][5:])
                tc = CM.topo(Q, lam, S, N, R, 3.0, 8.5)
            for sd, cl in side.items():
                E = []
                for c in cl:
                    m = match(tc, c["y"], c["x"])
                    E.append(m["E"] if m else np.nan)
                res[sd] = fit_ka(E)
        else:
            for sd, cl in side.items():
                res[sd] = fit_ka(func(cl))
        out["methods"][name] = res
    return out


def injection(mod, img, ext, sxys=(0.3, 0.8, 1.4, 1.8), n_inj=120, seed=1):
    Q, valid, sigma, lam, sw = load(mod, img, ext)
    rng = np.random.default_rng(seed)
    busy = Q > 8.0                       # real clusters (dark-current pixels are allowed nearby)
    from scipy import ndimage
    busy = ndimage.binary_dilation(busy, iterations=8) | ~ndimage.binary_erosion(valid, iterations=10)
    ys, xs = np.nonzero(~busy)
    res = {}
    for s in sxys:
        idx = rng.permutation(ys.size)
        pos = []
        for k in idx:
            y, x = ys[k], xs[k]
            if all((y - py) ** 2 + (x - px) ** 2 > 20 ** 2 for py, px in pos):
                pos.append((y + rng.uniform(-0.5, 0.5), x + rng.uniform(-0.5, 0.5)))
            if len(pos) >= n_inj:
                break
        n_e = 1580
        Qi, counts = CM.inject(Q, rng, n_e, s, pos)
        found = {}
        wc, _ = CM.wader(Qi, sw, 10, 30)
        found["WADERs default (10s, 30s)"] = wc
        for N in (1, 2, 3, 5, 7):
            for R in (0, 1, 2):
                found[f"pix>={N} e- rings{R}"] = CM.topo(Qi, lam, 10, N, R)
        wdef = found["pix>=3 e- rings0"]
        sub = [match(wdef, y, x) for y, x in pos]
        ok = [c for c in sub if c]
        for k in (2.0, 3.0, 4.0):
            Ea = CM.aperture(Qi, lam, ok, k=k, valid=valid)
            found[f"aperture k{k:g}"] = [dict(c, E=e) for c, e in zip(ok, Ea) if np.isfinite(e)]
        Eg, Sg = CM.gfit(Qi, lam, sigma, ok, valid=valid)
        found["gauss fit"] = [dict(c, E=e) for c, e in zip(ok, Eg) if np.isfinite(e)]
        r = {}
        for name, fl in found.items():
            ratios = []
            for (y, x), ne in zip(pos, counts):
                m = match(fl, y, x)
                if m:
                    ratios.append(m["E"] / (ne * CM.E_PER_E))
            ratios = np.array(ratios)
            r[name] = dict(eff=len(ratios) / len(pos), med=float(np.median(ratios)) if ratios.size else np.nan,
                           rms=float(1.4826 * np.median(np.abs(ratios - np.median(ratios)))) if ratios.size else np.nan)
        res[str(s)] = r
    return dict(sigma=sigma, lam=float(np.mean(lam)), sigma_wader=sw, results=res)


if __name__ == "__main__" and os.environ.get("ONLY_INJECT"):
    out = json.load(open(os.path.join(HERE, "study_real.json"))) if os.path.exists(os.path.join(HERE, "study_real.json")) else {"real": {}}
    out["inject"] = {}
    for key in ("DM10 EXT4 img1", "DM10 EXT4 img0", "DM09 EXT4 img1", "DM09 EXT4 img0"):
        mod, ext, img = key.split()
        out["inject"][key] = injection(mod, int(img[3:]), int(ext[3:]))
        print("injection", key, f"dark charge {out['inject'][key]['lam']:.2f}", flush=True)
        for s_, r in out["inject"][key]["results"].items():
            print(f"  sigma_xy {s_} px")
            for name, v in r.items():
                print(f"     {name:26s} eff {v['eff']:.2f}  E/E_true {v['med']:.4f}  rms {v['rms']:.4f}")
    json.dump(out, open(os.path.join(HERE, "study.json"), "w"), indent=1)
    sys.exit(0)

if __name__ == "__main__":
    out = {"real": {}, "inject": {}}
    for mod in ("DM09", "DM10"):
        for ext in (1, 4):
            for img in (0, 1, 2):
                key = f"{mod} EXT{ext} img{img}"
                out["real"][key] = real_data(mod, img, ext)
                print(key, f"noise {out['real'][key]['sigma']:.3f} e-, WADERs sigma {out['real'][key]['sigma_wader']:.3f} e-, dark charge {out['real'][key]['lam']:.3f} e-/pix", flush=True)
                for m, rr in out["real"][key]["methods"].items():
                    print(f"   {m:22s} front {rr['front'][0] if rr['front'] else float('nan'):.4f}  back {rr['back'][0] if rr['back'] else float('nan'):.4f}"
                          f"  (back width {rr['back'][2] * 1000 if rr['back'] else float('nan'):.0f} eV)", flush=True)
    for key in ("DM10 EXT4 img1", "DM10 EXT4 img0", "DM09 EXT4 img1"):
        mod, ext, img = key.split()
        out["inject"][key] = injection(mod, int(img[3:]), int(ext[3:]))
        print("injection", key, flush=True)
        for s, r in out["inject"][key]["results"].items():
            print(f"  sigma_xy {s} px")
            for name, v in r.items():
                print(f"     {name:22s} eff {v['eff']:.2f}  E/E_true {v['med']:.4f}  rms {v['rms']:.4f}")
    json.dump(out, open(os.path.join(HERE, "study.json"), "w"), indent=1)
