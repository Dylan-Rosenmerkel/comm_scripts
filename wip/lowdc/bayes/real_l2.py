"""Bayesian few-electron finder on the real L2 ACM 103 noise-test images (wip/lowdc/bayes).

Data: ~/Documents/comm_scripts_data/ndcm_images_2_L2_103/*.fits (read only). Only the 100 x 500 images are used (two per
NDCM setting); the five other shapes are left out. Per extension (CCD) and NDCM group (the two images of a setting
share the noise model and the dark charge): pedestals, negative-side noise model, candidates, evidences, pooled
background; R_N by EM over all images of the extension; hot pixels / columns tested over all 20 images of the
extension (persistence across the run). Column 0 (first-read artefact) is masked.

The gain is NOT calibrated: --gain (default 1.3 ADU/e-, tentative) converts ADU to e-; all electron numbers carry that
caveat. A profile likelihood of the gain from the pixel values is also computed per extension (gain_fit), using the
same noise model (scaled to ADU) and free occupation fractions of n = 0..7 electrons.

    python real_l2.py [--gain 1.3] [--prior physics] [--exts 1,2,3,4]
      -> results/real_l2/candidates.csv, groups.csv, asym.csv, hot.json, gain.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, glob, json, re, sys, time, warnings
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import special

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import noise as NZ, finder as FD, pipeline as PL, background as BG  # noqa: E402

DATA = os.path.expanduser("~/Documents/comm_scripts_data/ndcm_images_2_L2_103")
OUT = os.path.join(HERE, "results", "real_l2")


def load(ext):
    """{ndcm: [(Y_adu, valid, file), ...]} for the 100 x 500 images of one extension"""
    g = {}
    for f in sorted(glob.glob(os.path.join(DATA, "avg_img_100x500x*.fits"))):
        nd = int(re.search(r"img_100x500x(\d+)_", f).group(1))
        with fits.open(f) as h:
            Y = h[ext].data.astype(float)
        valid = np.ones(Y.shape, bool)
        valid[:, 0] = False                          # first-read artefact column
        g.setdefault(nd, []).append((Y, valid, os.path.basename(f)))
    return dict(sorted(g.items()))


def gain_fit(vs_adu, s_adu, nm_e, g_grid, cut_k=4.0):
    """profile log-likelihood of the gain from pedestal-subtracted ADU values above cut_k x s_adu (n = 0 density
    truncated at the cut; n >= 1 densities f((v - g n) / sc)); occupation fractions by EM at each g"""
    v = np.concatenate(vs_adu)
    v = v[np.isfinite(v)]
    cut = cut_k * s_adu
    x = v[v > cut]
    sc = s_adu / nm_e.core
    # P(eps > cut) of the noise model (for the truncated n = 0 component)
    grid = np.linspace(cut, 200 * s_adu, 20000)
    tail0 = np.trapz(np.exp(nm_e.logpdf(grid / sc) - np.log(sc)), grid)
    out = []
    for g in g_grid:
        n = np.arange(0, 8)
        lp = nm_e.logpdf((x[:, None] - g * n[None, :]) / sc) - np.log(sc)
        lp[:, 0] -= np.log(max(tail0, 1e-300))
        pi = np.full(n.size, 1.0 / n.size)
        for _ in range(300):
            t = lp + np.log(pi)
            r = np.exp(t - special.logsumexp(t, axis=1, keepdims=True))
            pi = (r.sum(0) + 0.1) / (x.size + 0.1 * n.size)
        out.append(float(special.logsumexp(lp + np.log(pi), axis=1).sum()))
    return np.array(out), int(x.size)


def _pos_tail(nm, a=(1.5, 5.0)):
    """model probability of a noise value beyond +a / below -a e- (positive / negative), for the asymmetry check"""
    out = []
    for x in a:
        g = np.geomspace(x, 2000, 4000)
        out.append(f"{np.trapz(np.exp(nm.logpdf(g)), g):.1e}/{np.trapz(np.exp(nm.logpdf(-g)), g):.1e}")
    return ";".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gain", type=float, default=1.3)
    ap.add_argument("--prior", default="physics")
    ap.add_argument("--exts", default="1,2,3,4")
    ap.add_argument("--tag", default="", help="output subfolder suffix (e.g. a gain variant)")
    ap.add_argument("--noise", default="mix", help="pipeline noise option (mix | empirical | empirical_asym ...)")
    ap.add_argument("--point", type=int, default=0, help="1: point-deposit component (population_em_pt)")
    a = ap.parse_args()
    opts = dict(noise=a.noise, point_defects=bool(a.point))
    warnings.filterwarnings("ignore")
    global OUT
    OUT = OUT + a.tag
    os.makedirs(OUT, exist_ok=True)
    prior = FD.Prior(a.prior)
    cand_rows, grp_rows, asym_rows, gain_rows, hot = [], [], [], [], {}
    for ext in [int(e) for e in a.exts.split(",")]:
        data = load(ext)
        allres = {}
        t0 = time.time()
        for nd, ims in data.items():
            res = PL.process_group([(Y, v) for Y, v, _ in ims], prior, gain=a.gain, nm_pooled=True, n_shuffle=4, seed=nd,
                                   **opts)
            allres[nd] = (res, ims)
        # R_N pooled over the extension (all NDCM groups), posteriors recomputed with it
        nbs = [nb for nd, (res, _) in allres.items() if nd >= 500 for nb in res["nbs"]]
        area = sum(res["area"] for nd, (res, _) in allres.items() if nd >= 500)
        R_pt = None
        if a.point:
            R_N, R_pt = PL.population_em_pt(nbs, area, prior.pN)
        else:
            R_N = PL.population_em(nbs, area, prior.pN)
        for res, _ in allres.values():
            for nb in res["nbs"]:
                FD.posterior(nb, R_N, R_pt=R_pt)
            for nbn in res["null"]:
                for nb in nbn:
                    FD.posterior(nb, R_N, R_pt=R_pt)
        # run-level persistence: hot pixels / columns over all images of the extension
        # (NDCM >= 500 only: below, read noise >= 0.2 e- makes ~20 % of the pixels candidates - out of scope)
        ok = [nd for nd in allres if nd >= 500]
        pts = [pt for nd in ok for pt in allres[nd][0]["pixtabs"]]
        nbl = [nb for nd in ok for nb in allres[nd][0]["nbs"]]
        vals = [v for nd in ok for _, v, _ in allres[nd][1]]
        _, bgrun = BG.fit(pts, nbl, vals)
        hot[f"EXT{ext}"] = dict(lam0_run=bgrun["lam0"], n_soft_run=bgrun["n_soft"], exposure_run=bgrun["exposure_pix"], hot_cols=bgrun["hot_cols"], hot_pix=bgrun["hot_pix"],
                                R_N=[float(x) for x in R_N], R_total=float(R_N.sum()))
        for nd, (res, ims) in allres.items():
            nm = res["noise"][0]["nm"]
            nnull = sum(1 for nbn in res["null"] for nb in nbn for d in nb if "logJN" in d and d["logJ1"] > 5)
            ni = getattr(nm, "info", {})
            grp_rows.append(dict(ext=ext, ndcm=nd, n_img=len(ims), s_core_e=nm.s[0], s_core_adu=nm.s[0] * a.gain,
                                 noise_mode=ni.get("mode", "mix"), c_tail=ni.get("c_tail", np.nan),
                                 c_exc=ni.get("c_exc", np.nan), ratio_safe=ni.get("ratio_safe", np.nan),
                                 p_pos_tail=_pos_tail(nm),
                                 w_tail=nm.w[1], s_tail_e=nm.s[1], w_broad=nm.w[2], s_broad_e=nm.s[2],
                                 lam0=res["bg"]["lam0"], lam_mean=res["bg"]["lam_mean"], n_soft_dark=res["bg"]["n_soft"],
                                 lam_init=[float(n["lam1"]) for n in res["noise"]],
                                 n_cand=sum(int(p["r"].size) for p in res["pixtabs"]),
                                 n_big=sum(1 for nb in res["nbs"] for d in nb if d.get("big")),
                                 n_logJ1_gt5=sum(1 for nb in res["nbs"] for d in nb if "logJN" in d and d["logJ1"] > 5),
                                 n_null_logJ1_gt5=nnull, null_reps=len(res["null"]),
                                 n_p50=sum(1 for nb in res["nbs"] for d in nb if d.get("p_ev", 0) > 0.5),
                                 t_find=float(np.sum(res["t_find"])), t_noise=float(np.sum(res["t_noise"]))))
            for k, n in enumerate(res["noise"]):
                for r in n["asym"]:
                    asym_rows.append(dict(ext=ext, ndcm=nd, img=ims[k][2], s_core=n["nm"].s[0], **r))
            for k, nb in enumerate(res["nbs"]):
                v = res["noise"][k]["v"]
                for d in nb:
                    if d.get("big"):
                        cand_rows.append(dict(ext=ext, ndcm=nd, file=ims[k][2], kind="big", y=d["y"], x=d["x"], n_cand=d["n_cand"],
                                              q_sum=d["q_sum"]))
                        continue
                    if "logJN" not in d or d["logJ1"] < 0:
                        continue
                    ps = np.exp(d["logJsig"] - special.logsumexp(d["logJsig"])) if "logJsig" in d else None
                    rr, cc = d["pix"]
                    cand_rows.append(dict(ext=ext, ndcm=nd, file=ims[k][2], kind="halo" if d.get("halo") else "nb",
                                          y=round(d["y"], 2), x=round(d["x"], 2), n_cand=d["n_cand"], q_sum=round(d["q_sum"], 2),
                                          pix=";".join(f"{r}:{c}:{v[r, c]:.2f}" for r, c in zip(rr, cc)),
                                          logJ1=round(d["logJ1"], 2), log10BF_R=round((d["logJ1"] + np.log(max(R_N.sum(), 1e-12))) / np.log(10), 2),
                                          p_ev=d.get("p_ev"), N_mean=d.get("N_mean"), N_map=d.get("N_map"),
                                          sig_mean=float((ps * FD.SIG_NODES).sum()) if ps is not None else np.nan))
        # gain profile (NDCM >= 1000: well-separated peaks)
        g_grid = np.round(np.arange(0.7, 2.21, 0.01), 3)
        sel = [nd for nd in allres if nd >= 1000]
        vs, ss = [], []
        for nd in sel:
            res, ims = allres[nd]
            for n, (_, vmask, _) in zip(res["noise"], ims):
                vs.append(n["v"][vmask] * a.gain)          # back to ADU
                ss.append(n["nm"].s[0] * a.gain)
        nm_e = allres[sel[0]][0]["noise"][0]["nm"]
        ll, npx = gain_fit(vs, float(np.median(ss)), nm_e, g_grid)
        for g, l in zip(g_grid, ll):
            gain_rows.append(dict(ext=ext, gain=g, logL=l, n_pix=npx))
        print(f"EXT{ext}: {time.time() - t0:.0f} s, R={R_N.sum():.2e}, gain MLE {g_grid[np.argmax(ll)]:.2f} ({npx} px)", flush=True)
    pd.DataFrame(cand_rows).to_csv(os.path.join(OUT, "candidates.csv"), index=False)
    pd.DataFrame(grp_rows).to_csv(os.path.join(OUT, "groups.csv"), index=False)
    pd.DataFrame(asym_rows).to_csv(os.path.join(OUT, "asym.csv"), index=False)
    pd.DataFrame(gain_rows).to_csv(os.path.join(OUT, "gain.csv"), index=False)
    json.dump(hot, open(os.path.join(OUT, "hot.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
