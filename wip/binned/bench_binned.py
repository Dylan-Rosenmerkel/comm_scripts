"""Binned-image benchmark with truth (plan docs/superWADERs_plan_binned_crowded.md, 1.6).

Real unbinned images are binned in software the way the CCD bins on chip: the electron counts of b_y x b_x pixels are
summed (the dark charge and every event with them) and the read noise is added ONCE per bin. The unbinned images'
own clusters (run_superWADERs, --clusterer auto) are the truth.

    python bench_binned.py make OUT [--bins 2x2 5x5 10x10 2x1 1x1]   binned FITS per setting (1x1: re-noised, sanity)
    python bench_binned.py run OUT                                    run_superWADERs on every setting + the unbinned
    python bench_binned.py analyse OUT                                report.md / report.json

Source: DM09 Image_5 (3 images x 4 CCDs, Fe-55, 480 x 1280, no overscan). Electrons per pixel = round(pedestal-
subtracted ADU / gain) from superWADERs' engine (read noise ~0.15 e-, so the rounding is the counting itself).
Per bin: sum of the integer electrons + N(0, sigma_read) e-, x gain, + a pedestal of 1000 ADU. Header: NPBIN / NSBIN
= b_y / b_x, NROW / NCOL binned, PRESCAN 0 (the 8 prescan columns are dropped before binning).

Metrics per setting (against the truth, positions divided by b):
  K-alpha     peak of all clusters in 5.4-6.6 keV (Gaussian on a histogram) and its width
  counts      clusters > 1 keV per CCD
  matching    a truth cluster > 1 keV is found when a binned cluster lies within 1.5 bins (+ 0.5 bin) with an energy
              within 10 % or 0.3 keV (merges of neighbours count as found for the brighter one); fakes: binned clusters
              > 1 keV with no truth cluster within 2 bins
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = "/opt/homebrew/Caskroom/miniforge/base/envs/damic/bin/python"
SRC_LIST = os.path.expanduser("~/Documents/comm_scripts_data/waders_integration_example/DM09_Image5.txt")
PRE = 8


def images():
    return [l.strip() for l in open(SRC_LIST) if l.strip() and not l.startswith("#")]


def electrons(path):
    """{ext: (e- per pixel as integers [rows, active cols], gain, read noise e-)} from superWADERs' engine"""
    sys.path.insert(0, os.path.join(REPO, "superWADERs", "engine"))
    sys.path.insert(0, os.path.join(REPO, "superWADERs", "clustering"))
    import native_pipeline as NP
    s = NP.process_image(path, os.devnull, outputs=[], write=False, keep_arrays=True, all_ext=True, verbose=False)
    out = {}
    for ext, c in s["ccds"].items():
        if "ps" not in c:
            continue
        e = np.round(c["ps"][:, PRE:] / c["gain"])
        out[int(ext[3:])] = (e, float(c["gain"]), float(c["noise_e"]))
    return out


def binned(e, by, bx, noise, rng):
    h, w = e.shape[0] // by * by, e.shape[1] // bx * bx
    q = e[:h, :w].reshape(h // by, by, w // bx, bx).sum(axis=(1, 3))
    return q + rng.normal(0.0, noise, q.shape)


def make(out, bins):
    from astropy.io import fits
    rng = np.random.default_rng(20261113)
    for p in images():
        name = os.path.basename(p).split(".")[0]
        E = electrons(p)
        with fits.open(p) as hd:
            h0 = hd[0].header.copy()
            hx = {n: hd[n].header.copy() for n in range(1, len(hd))}
        for b in bins:
            by, bx = (int(v) for v in b.split("x"))
            d = os.path.join(out, "data", b)
            os.makedirs(d, exist_ok=True)
            ph = h0.copy()
            hdus = []
            for n in sorted(E):
                e, g, nz = E[n]
                img = binned(e, by, bx, nz, rng) * g + 1000.0
                hdr = hx[n].copy()
                hdus.append(fits.ImageHDU(img.astype(np.float32), header=hdr))
            ny, nx = hdus[0].data.shape
            ph.update(NPBIN=by, NSBIN=bx, NROW=ny, NCOL=nx, PRESCAN=0)
            # same file-name pattern (image number at the end), the binning in the name for the reader
            fits.HDUList([fits.PrimaryHDU(header=ph)] + hdus).writeto(os.path.join(d, f"{name}_bin{b}.fits"), overwrite=True)
        print("made", name, flush=True)


def run(out):
    jobs = [("unbinned", images())] + [(os.path.basename(d), sorted(glob.glob(os.path.join(d, "*.fits"))))
                                       for d in sorted(glob.glob(os.path.join(out, "data", "*")))]
    for name, imgs in jobs:
        d = os.path.join(out, "runs", name)
        os.makedirs(d, exist_ok=True)
        lst = os.path.join(d, f"{name}.txt")
        open(lst, "w").write("\n".join(imgs) + "\n")
        with open(os.path.join(d, "run.log"), "w") as fh:
            rc = subprocess.run([PY, os.path.join(REPO, "superWADERs", "run_superWADERs"), lst, "-o", d, "-j", "2",
                                 "--overwrite", "--no-review-helper"], stdout=fh, stderr=subprocess.STDOUT).returncode
        print(name, "rc", rc, flush=True)


def clusters(rd):
    """{(image key, ext): dict(E [keV], y, x)} of a run folder"""
    sys.path.insert(0, os.path.join(REPO, "wip", "native_pipeline"))
    import run as R
    out = {}
    for f in sorted(glob.glob(os.path.join(rd, "root", "panaSKImg_clustersRec_*_ext*.root"))):
        base = os.path.basename(f)
        ext = int(base.rsplit("_ext", 1)[1][:-5])
        key = base.split("Low_Temp_109_")[1].split("_bin")[0].split("_ext")[0]
        c = R.read_clusters(f, names=("Energy", "PosX", "PosY", "STD_XY"))
        out[(key, ext)] = dict(E=np.asarray(c.get("Energy", []), float), x=np.asarray(c.get("PosX", []), float),
                               y=np.asarray(c.get("PosY", []), float))
    return out


def kalpha(E):
    sel = E[(E > 5.4) & (E < 6.6)]
    if sel.size < 30:
        return None
    h, ed = np.histogram(sel, 60, (5.4, 6.6))
    c = 0.5 * (ed[1:] + ed[:-1])
    from scipy.optimize import curve_fit
    try:
        p, cov = curve_fit(lambda x, a, m, s, k: a * np.exp(-0.5 * ((x - m) / s) ** 2) + k, c, h,
                           p0=(h.max(), c[np.argmax(h)], 0.1, h.min()))
        return dict(mu=float(p[1]), err=float(np.sqrt(cov[1, 1])), sigma=float(abs(p[2])), n=int(sel.size))
    except Exception:
        return None


def analyse(out):
    from scipy.spatial import cKDTree
    runs = {os.path.basename(d): glob.glob(os.path.join(d, "superWADERs_*"))[0]
            for d in glob.glob(os.path.join(out, "runs", "*")) if glob.glob(os.path.join(d, "superWADERs_*"))}
    T = clusters(runs["unbinned"])
    rep = {}
    for name, rd in sorted(runs.items()):
        B = clusters(rd)
        by, bx = (1, 1) if name == "unbinned" else (int(v) for v in name.split("x"))
        allE = np.concatenate([v["E"] for v in B.values()]) if B else np.zeros(0)
        found = tot = fakes = nb = 0
        ratio = []
        for k, t in T.items():
            if k not in B:
                continue
            b = B[k]
            ts = t["E"] > 1.0
            off = 0 if name == "unbinned" else PRE        # the binned images start at the first active column
            ty, tx = (t["y"][ts]) / by, (t["x"][ts] - off) / bx
            bs = b["E"] > 1.0
            nb += int(bs.sum())
            if not bs.any():
                tot += int(ts.sum())
                continue
            tree = cKDTree(np.c_[b["y"][bs], b["x"][bs]])
            d, j = tree.query(np.c_[ty, tx], k=3, distance_upper_bound=2.0)
            for i, te in enumerate(t["E"][ts]):
                tot += 1
                ok = [jj for dd, jj in zip(d[i], j[i]) if np.isfinite(dd) and dd <= 1.5 + 0.5
                      and abs(b["E"][bs][jj] - te) <= max(0.1 * te, 0.3)]
                if ok:
                    found += 1
                    ratio.append(b["E"][bs][ok[0]] / te)
            tt = cKDTree(np.c_[ty, tx]) if ts.any() else None
            if tt is not None:
                dd, _ = tt.query(np.c_[b["y"][bs], b["x"][bs]], distance_upper_bound=2.0)
                fakes += int(np.sum(~np.isfinite(dd)))
        rep[name] = dict(kalpha=kalpha(allE), clusters_gt1keV=nb, truth_gt1keV=tot, efficiency=found / max(tot, 1),
                         fakes=fakes, e_ratio_median=float(np.median(ratio)) if ratio else None,
                         chains=open(os.path.join(rd, "summary.txt")).read().count("-> root") if os.path.exists(
                             os.path.join(rd, "summary.txt")) else None)
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
    L = ["| setting | K-alpha [keV] | sigma [keV] | clusters > 1 keV (truth) | efficiency | fakes | E ratio |", "|---|---|---|---|---|---|---|"]
    for k, v in rep.items():
        ka = v["kalpha"] or {}
        L.append(f"| {k} | {ka.get('mu', float('nan')):.4f} +- {ka.get('err', float('nan')):.4f} | {ka.get('sigma', float('nan')):.3f} | "
                 f"{v['clusters_gt1keV']} ({v['truth_gt1keV']}) | {v['efficiency']:.3f} | {v['fakes']} | {v['e_ratio_median']} |")
    open(os.path.join(out, "report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("make", "run", "analyse", "all"))
    ap.add_argument("out")
    ap.add_argument("--bins", nargs="*", default=["1x1", "2x2", "5x5", "10x10", "2x1"])
    a = ap.parse_args()
    if a.what in ("make", "all"):
        make(a.out, a.bins)
    if a.what in ("run", "all"):
        run(a.out)
    if a.what in ("analyse", "all"):
        analyse(a.out)


if __name__ == "__main__":
    main()
