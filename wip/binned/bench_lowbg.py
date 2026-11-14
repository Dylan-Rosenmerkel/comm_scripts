"""Benchmark with truth for ultra-low-background binned images (plan 1A.1): dark current 1e-6..1e-4 e-/pix, few events,
many images, any binning, consecutive images of a continuous readout.

    python bench_lowbg.py make OUT [--bins 1x1 10x10 ...] [--dc 1e-6 1e-5 1e-4] [--n 20]
    python bench_lowbg.py run OUT                         run_superWADERs --gain 1.3 on every (binning, dark current)
    python bench_lowbg.py analyse OUT                     OUT/report.md, report.json

Simulation (per CCD, 2 CCDs per image):
  frame       one continuous readout of n images x 600 unbinned rows x 2000 columns: the images are consecutive cuts
              of it, so an event crossing an image boundary is split between two images (truth kept per image)
  dark        Poisson per bin with mean dc x b_y x b_x (sum of the pixels' Poisson charges); a hot column (+2e-3 e-/pix)
              and 3 hot pixels (+0.5 e- per image), the same in every image
  events      few-electron events (2-10 e-, ~6 per image and CCD), X-rays (1-8 keV, ~2), muons (~0.5): electrons
              spread by the DAMIC diffusion law sigma^2 = -A ln(1 - b z) (A 285 um^2, b 8.2e-4 /um, 15 um pixels),
              depth uniform in 0-675 um; muons straight, depth linear from 0 to 675 um along the track, 80 e-/um
  readout     read noise 0.12 e- once per bin, gain 1.3 ADU/e-, pedestal 1000 ADU, 20 overscan columns (no charge)
Truth: OUT/<setting>/truth.json, every event (or event part) per image file and CCD: type, electrons, binned centroid,
split flag. Matching: a cluster within 1.5 bins in both directions with charge within 1 e- + 15 %; muons: found if a
cluster within the track's box holds >= 50 % of its charge, whole if one holds >= 90 %. Fakes: clusters < 20 e- more
than 2 bins from any truth event, hot pixel or hot column.
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
TEMPLATE = sorted(glob.glob(os.path.expanduser("~/Documents/comm_scripts_data/ndcm_images_2_L2_103/avg_img_100x500x1000_*.fits")))[0]
H, W, NOV, G, NOISE, PED, NEXT = 600, 2000, 20, 1.3, 0.12, 1000.0, 2
EPS = 3.75e-3                          # keV per electron, for matching only
A_UM2, B_UM, THICK, PIX = 285.0, 8.2e-4, 675.0, 15.0


def sigma_px(z):
    return np.sqrt(-A_UM2 * np.log(1 - B_UM * np.minimum(z, THICK))) / PIX


def deposit(rng, y0, x0, n_e, z):
    """electron positions (unbinned pixel coordinates) of a point event of n_e electrons at depth z"""
    s = sigma_px(z)
    return y0 + rng.normal(0, s, n_e), x0 + rng.normal(0, s, n_e)


def muon(rng, Ht, Wt):
    L = rng.uniform(30, 400)
    ang = rng.uniform(0, np.pi)
    y0, x0 = rng.uniform(0, Ht), rng.uniform(0, Wt)
    n_step = int(L * 4)
    t = (np.arange(n_step) + 0.5) / n_step
    ys, xs = y0 + t * L * np.sin(ang), x0 + t * L * np.cos(ang)
    path = np.hypot(L * PIX, THICK)
    q = rng.poisson(80.0 * path / n_step, n_step)
    z = t * THICK
    Y = np.concatenate([np.full(k, y) for k, y in zip(q, ys)])
    X = np.concatenate([np.full(k, x) for k, x in zip(q, xs)])
    S = np.concatenate([np.full(k, s) for k, s in zip(q, sigma_px(z))])
    return Y + rng.normal(0, 1, Y.size) * S, X + rng.normal(0, 1, X.size) * S


def make_setting(out, by, bx, dc, n_img, seed):
    from astropy.io import fits
    rng = np.random.default_rng(seed)
    name = f"{by}x{bx}_dc{dc:g}"
    d = os.path.join(out, name, "data")
    os.makedirs(d, exist_ok=True)
    Ht = n_img * H
    hb, wb = H // by, W // bx
    truth = {}
    frames = []
    for ext in range(1, NEXT + 1):
        ev = []                                             # (type, Y, X electron positions)
        for _ in range(rng.poisson(6 * n_img)):
            ev.append(("fewe",) + deposit(rng, rng.uniform(0, Ht), rng.uniform(0, W), int(rng.integers(2, 11)), rng.uniform(0, THICK)))
        for _ in range(rng.poisson(2 * n_img)):
            ev.append(("xray",) + deposit(rng, rng.uniform(0, Ht), rng.uniform(0, W), int(rng.uniform(1.0, 8.0) / EPS), rng.uniform(0, THICK)))
        for _ in range(rng.poisson(0.5 * n_img)):
            ev.append(("muon",) + muon(rng, Ht, W))
        hot_col = int(rng.integers(100, W - 100))
        hot_pix = [(int(rng.integers(0, H)), int(rng.integers(0, W))) for _ in range(3)]
        frames.append((ev, hot_col, hot_pix))
    for i in range(n_img):
        fn = f"avg_img_{hb}x{wb + NOV}x1000_{by}x{bx}_L2_103_20261113_{100000 + i * 100:06d}_{i}.fits"
        hdus = []
        for ext, (ev, hot_col, hot_pix) in enumerate(frames, start=1):
            lam = np.full((hb, wb), dc * by * bx)
            lam[:, hot_col // bx] += 2e-3 * by                  # the hot column: +2e-3 e-/pix in one unbinned column
            q = rng.poisson(lam).astype(float)
            for (py, px) in hot_pix:
                q[py // by, px // bx] += rng.poisson(0.5)
            tr = []
            for k, (kind, Y, X) in enumerate(ev):
                inside = (Y >= i * H) & (Y < (i + 1) * H) & (X >= 0) & (X < W)
                if not inside.any():
                    continue
                yy = ((Y[inside] - i * H) // by).astype(int)
                xx = (X[inside] // bx).astype(int)
                np.add.at(q, (yy, xx), 1.0)
                n_all = int(((X >= 0) & (X < W) & (Y >= 0) & (Y < Ht)).sum())
                tr.append(dict(id=k, type=kind, n=int(inside.sum()), n_event=n_all, split=bool(inside.sum() < n_all),
                               y=float(((Y[inside] - i * H) / by).mean()), x=float((X[inside] / bx).mean()),
                               y0=float(yy.min()), y1=float(yy.max()), x0=float(xx.min()), x1=float(xx.max())))
            tr += [dict(id=-1, type="hot_column", x=float(hot_col // bx))] + [dict(id=-2, type="hot_pixel", y=float(py // by), x=float(px // bx))
                                                                           for (py, px) in hot_pix]
            truth[f"{fn}|{ext}"] = tr
            img = np.hstack([q + rng.normal(0, NOISE, q.shape), rng.normal(0, NOISE, (hb, NOV))]) * G + PED
            hdus.append(fits.ImageHDU(img.astype(np.float32)))
        with fits.open(TEMPLATE) as t:
            h0 = t[0].header.copy()
        h0.update(NPBIN=by, NSBIN=bx, NROW=hb, NCOL=wb + NOV, PRESCAN=0, NCOLPRE=int(h0["PHYSCOL"]) - wb * bx, NDCM=1000,
                  ACMID=103)
        fits.HDUList([fits.PrimaryHDU(header=h0)] + hdus).writeto(os.path.join(d, fn), overwrite=True)
    json.dump(dict(by=by, bx=bx, dc=dc, n_img=n_img, truth=truth), open(os.path.join(out, name, "truth.json"), "w"))
    print("made", name, flush=True)


def run(out, jobs=2, parallel=1, extra=()):
    """run_superWADERs on every setting not done yet (no summary.txt), `parallel` settings at a time, -j jobs each"""
    from concurrent.futures import ThreadPoolExecutor

    def one(sd):
        imgs = sorted(glob.glob(os.path.join(sd, "data", "*.fits")))
        lst = os.path.join(sd, "images.txt")
        open(lst, "w").write("\n".join(imgs) + "\n")
        with open(os.path.join(sd, "run.log"), "w") as fh:
            rc = subprocess.run([PY, os.path.join(REPO, "superWADERs", "run_superWADERs"), lst, "-o", sd, "-j", str(jobs),
                                 "--overwrite", "--no-review-helper", "--gain", str(G)] + list(extra),
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        print(os.path.basename(sd), "rc", rc, flush=True)
    todo = [sd for sd in sorted(glob.glob(os.path.join(out, "*_dc*")))
            if not glob.glob(os.path.join(sd, "superWADERs_*", "summary.txt"))]
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        list(ex.map(one, todo))


def read(f):
    sys.path.insert(0, os.path.join(REPO, "wip", "native_pipeline"))
    import run as R
    c = R.read_clusters(f, names=("Energy", "PosX", "PosY", "finder"))
    return {k: np.asarray(c.get(k, []), float) for k in ("Energy", "PosX", "PosY", "finder")}


def analyse(out):
    rep = {}
    for sd in sorted(glob.glob(os.path.join(out, "*_dc*"))):
        T = json.load(open(os.path.join(sd, "truth.json")))
        rd = glob.glob(os.path.join(sd, "superWADERs_*"))
        if not rd:
            continue
        rd = rd[0]
        stat = {}
        fakes = fakes_bayes = n_files = 0
        split = [0, 0]
        jobs = []                                  # (cluster file, truth list in that file's row coordinates)
        stitched = sorted(glob.glob(os.path.join(rd, "stitched", "*.stitch.json")))
        if stitched:                               # --stitch: truth of the members in stitched rows, split events merged
            for sj in stitched:
                S = json.load(open(sj))
                stem = os.path.basename(S["stitched"])[:-5]
                for ext in range(1, NEXT + 1):
                    ev, other = {}, []
                    for m in S["members"]:
                        for t0 in T["truth"].get(f"{os.path.basename(m['file'])}|{ext}", []):
                            t = dict(t0)
                            for k in ("y", "y0", "y1"):
                                if k in t:
                                    t[k] = t[k] + m["row0"]
                            if t["type"] in ("hot_column", "hot_pixel"):
                                other.append(t)
                            elif t["id"] in ev:            # the other part of a split event
                                e = ev[t["id"]]
                                w = e["n"] + t["n"]
                                e.update(y=(e["y"] * e["n"] + t["y"] * t["n"]) / max(w, 1), x=(e["x"] * e["n"] + t["x"] * t["n"]) / max(w, 1),
                                         y0=min(e["y0"], t["y0"]), y1=max(e["y1"], t["y1"]), x0=min(e["x0"], t["x0"]), x1=max(e["x1"], t["x1"]), n=w)
                            else:
                                ev[t["id"]] = t
                    jobs.append((os.path.join(rd, "root", f"panaSKImg_clustersRec_{stem}_ext{ext}.root"), list(ev.values()) + other))
        else:
            for key, tr in T["truth"].items():
                fn, ext = key.split("|")
                jobs.append((os.path.join(rd, "root", f"panaSKImg_clustersRec_{fn[:-5]}_ext{ext}.root"), tr))
        for f, tr in jobs:
            if not os.path.exists(f):
                continue
            n_files += 1
            c = read(f)
            ne = c["Energy"] / EPS
            used = np.zeros(ne.size, bool)
            for t in tr:
                if t["type"] in ("hot_column", "hot_pixel"):
                    near = (np.abs(c["PosX"] - t["x"]) <= 2) & ((t["type"] == "hot_column") | (np.abs(c["PosY"] - t.get("y", 0)) <= 2))
                    used |= near
                    continue
                if t["type"] == "muon":
                    box = (c["PosY"] >= t["y0"] - 2) & (c["PosY"] <= t["y1"] + 2) & (c["PosX"] >= t["x0"] - 2) & (c["PosX"] <= t["x1"] + 2)
                    used |= box
                    best = ne[box].max() / max(t["n"], 1) if box.any() else 0.0
                    s = stat.setdefault("muon", [0, 0, 0])
                    s[0] += 1
                    s[1] += best >= 0.5
                    s[2] += best >= 0.9
                    continue
                cat = t["type"] if t["type"] != "fewe" else ("fewe_%d" % t["n"] if t["n"] <= 3 else "fewe_4-5" if t["n"] <= 5 else "fewe_6-10")
                if t["n"] < 2:
                    cat = "fragment<2e"
                near = (np.abs(c["PosY"] - t["y"]) <= 1.5) & (np.abs(c["PosX"] - t["x"]) <= 1.5)
                ok = near & (np.abs(ne - t["n"]) <= 1 + 0.15 * t["n"])
                used |= near
                s = stat.setdefault(cat, [0, 0])
                s[0] += 1
                s[1] += bool(ok.any())
                if t["split"]:
                    split[0] += 1
                    split[1] += bool(ok.any())
            fk = (~used) & (ne < 20)
            fakes += int(fk.sum())
            fakes_bayes += int((fk & (c["finder"] == 2)).sum())
        dcs = []
        for s in json.load(open(os.path.join(rd, "native_summary.json"))):
            for e, cc in s["ccds"].items():
                if "dc_e" in cc:
                    dcs.append(cc["dc_e"])
        mpix = (len(T["truth"]) if stitched else n_files) * H * W / 1e6
        rep[os.path.basename(sd)] = dict(
            eff={k: (v[1] / v[0] if v[0] else None, v[0]) for k, v in sorted(stat.items()) if k != "muon"},
            muon=dict(n=stat.get("muon", [0])[0], found=(stat["muon"][1] / stat["muon"][0]) if stat.get("muon", [0])[0] else None,
                      whole=(stat["muon"][2] / stat["muon"][0]) if stat.get("muon", [0])[0] else None),
            fakes=fakes, fakes_bayes=fakes_bayes, fakes_per_mpix=fakes / max(mpix, 1e-9), files=n_files,
            split_found=(split[1] / split[0]) if split[0] else None, split_n=split[0],
            dc_fit_median=float(np.median(dcs)) if dcs else None, dc_true=T["dc"] * T["by"] * T["bx"],
            chains=open(os.path.join(rd, "summary.txt")).read().count("A_GMX") if os.path.exists(os.path.join(rd, "summary.txt")) else None)
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
    cats = ["fewe_2", "fewe_3", "fewe_4-5", "fewe_6-10", "xray"]
    L = ["| setting | " + " | ".join(cats) + " | muon found / whole | split events found | fakes (/Mpix) | DC fit / true [e-/bin] |",
         "|---" * (len(cats) + 5) + "|"]
    fe = lambda x: "-" if x is None else f"{x:.2e}"
    for k, v in rep.items():
        e = [f"{v['eff'][c][0]:.2f}" if c in v["eff"] and v["eff"][c][0] is not None else "-" for c in cats]
        m = v["muon"]
        L.append(f"| {k} | " + " | ".join(e) + f" | {m['found'] if m['found'] is None else round(m['found'], 2)} / "
                 f"{m['whole'] if m['whole'] is None else round(m['whole'], 2)} | "
                 f"{'-' if v['split_found'] is None else round(v['split_found'], 2)} ({v['split_n']}) | "
                 f"{v['fakes']} ({v['fakes_per_mpix']:.2f}) | {fe(v['dc_fit_median'])} / {v['dc_true']:.2e} |")
    open(os.path.join(out, "report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("make", "run", "analyse", "all"))
    ap.add_argument("out")
    ap.add_argument("--bins", nargs="*", default=["1x1", "10x10", "1x10", "10x1", "1x100", "100x1", "100x100"])
    ap.add_argument("--dc", nargs="*", type=float, default=[1e-6, 1e-5, 1e-4])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("-j", "--jobs", type=int, default=2, help="workers per run_superWADERs run")
    ap.add_argument("--parallel", type=int, default=1, help="settings run at the same time")
    ap.add_argument("--extra", default="", help="more run_superWADERs arguments, e.g. '--clusterer threshold'")
    a = ap.parse_args()
    if a.what in ("make", "all"):
        for k, b in enumerate(a.bins):
            by, bx = (int(v) for v in b.split("x"))
            for j, dc in enumerate(a.dc):
                make_setting(a.out, by, bx, dc, a.n, seed=1000 + 10 * k + j)
    if a.what in ("run", "all"):
        run(a.out, a.jobs, a.parallel, a.extra.split())
    if a.what in ("analyse", "all"):
        analyse(a.out)


if __name__ == "__main__":
    main()
