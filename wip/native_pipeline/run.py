#!/usr/bin/env python3
"""Robustness suite of the native pipeline with --clusterer auto (ANYIMAGE.md).

One or a few images of every data type on this machine go through native_pipeline.py --clusterer auto (one process
per case, at most 2 at a time). Per case the suite records success / failure, the chain chosen per CCD, timings and
sanity checks (cluster counts per finder, Fe-55 K-alpha where present, dark charge, gain source, fallbacks), and
compares with a reference where one exists:
  dm09          DM09 Image_5 (3 images): K-alpha of auto must equal --clusterer agmx (same g(n) curve)
  dm07/27/28    Image_8 Fe-55 (2025-09-23), first images: K-alpha vs the A_GMX + g(n) files of the four-scenario run
                (root_agmx_gaincorr, same images) and the composite energy_summary.csv (all 30 images)
  pd06_A/B/C    PD06 settings (noisy, non-Poisson background: 'unmodelled' expected)
  amtest        AMtest runs_29img, Am-241 (crowded EXT4 expected)
  l2_103        L2 ACM 103 noise test: 100 x 500 pairs, strips (4 x 6294, 12 x 2000, ...), --gain 1.3 (Bayes regime)
  binned        real binned images (20 x 20, 10 x 10) and a synthetic 2 x 2 binned image
  raw_skips     synthetic skip-resolved file (no real one on this machine): compression, skip_samples, processing
  single_skip   NDCM = 1 images (unresolved, noise ~4 e-)
  std_image     a var_img (std image): must be refused, not processed

    python run.py [CASE ...] [-o OUT] [-j 2] [--list]    -> OUT/<case>/ (pipeline output), OUT/suite.json, OUT/suite.md
Outputs never go into data folders.
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/opt/homebrew/Caskroom/miniforge/base/envs/damic/bin/python"
OUT = "/private/tmp/claude-501/-Users-dylanrosenmerkel/511be665-1540-4ae6-86c3-c4f8e8738785/scratchpad/anyimage/suite"
DOC = os.path.expanduser("~/Documents")
DATA = f"{DOC}/comm_scripts_data"
AN = f"{DATA}/2025-09-23-DM07_DM27_DM28_analysis"
RAW8 = f"{DATA}/2025-09-23-DM07_DM27_DM28_Low_Temp_New_Images"
PD = f"{DATA}/PD05_PD06_nonlinearity_20261023"
PDRAW = os.path.expanduser("~/Downloads/2026-10-23-PD05_PD06_nonlinearity_tests")
L2 = f"{DATA}/ndcm_images_2_L2_103"
COMPOSITE = f"{DATA}/nonlinearity_workdir/composite/out/gn/Fe_55_Analysis_Plots"
DM09_LIST = f"{DATA}/waders_integration_example/DM09_Image5.txt"
DM09_CURVE = f"{DATA}/waders_integration_example/WADERs_analysis_DM09_Image5/gain_vs_charge"
MODULES = {"dm28": ("DM28", 106), "dm27": ("DM27", 109), "dm07": ("DM07", 110)}


# =============================================================================
# cases
# =============================================================================
def _lines(path):
    return [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]


def cases(n_fe=2):
    """{name: dict(images, args, expect, ref)}"""
    C = {}
    C["dm09"] = dict(images=_lines(DM09_LIST), args=["--curve", DM09_CURVE], expect=dict(chain={"agmx"}),
                     ref=dict(kind="same_as_agmx"))
    for k, (dm, acm) in MODULES.items():          # crowded expected too: core fill 0.13-0.155 on these 500 x 3500 images
        imgs = _lines(f"{AN}/{dm}_ACM{acm}.txt")[:n_fe]
        C[k] = dict(images=imgs, args=["--curve", f"{AN}/WADERs_analysis_{dm}_ACM{acm}/gain_vs_charge"],
                    expect=dict(chain={"agmx", "crowded"}), ref=dict(kind="fe55_run", run=f"{AN}/WADERs_analysis_{dm}_ACM{acm}",
                                                           module=dm))
    for s in "ABC":
        w = open(f"{PD}/settings_PD06/curves/setting_{s}_images.txt").read().split()
        imgs = [f"{PDRAW}/{x}" for x in w if x.startswith("avg") and re.search(r"_(8|14|20)\.fits", x)][:1]   # full readouts
        C[f"pd06_{s}"] = dict(images=imgs, args=["--curve", f"{PD}/settings_PD06/curves/setting_{s}_gain_vs_charge.json",
                                                 "--ext", "2", "3", "4"],
                              expect=dict(chain={"agmx", "crowded"}), ref=dict(kind="fe55", module=f"PD06_{s}",
                                                                              image_number=12))
    am = _lines(f"{DOC}/JH1_Testing/AMTEst/runs_29img/AMtest_29.txt")
    C["amtest"] = dict(images=am[:1], args=[], expect=dict(chain_ext4={"crowded"}), ref=dict(kind="none"))
    l2 = sorted(glob.glob(f"{L2}/avg_img_100x500x1000_*.fits")) + sorted(glob.glob(f"{L2}/avg_img_100x500x3000_*.fits"))
    strips = sorted(glob.glob(f"{L2}/avg_img_4x6294x*.fits")) + sorted(glob.glob(f"{L2}/avg_img_12x2000x*.fits"))
    C["l2_103"] = dict(images=l2 + strips, args=["--gain", "1.3"], expect=dict(chain={"agmx"}, gain_source="argument"),
                       ref=dict(kind="none"))
    C["l2_103_ext4_g082"] = dict(images=l2[:2], args=["--gain", "EXT1=1.3,EXT2=1.3,EXT3=1.3,EXT4=0.82", "--ext", "4"],
                                 expect=dict(chain={"agmx"}), ref=dict(kind="none"))
    C["bayes_inject"] = dict(images=[f"{OUT}/synthetic/avg_img_100x500x1000_1x1_L2_103_inject_20261104_00000{k}_{k}.fits"
                                     for k in (0, 1)], args=["--gain", "1.3"], expect=dict(chain={"agmx"}),
                             ref=dict(kind="inject"), synthetic="inject")
    C["l2_103_nogain"] = dict(images=l2[:1], args=["--ext", "1"], expect=dict(gain_source="reference prior"),
                              ref=dict(kind="none"))
    binned = sorted(glob.glob(f"{DOC}/JH1_Testing/2026-09-16-2026-09-14-PA03_DR03_135K_Test1/avg_img_80x320x500_20x20_*.fits.bz2"))[:1] \
        + sorted(glob.glob(f"{DOC}/JH1_Testing/2026-09-16-2026-09-14-PA03_DR03_135K_Test1/avg_img_*_10x10_*.fits.bz2"))[:1]
    C["binned"] = dict(images=binned + [f"{OUT}/synthetic/avg_synth_binned2x2_109_20250827_000000_0.fits"], args=[],
                       expect=dict(), ref=dict(kind="none"), synthetic="binned")
    C["raw_skips"] = dict(images=[f"{OUT}/synthetic/skp_synth_raw_109_20250827_000000_0.fits"], args=[],
                          expect=dict(form="raw", chain={"agmx"}), ref=dict(kind="raw_vs_avg"), synthetic="raw")
    ss = sorted(glob.glob(f"{DOC}/LSM_Module_Testing_Analysis/Production_Modules_Data/*/Analysis/"
                          "avg_Image_test_single_skip_*.fz"))[:1]
    C["single_skip"] = dict(images=ss, args=[], expect=dict(chain={"unmodelled"}), ref=dict(kind="none"))
    C["std_image"] = dict(images=sorted(glob.glob(f"{DOC}/JH1_Testing/2026-09-16-2026-09-14-PA03_DR03_135K_Test1/"
                                                  "var_img_80x320x500_20x20_L2_101_*.fits.bz2"))[:1],
                          args=[], expect=dict(refused=True), ref=dict(kind="none"))
    return C


# =============================================================================
# synthetic inputs
# =============================================================================
def make_synthetic(kind, out_dir):
    """synthetic inputs (real noise and clusters underneath): 'raw' = skip-resolved file from a 120 x 400 crop of DM09
    Image_5 image 1 (20 skips per pixel = crop + Gaussian per-skip noise of 0.8 ADU, one 400 ADU excursion in one skip
    of pixel (5, 10)); 'binned' = 2 x 2 sums of the crop (NPBIN = NSBIN = 2); 'inject' = few-electron events added to
    two L2 ACM 103 noise-test images. Written to out_dir (scratchpad), never into data folders."""
    from astropy.io import fits
    os.makedirs(out_dir, exist_ok=True)
    src = _lines(DM09_LIST)[1]
    with fits.open(src) as h:
        h0 = h[0].header.copy()
        crops = [np.asarray(h[i].data, float)[:120, :400] for i in (1, 2)]
    rng = np.random.default_rng(1)
    if kind == "raw":
        p = f"{out_dir}/skp_synth_raw_109_20250827_000000_0.fits"
        nsk, s_skip = 20, 0.8       # per-skip noise [ADU]: the mean gets 0.8/sqrt(20) = 0.18 ADU on top of the crop
        hdus = [fits.PrimaryHDU(header=h0)]
        for c in crops:
            cube = c[:, :, None] + rng.normal(0, s_skip, c.shape + (nsk,))
            cube[5, 10, 3] += 400.0          # one per-skip excursion (for skip_samples)
            hdus.append(fits.ImageHDU(cube.reshape(c.shape[0], c.shape[1] * nsk).astype(np.float32)))
        for hh in (hdus[0].header,):
            hh["NDCM"], hh["NROW"], hh["NCOL"] = nsk, crops[0].shape[0], crops[0].shape[1]
        fits.HDUList(hdus).writeto(p, overwrite=True)
        np.save(f"{out_dir}/skp_synth_truth.npy", np.stack(crops))
        return p
    if kind == "inject":
        # two L2 ACM 103 noise-test images (NDCM 1000) with few-electron events added (gain 1.3 ADU/e- assumed):
        # 8 events per CCD, N = 3..10 e-, Gaussian sigma 0.3-0.7 px, electrons drawn per pixel (multinomial)
        src = sorted(glob.glob(f"{L2}/avg_img_100x500x1000_*.fits"))[:2]
        truth = []
        out = []
        for k, sp in enumerate(src):
            p = f"{out_dir}/avg_img_100x500x1000_1x1_L2_103_inject_20261104_00000{k}_{k}.fits"
            with fits.open(sp) as h:
                hd = fits.HDUList([fits.PrimaryHDU(header=h[0].header.copy())] +
                                  [fits.ImageHDU(np.asarray(h[i].data, float)) for i in range(1, len(h))])
            for i in range(1, len(hd)):
                d = hd[i].data
                for e in range(8):
                    y, x = rng.uniform(10, 90), rng.uniform(20, 330)
                    N, sg = int(rng.integers(3, 11)), rng.uniform(0.3, 0.7)
                    ys = np.floor(rng.normal(y, sg, N)).astype(int); xs = np.floor(rng.normal(x, sg, N)).astype(int)
                    np.add.at(d, (ys, xs), 1.3)
                    truth.append(dict(image=k, ext=i, y=y, x=x, N=N, sigma=sg))
                hd[i].data = d.astype(np.float32)
            hd.writeto(p, overwrite=True)
            out.append(p)
        json.dump(truth, open(f"{out_dir}/inject_truth.json", "w"), indent=0)
        return out[0]
    if kind == "binned":
        p = f"{out_dir}/avg_synth_binned2x2_109_20250827_000000_0.fits"
        hdus = [fits.PrimaryHDU(header=h0)]
        for c in crops:
            b = c.reshape(c.shape[0] // 2, 2, c.shape[1] // 2, 2).sum(axis=(1, 3))
            hdus.append(fits.ImageHDU(b.astype(np.float32)))
        hh = hdus[0].header
        hh["NPBIN"], hh["NSBIN"], hh["NROW"], hh["NCOL"] = 2, 2, crops[0].shape[0] // 2, crops[0].shape[1] // 2
        fits.HDUList(hdus).writeto(p, overwrite=True)
        return p
    raise ValueError(kind)


# =============================================================================
# reading results
# =============================================================================
BR = ("Energy", "fSTD_X", "fSTD_Y", "agmx_cls", "finder", "p_event", "N_post", "PosX", "PosY", "Npix")


def read_clusters(path, names=BR):
    """{branch: array} of the scalar-vector clustersRec branches in names (missing -> absent); {} without clusters"""
    import ROOT
    f = ROOT.TFile.Open(path)
    t = f.Get("clustersRec")
    out = {}
    if t:
        t.GetEntry(0)
        have = {b.GetName() for b in t.GetListOfBranches()}
        for n in names:
            if n in have:
                out[n] = np.asarray(list(getattr(t, n)), float)
    tn = f.Get("characterisation")
    out["_char"] = json.loads(tn.GetTitle()) if tn else None
    f.Close()
    return out


def kalpha(E):
    """MakeComposite's K-alpha fit (utils/validate_native.kalpha): (mu, err, sigma, n)"""
    from scipy.optimize import curve_fit
    h, e = np.histogram(E, 50, (4.5, 7.0))
    c = 0.5 * (e[1:] + e[:-1])
    m = (c > 5.3) & (c < 6.1) & (h > 0)
    if m.sum() < 4:
        return (np.nan, np.nan, np.nan, int(h[m].sum()))
    g = lambda x, a, mu, s: a * np.exp(-0.5 * ((x - mu) / s) ** 2)
    try:
        p, cov = curve_fit(g, c[m], h[m], p0=[h.max(), 5.9, 0.1], sigma=np.sqrt(h[m]), absolute_sigma=True,
                           bounds=([0, 5.3, 0.03], [np.inf, 6.1, 0.40]))
        return (float(p[1]), float(np.sqrt(cov[1, 1])), float(p[2]), int(h[m].sum()))
    except RuntimeError:
        return (np.nan, np.nan, np.nan, int(h[m].sum()))


def front(d):
    """front-side Fe-55 window clusters (sigma_xy < 1; A_GMX cls x-ray / x-ray diffuse where present)"""
    if "Energy" not in d:
        return np.zeros(0)
    E = d["Energy"]
    sel = (E > 4.5) & (E < 7.0) & (np.sqrt(0.5 * (d["fSTD_X"] ** 2 + d["fSTD_Y"] ** 2)) < 1.0)
    if "agmx_cls" in d:
        sel &= np.isin(d["agmx_cls"], (1, 2))
    return E[sel]


def kalpha_per_ext(files):
    """{ext: (mu, err, sigma, n)} of the front K-alpha, images pooled per CCD"""
    acc = {}
    for f in files:
        n = int(f.rsplit("_ext", 1)[1][:-5])
        acc.setdefault(n, []).append(front(read_clusters(f)))
    return {n: kalpha(np.concatenate(v)) for n, v in sorted(acc.items())}


def composite_ka(module, num=8):
    """{ext: K-alpha front} of the composite energy_summary.csv (A_GMX + g(n), all images)"""
    p = f"{COMPOSITE}/{module}/Image_{num}_{module}_list/energy_summary.csv"
    out = {}
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            if r["location"] == "0":
                out[int(r["Extension"])] = float(r["energypeak1"].split("+/-")[0])
    return out


# =============================================================================
# one case
# =============================================================================
def run_case(name, c, out, jobs=1):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(os.path.join(d, "*.root")) + glob.glob(os.path.join(d, "characterisation", "*.json")):
        os.remove(f)                       # results of an earlier run of this case
    if c.get("synthetic"):
        make_synthetic(c["synthetic"], os.path.join(out, "synthetic"))
    lst = os.path.join(d, "images.txt")
    open(lst, "w").write("".join(p + "\n" for p in c["images"]))
    cmd = [PY, os.path.join(HERE, "native_pipeline.py"), lst, "-o", d, "--clusterer", "auto", "-j", str(jobs)] + c["args"]
    t0 = time.time()
    with open(os.path.join(d, "run.log"), "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    res = dict(case=name, cmd=" ".join(cmd), rc=rc, seconds=round(time.time() - t0, 1), images=len(c["images"]))
    res.update(check(name, c, d))
    return res


def check(name, c, d):
    """sanity checks of one finished case"""
    out = dict(ccds=[], problems=[], fallbacks=[])
    files = sorted(glob.glob(os.path.join(d, "panaSKImg_clustersRec_*_ext*.root")))
    out["n_files"] = len(files)
    chars = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(d, "characterisation", "*.json"))}
    out["refused"] = [dict(file=os.path.basename(r["file"]), reason=r.get("reason")) for r in chars.values()
                      if not r.get("process")]
    for stem, r in sorted(chars.items()):
        out["fallbacks"] += [f"{stem}: {x}" for x in r.get("fallbacks", [])]
        for ext, cc in (r.get("ccds") or {}).items():
            res = cc.get("result") or {}
            dec = res.get("decision") or {}
            auto = res.get("auto") or {}
            row = dict(image=stem[-30:], ext=ext, form=r["form"]["value"],
                       shape="x".join(str(v) for v in (r.get("form", {}).get("shape") or [])),
                       channel=cc["channel"]["value"],
                       gain=res.get("gain"), gain_source=(cc.get("gain") or {}).get("source"),
                       gain_conf=(cc.get("gain") or {}).get("confidence"),
                       noise_e=res.get("noise_e"), dc_fit_e=res.get("dc_e"),
                       dc_bg_e=((dec.get("background") or {}).get("value") or {}).get("lam_e"),
                       counting=(cc.get("counting") or {}).get("value"),
                       core_fill=(dec.get("core_fill") or {}).get("value"),
                       chain=dec.get("chain") or ("skipped" if res.get("skipped") or not cc.get("plan", {}).get("process", True)
                                                   else None),
                       fewe=dec.get("fewe"), fewe_pooled=dec.get("fewe_pooled"),
                       fewe_used=((dec.get("agmx_info") or {}).get("fewe") or {}).get("used"),
                       fewe_rec={k: v for k, v in (((dec.get("agmx_info") or {}).get("fewe")) or {}).items()
                                 if k in ("lam", "threshold", "n_bayes", "n_agmx_replaced", "f_c", "reason")},
                       dc_pooled=((cc.get("plan") or {}).get("fewe_pool") or {}).get("lam_median"),
                       agmx_regime=(dec.get("agmx_info") or {}).get("regime"),
                       n_clusters=auto.get("n_clusters"), by_finder=auto.get("n_by_finder"),
                       masked_cols=res.get("masked_columns"), fit_at_limit=res.get("fit_at_limit"),
                       t_ccd=None)
            out["fallbacks"] += [f"{stem} {ext}: {x}" for x in cc.get("fallbacks", []) + dec.get("fallbacks", []) + res.get("fallbacks", [])]
            out["ccds"].append(row)
    exp = c.get("expect", {})
    if exp.get("refused"):
        if not out["refused"]:
            out["problems"].append("expected the file to be refused")
    for row in out["ccds"]:
        if row["chain"] in (None, "skipped"):
            continue
        if "chain" in exp and row["chain"] not in exp["chain"]:
            out["problems"].append(f"{row['image']} {row['ext']}: chain {row['chain']}, expected {sorted(exp['chain'])}")
        if "chain_ext4" in exp and row["ext"] == "EXT4" and row["chain"] not in exp["chain_ext4"]:
            out["problems"].append(f"{row['image']} EXT4: chain {row['chain']}, expected {sorted(exp['chain_ext4'])}")
        if "gain_source" in exp and row["gain_source"] != exp["gain_source"]:
            out["problems"].append(f"{row['image']} {row['ext']}: gain source {row['gain_source']}")
    ref = c.get("ref", {})
    if ref.get("kind") in ("fe55_run", "same_as_agmx", "fe55"):
        out["kalpha"] = {n: v for n, v in kalpha_per_ext(files).items()}
    if ref.get("kind") == "fe55_run":
        stems = {os.path.basename(f).split("panaSKImg_clustersRec_")[1].rsplit("_ext", 1)[0] for f in files}
        rf = [f for f in sorted(glob.glob(os.path.join(ref["run"], "root_agmx_gaincorr", "*.root")))
              if os.path.basename(f).split("panaSKImg_clustersRec_")[1].rsplit("_ext", 1)[0] in stems]
        out["kalpha_ref_same_images"] = kalpha_per_ext(rf)
        out["kalpha_composite_all_images"] = composite_ka(ref["module"])
    if ref.get("kind") == "fe55" and ref.get("module"):
        out["kalpha_composite_all_images"] = composite_ka(ref["module"], ref.get("image_number", 8))
    if ref.get("kind") == "same_as_agmx":
        rd = os.path.join(os.path.dirname(d), f"{name}_agmx")
        if not glob.glob(os.path.join(rd, "*.root")):
            subprocess.run([PY, os.path.join(HERE, "native_pipeline.py"), DM09_LIST, "-o", rd, "--clusterer", "agmx"]
                           + c["args"], stdout=open(os.path.join(rd + ".log"), "w"), stderr=subprocess.STDOUT)
        rf = sorted(glob.glob(os.path.join(rd, "panaSKImg_clustersRec_*_ext*.root")))
        out["kalpha_agmx"] = kalpha_per_ext(rf)
        same = []
        for f in files:
            g = os.path.join(rd, os.path.basename(f))
            a, b = read_clusters(f, ("Energy", "PosX", "PosY")), read_clusters(g, ("Energy", "PosX", "PosY"))
            same.append(bool(np.array_equal(np.sort(a.get("Energy", [])), np.sort(b.get("Energy", []))) and
                             np.array_equal(np.sort(a.get("PosX", [])), np.sort(b.get("PosX", [])))))
        out["identical_to_agmx"] = f"{sum(same)} of {len(same)} files"
        if not all(same):
            out["problems"].append("clusters differ from --clusterer agmx")
    if ref.get("kind") == "inject":
        tr = json.load(open(os.path.join(os.path.dirname(d), "synthetic", "inject_truth.json")))
        found, nb = 0, 0
        for f in files:
            k = int(re.search(r"_(\d)_ext", f).group(1)); ext = int(f.rsplit("_ext", 1)[1][:-5])
            cl = read_clusters(f)
            if "finder" not in cl:
                continue
            b = cl["finder"] == 2
            nb += int(b.sum())
            for t in (t for t in tr if t["image"] == k and t["ext"] == ext):
                if b.any() and np.min(np.hypot(cl["PosX"][b] - 1 - t["x"], cl["PosY"][b] - 1 - t["y"])) < 2.0:
                    found += 1
        out["injected"] = dict(events=len(tr), bayes_clusters=nb, matched_within_2px=found)
    if ref.get("kind") == "raw_vs_avg":
        out.update(check_raw(d))
    return out


def check_raw(d):
    """compressed mean vs the truth crop; skip_samples of the planted excursion"""
    sys.path.insert(0, HERE)
    import characterise as CH
    from astropy.io import fits
    raw = glob.glob(os.path.join(os.path.dirname(d), "synthetic", "skp_synth_raw_*.fits"))[0]
    comp = glob.glob(os.path.join(d, "compressed", "skp_synth_raw_*_0.fits"))
    truth = np.load(os.path.join(os.path.dirname(d), "synthetic", "skp_synth_truth.npy"))
    out = {}
    if comp:
        with fits.open(comp[0]) as h:
            m = np.asarray(h[1].data, float)
        r = m - truth[0]
        r[5, 10] -= 400.0 / 20
        out["compressed_minus_truth_adu"] = dict(mean=float(r.mean()), std=float(r.std()), expected_std=0.8 / np.sqrt(20))
    s = CH.skip_samples(raw, 1, [5], [10])
    out["skip_samples_shape"] = list(s.shape)
    out["skip_samples_excursion"] = dict(skip=int(np.argmax(s[0] - np.median(s[0]))),
                                         excess_adu=float(np.max(s[0] - np.median(s[0]))))
    return out


# =============================================================================
# MakeComposite compatibility
# =============================================================================
def composite(name, out, module, num=8):
    """MakeComposite_Fe55_Clusters_Extensions.cc (copy in composite/, compiled there) on a finished case's files,
    called as run_WADERs step3c / run_composite.py; returns {rc, energy_summary (front K-alpha per ext), log}."""
    d = os.path.join(out, name)
    files = sorted(glob.glob(os.path.join(d, "panaSKImg_clustersRec_*_ext*.root")))
    cdir = os.path.join(d, "composite")
    os.makedirs(cdir, exist_ok=True)
    lst = os.path.join(cdir, f"Image_{num}_{module}_list.txt")
    open(lst, "w").write("".join(f + "\n" for f in files))
    root = os.path.join(os.path.dirname(PY), "root")
    macro = os.path.join(HERE, "composite", "MakeComposite_Fe55_Clusters_Extensions.cc")
    env = dict(os.environ, PATH=os.path.dirname(PY) + ":" + os.environ["PATH"])
    log = os.path.join(cdir, "composite.log")
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run([root, "-l", "-q", "-b", f'{macro}+("{lst}",{num},"{module}","{cdir}")'], stdout=fh,
                            stderr=subprocess.STDOUT, env=env, cwd=cdir).returncode
    es = glob.glob(os.path.join(cdir, "Fe_55_Analysis_Plots", module, "*", "energy_summary.csv"))
    res = dict(rc=rc, seconds=round(time.time() - t0, 1), n_files=len(files), log=log, energy_summary=es[0] if es else None)
    if es:
        res["kalpha_front"] = {int(r["Extension"]): r["energypeak1"] for r in csv.DictReader(open(es[0]))
                               if r["location"] == "0"}
        res["kalpha_back"] = {int(r["Extension"]): r["energypeak1"] for r in csv.DictReader(open(es[0]))
                              if r["location"] == "1"}
    return res


# =============================================================================
# report
# =============================================================================
def fmt(v, f="{:.4g}"):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f.format(v)
    return str(v)


def _few(r):
    if not r or "n_bayes" not in r:
        return ""
    return (f"; Bayes {r['n_bayes']} kept, {r['n_agmx_replaced']} A_GMX replaced, lam {fmt(r.get('lam'), '{:.1e}')}, "
            f"t {fmt(r.get('threshold'), '{:.3f}')}")


def report(results, path):
    L = ["# native pipeline --clusterer auto: robustness suite", ""]
    L.append("| case | rc | s | images | files | refused | problems |")
    L.append("|---|---|---|---|---|---|---|")
    for r in results:
        L.append(f"| {r['case']} | {r['rc']} | {r['seconds']} | {r['images']} | {r.get('n_files')} | "
                 f"{len(r.get('refused', []))} | {'; '.join(r.get('problems', [])) or 'none'} |")
    for r in results:
        L += ["", f"## {r['case']}", "", f"`{r['cmd']}`", ""]
        if r.get("refused"):
            L += [f"- refused: {x['file']}: {x['reason']}" for x in r["refused"]] + [""]
        L.append("| image | ext | form | gain (source, conf) | noise e- | DC fit / bg map / run-pooled e-/pix | counting | core fill | "
                 "chain | fewe (pooled, used) | A_GMX regime | clusters (by finder) | masked cols | fit at limit |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in r.get("ccds", []):
            L.append(f"| {c['image']} | {c['ext']} | {c['form']} | {fmt(c['gain'])} ({c['gain_source']}, {c['gain_conf']}) | "
                     f"{fmt(c['noise_e'], '{:.3f}')} | {fmt(c['dc_fit_e'], '{:.2e}')} / {fmt(c['dc_bg_e'], '{:.2e}')} / "
                     f"{fmt(c.get('dc_pooled'), '{:.2e}')} | "
                     f"{c['counting']} | {fmt(c['core_fill'], '{:.3f}')} | {c['chain']} | {c['fewe']} "
                     f"({c['fewe_pooled']}, {c['fewe_used']}{_few(c.get('fewe_rec'))}) | {c['agmx_regime']} | {c['n_clusters']} "
                     f"{c['by_finder'] or ''} | {c['masked_cols'] or ''} | {'; '.join(c['fit_at_limit'] or [])} |")
        for k in ("kalpha", "kalpha_agmx", "kalpha_ref_same_images", "kalpha_composite_all_images"):
            if k in r:
                L.append("")
                L.append(f"- {k}: " + ", ".join(
                    f"EXT{n} " + (f"{v[0]:.4f} +- {v[1]:.4f} (N {v[3]})" if isinstance(v, (list, tuple)) else f"{v:.4f}")
                    for n, v in sorted(r[k].items(), key=lambda kv: int(kv[0]))))
        for k in ("identical_to_agmx", "injected", "compressed_minus_truth_adu", "skip_samples_shape", "skip_samples_excursion",
                  "composite"):
            if k in r:
                L.append(f"- {k}: {r[k]}")
        if r.get("fallbacks"):
            L += ["", "Fallbacks (logged):"] + [f"- {x}" for x in sorted(set(r["fallbacks"]))[:40]]
    open(path, "w").write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*")
    ap.add_argument("-o", "--out", default=OUT)
    ap.add_argument("-j", "--jobs", type=int, default=1, help="cases in parallel (each one process; keep <= 2)")
    ap.add_argument("--n-fe", type=int, default=2, help="images per DM07/27/28 module")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="re-check finished cases, no runs")
    ap.add_argument("--composite", nargs=3, metavar=("CASE", "MODULE", "IMAGE_NUMBER"),
                    help="run MakeComposite on a finished case (e.g. dm28 DM28 8)")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    C = cases(a.n_fe)
    names = a.cases or list(C)
    if a.list:
        for n in names:
            print(n, len(C[n]["images"]), C[n]["args"])
        return 0
    os.makedirs(a.out, exist_ok=True)
    sj = os.path.join(a.out, "suite.json")
    old = {r["case"]: r for r in json.load(open(sj))} if os.path.exists(sj) else {}
    if a.composite:
        n, mod, num = a.composite
        old[n]["composite"] = composite(n, a.out, mod, int(num))
        print(old[n]["composite"])
    elif a.report_only:
        for n in names:
            if n in old:
                old[n].update(check(n, C[n], os.path.join(a.out, n)))
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max(1, min(a.jobs, 2))) as ex:
            for r in ex.map(lambda n: run_case(n, C[n], a.out), names):
                old[r["case"]] = r
                print(f"{r['case']}: rc {r['rc']}, {r['seconds']} s, {r.get('n_files')} files, problems: "
                      f"{r.get('problems') or 'none'}", flush=True)
                json.dump(list(old.values()), open(sj, "w"), indent=1, default=str)
    json.dump(list(old.values()), open(sj, "w"), indent=1, default=str)
    report([old[n] for n in C if n in old], os.path.join(a.out, "suite.md"))
    print("report:", os.path.join(a.out, "suite.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
