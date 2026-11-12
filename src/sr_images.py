"""Serial-register (SR) images (run_superWADERs --sr): images of the horizontal register only. Every image "row" is
one more readout of the register, so rows are reads in time, not positions on the CCD.

Serial-register events are streaks in the serial register: readout artifacts, i.e. runs of charged pixels along a
row that do not continue into the rows before / after (a particle track in the active area would). Per CCD and
image (pedestal and fits from the native engine):
  streak finder   pixels above --sr-threshold x noise (in e- when the electron peaks are resolved, else ADU),
                  runs of >= --sr-min-length consecutive pixels along a row; a run is a streak when at most 20 % of
                  the pixels just above and below it are also above threshold (confined to its row)
  numbers         streaks per image, per read (row) and per 1000 reads, lengths, charge, position along the register
  DC plots        dark charge per read (row, from the 0e- fraction) against the read number / time, and the fit's DC
Outputs (out_dir/sr/): sr_streaks.csv (one line per streak), sr_summary.csv, per CCD a PNG, and one interactive
page per module (CCD check boxes): a Diagnostics tab (per image: streaks, streaks per 1000 reads, dark charge (fit and
0e-/1e- counts), read noise, gain; dark charge per read; streak lengths and start positions; a table of every
number) and one tab per image with the streaks marked (frame slider over the CCDs; no streak list on the page).
Tested only on synthetic data so far (no SR image files yet): thresholds may need tuning on real SR images.
"""
import os

import numpy as np


def find_streaks(img, active, thr, min_len=3, max_neighbour=0.2):
    """Streaks in img (2-D, in units where thr applies): runs along a row of >= min_len pixels > thr, with at most
    max_neighbour of the pixels directly above / below them > thr. Returns [(row, col0, col1, length, sum, peak)]."""
    hi = (img > thr) & active
    out = []
    h, w = img.shape
    for r in range(h):
        row = hi[r]
        if not row.any():
            continue
        d = np.diff(np.r_[0, row.astype(np.int8), 0])
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        for c0, c1 in zip(starts, ends):
            if c1 - c0 < min_len:
                continue
            nb = []
            if r > 0:
                nb.append(hi[r - 1, c0:c1])
            if r < h - 1:
                nb.append(hi[r + 1, c0:c1])
            frac = np.mean(np.concatenate(nb)) if nb else 0.0
            if frac <= max_neighbour:
                seg = img[r, c0:c1]
                out.append((r, int(c0), int(c1 - 1), int(c1 - c0), float(seg.sum()), float(seg.max())))
    return out


def run(images, summaries, out_dir, mods_of, meta_of, thr_sigma=4.0, min_len=3, autoconfig_args=(), log=print):
    """--sr on the run's images (native summaries with keep_arrays). Returns summary lines."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import native_pipeline as NP
    import packet_report as PR
    plt.rcParams["text.usetex"] = False
    d = os.path.join(out_dir, "sr")
    os.makedirs(d, exist_ok=True)
    per_ccd, rows_csv, sum_csv = {}, [], []
    for p in images:
        log(f"  sr {os.path.basename(p)}")
        s = (summaries or {}).get(p) or NP.process_image(p, os.devnull, outputs=[], write=False, keep_arrays=True,
                                                          all_ext=True, verbose=False, autoconfig_args=autoconfig_args)
        m = meta_of(p)
        acm = int(m.get("ACMID") or PR.acm_of(p) or 0)
        geo = NP.read_image(p, autoconfig_args)["geo"]
        for ext, c in s["ccds"].items():
            if "ps" not in c:
                continue
            n = int(ext[3:])
            ps, g = c["ps"], c["gain"]
            active, _ = NP.regions(ps.shape, geo)
            resolved = bool(c.get("resolved")) and c["noise_e"] < 0.35
            img, unit = (ps / g, "e-") if resolved else (ps, "ADU")
            noise = c["noise_e"] if resolved else float(np.nanmedian(c["sig_row"]))
            thr = thr_sigma * noise if not resolved else max(thr_sigma * noise, 1.5)
            st = find_streaks(img, active, thr, min_len)
            ccd = m.get(f"CH{n - 1}") or f"ch{n - 1}"
            per_ccd.setdefault((acm, n, ccd), []).append(dict(image=os.path.basename(p), img=img, unit=unit, thr=thr, streaks=st,
                                                              lam_rows=c.get("lam_rows"), dc=c["dc_e"], noise=noise, rows=ps.shape[0],
                                                              noise_e=c["noise_e"], gain=g, gain_err=c.get("gain_err"),
                                                              lam01=c.get("lam01"), resolved=resolved))
            for r in st:
                rows_csv.append([mods_of(acm, n, ccd), acm, n, ccd, os.path.basename(p), *r[:4], f"{r[4]:.4g}", f"{r[5]:.4g}", unit])
            sum_csv.append([mods_of(acm, n, ccd), acm, n, ccd, os.path.basename(p), ps.shape[0], len(st),
                            f"{1000 * len(st) / max(ps.shape[0], 1):.3g}", f"{np.median([r[3] for r in st]) if st else 0:.3g}",
                            f"{c['dc_e']:.4g}", f"{thr:.3g} {unit}"])
    with open(os.path.join(d, "sr_streaks.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["module", "acm", "ext", "ccd", "image", "read_row", "col_first", "col_last", "length", "sum", "peak", "unit"])
        w.writerows(rows_csv)
    with open(os.path.join(d, "sr_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["module", "acm", "ext", "ccd", "image", "reads", "streaks", "streaks_per_1000_reads", "median_length",
                    "dc_fit_e_per_pix", "threshold"])
        w.writerows(sum_csv)
    lines = [f"serial-register images (--sr; streaks = runs >= {min_len} pixels > {thr_sigma} x noise along a read, confined "
             f"to it; {d}/):"]
    modules = {}                         # module -> [(acm, ext, ccd, tabs, streaks)]: one page per module
    for (acm, n, ccd), items in sorted(per_ccd.items()):
        mod = mods_of(acm, n, ccd)
        rid = f"{mod}_ACM{acm}_EXT{n}_{ccd}_sr"
        # figure (PNG file): counts per image, lengths, positions, DC per read
        fig, ax = plt.subplots(2, 2, figsize=(13, 8))
        ax[0, 0].bar(range(len(items)), [len(it["streaks"]) for it in items])
        ax[0, 0].set(xlabel="image (time order of the list)", ylabel="streaks", title="serial-register streaks per image")
        allst = [r for it in items for r in it["streaks"]]
        ax[0, 1].hist([r[3] for r in allst] or [0], bins=30)
        ax[0, 1].set(xlabel="length [pixels]", ylabel="streaks", title="streak lengths")
        ax[1, 0].hist([r[1] for r in allst] or [0], bins=50)
        ax[1, 0].set(xlabel="first column (position along the register)", ylabel="streaks", title="where the streaks start")
        for it in items:
            if it["lam_rows"]:
                ax[1, 1].plot(it["lam_rows"], ".", ms=2, label=it["image"][-20:])
        ax[1, 1].set(xlabel="read (row)", ylabel="dark charge per pixel [e-]", title="dark charge per read")
        fig.suptitle(f"{mod} ACM {acm} EXT{n} {ccd}: serial-register images")
        fig.tight_layout()
        fig.savefig(os.path.join(d, rid + ".png"), dpi=100)
        plt.close(fig)
        # Diagnostics tab: interactive plots per image (CCDs overlaid on the module page), a table of every number
        enc = PR.enc
        x = np.arange(1, len(items) + 1, dtype=float)
        nan = float("nan")
        ser = lambda y, name, style="line": dict(x=enc(x, "f8"), y=enc(np.array(y, float)), name=name, style=style)
        lens = np.array([r[3] for r in allst], float)
        cols = np.array([r[1] for r in allst], float)
        def hist(v, bins):
            if not v.size:
                return dict(x=enc(np.zeros(1), "f8"), y=enc(np.zeros(1)), name="streaks", style="step")
            h, e = np.histogram(v, bins)
            return dict(x=enc(e[:-1], "f8"), y=enc(h.astype(float)), name="streaks", style="step")
        plots = [dict(title="streaks per image", xlabel="image (order of the list)", ylabel="streaks",
                      series=[ser([len(it["streaks"]) for it in items], "streaks")]),
                 dict(title="streaks per 1000 reads", xlabel="image (order of the list)", ylabel="per 1000 reads",
                      series=[ser([1000 * len(it["streaks"]) / max(it["rows"], 1) for it in items], "per 1000 reads")]),
                 dict(title="dark charge per image", xlabel="image (order of the list)", ylabel="e-/pix",
                      series=[ser([it["dc"] for it in items], "dark-current fit"),
                              ser([it["lam01"] if it["lam01"] is not None else nan for it in items], "0e-/1e- pixel counts", "points")]),
                 dict(title="read noise per image", xlabel="image (order of the list)", ylabel="e-",
                      series=[ser([it["noise_e"] for it in items], "noise")]),
                 dict(title="gain per image", xlabel="image (order of the list)", ylabel="ADU/e-",
                      series=[ser([it["gain"] for it in items], "gain")]),
                 dict(title="dark charge per read (one series per image)", xlabel="read (row)", ylabel="e-/pix",
                      series=[dict(x=enc(np.arange(len(it["lam_rows"]), dtype=float), "f8"), y=enc(np.array(it["lam_rows"], float)),
                                   name=it["image"][-28:], style="points") for it in items if it["lam_rows"]]),
                 dict(title="streak lengths (all images)", xlabel="length [pixels]", ylabel="streaks",
                      series=[hist(lens, np.arange(min_len, max(lens.max() if lens.size else min_len, min_len) + 2))]),
                 dict(title="where the streaks start (all images)", xlabel="first column", ylabel="streaks",
                      series=[hist(cols, 50)])]
        fmt = lambda v, f="{:.4g}": "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f.format(v)
        table = dict(head=["#", "image", "reads", "streaks", "per 1000 reads", "median length", "DC fit [e-/pix]",
                           "DC 0e-/1e- [e-/pix]", "noise [e-]", "gain [ADU/e-]", "peaks resolved", "threshold"],
                     rows=[[str(k + 1), it["image"], str(it["rows"]), str(len(it["streaks"])),
                            fmt(1000 * len(it["streaks"]) / max(it["rows"], 1), "{:.3g}"),
                            fmt(float(np.median([r[3] for r in it["streaks"]])) if it["streaks"] else None, "{:.3g}"),
                            fmt(it["dc"]), fmt(it["lam01"]), fmt(it["noise_e"], "{:.3f}"), fmt(it["gain"], "{:.4f}"),
                            "yes" if it["resolved"] else "no", f"{it['thr']:.3g} {it['unit']}"] for k, it in enumerate(items)])
        nst_all = len(allst)
        nums = [["images", str(len(items))], ["streaks", f"{nst_all} ({1000 * nst_all / max(sum(it['rows'] for it in items), 1):.3g} per 1000 reads)"],
                ["median dark charge", f"{np.median([it['dc'] for it in items]):.4g} e-/pix"],
                ["median read noise", f"{np.median([it['noise_e'] for it in items]):.4g} e-", bool(np.median([it['noise_e'] for it in items]) > 0.3)],
                ["median gain", f"{np.median([it['gain'] for it in items]):.4g} ADU/e-"]]
        tabs = [dict(id="diag", label="Diagnostics", kind="plots", data=dict(
            title="Diagnostics: streaks, dark charge, noise and gain per image", numbers=nums, table=table, plots=plots),
            log=[f"{it['image']}: noise {it['noise']:.3g} {it['unit']}, threshold {it['thr']:.3g} {it['unit']}, min length {min_len}, "
                 f"{len(it['streaks'])} streaks, DC {it['dc']:.4g} e-/pix, gain {it['gain']:.4f} ADU/e-" for it in items])]
        for k, it in enumerate(items):
            mask = np.zeros(it["img"].shape, np.uint8)
            for r in it["streaks"]:
                mask[r[0], r[1]:r[2] + 1] = 1
            tabs.append(dict(id=f"img{k}", label=f"image {k + 1}", kind="defects", log=[it["image"]], data=dict(
                title=f"{it['image']}: streaks marked", unit=it["unit"], w=int(it["img"].shape[1]), h=int(it["img"].shape[0]),
                pixels=PR.enc(it["img"]), mask=PR.enc(mask, "u1"), types=[dict(name="streak", color="#ff3b30", n=len(it["streaks"]))],
                defects=None, numbers=[["streaks", str(len(it["streaks"]))], ["reads", str(it["rows"])],
                                       ["threshold", f"{it['thr']:.3g} {it['unit']}"]])))
        nst = sum(len(it["streaks"]) for it in items)
        modules.setdefault(mod, []).append((acm, n, ccd, tabs, nst))
        lines.append(f"  {mod} ACM {acm} EXT{n} {ccd}: {nst} streaks in {len(items)} image(s)")
    index = []
    for mod, its in sorted(modules.items()):
        rid, name, ccds, tabs = PR.module_page(mod, [it[:4] for it in its])
        size = PR.write_page(os.path.join(d, rid + "_sr.html"), rid + "_sr", f"{name} · serial register",
                             "serial-register images (run_superWADERs --sr)", tabs, ccds)
        index.append((mod, ", ".join(sorted({str(a) for a, *_ in its})), ", ".join(f"EXT{e}" for _, e, *_ in its),
                      ", ".join(c for _, _, c, *_ in its), rid + "_sr.html", [t["label"] for t in tabs],
                      [[f"EXT{e} streaks", str(k)] for _, e, _, _, k in its], size, []))
    PR.write_index(d, index, "Serial-register images")
    return lines
