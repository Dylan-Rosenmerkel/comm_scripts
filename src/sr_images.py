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
page per module (CCD check boxes; streak counts, DC per read, the image with the streaks marked with a slider over the
CCDs, a clickable streak list).
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
                                                              lam_rows=c.get("lam_rows"), dc=c["dc_e"], noise=noise, rows=ps.shape[0]))
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
        # figure: counts per image, per read, lengths, positions, DC per read
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
        png = os.path.join(d, rid + ".png")
        fig.savefig(png, dpi=100)
        plt.close(fig)
        import base64
        tabs = [dict(id="sr", label="Streaks", kind="figure", data=dict(
            title="Serial-register streaks and dark charge per read",
            numbers=[[it["image"], f"{len(it['streaks'])} streaks in {it['rows']} reads ({1000 * len(it['streaks']) / max(it['rows'], 1):.3g} "
                      f"per 1000), threshold {it['thr']:.3g} {it['unit']}, DC {it['dc']:.4g} e-/pix"] for it in items],
            figures=[dict(caption="counts, lengths, positions, dark charge per read", png=base64.b64encode(open(png, "rb").read()).decode())]),
            log=[f"{it['image']}: noise {it['noise']:.3g} {it['unit']}, threshold {it['thr']:.3g} {it['unit']}, min length {min_len}" for it in items])]
        for k, it in enumerate(items):
            mask = np.zeros(it["img"].shape, np.uint8)
            for r in it["streaks"]:
                mask[r[0], r[1]:r[2] + 1] = 1
            tabs.append(dict(id=f"img{k}", label=f"image {k + 1}", kind="defects", log=[it["image"]], data=dict(
                title=f"{it['image']}: streaks marked", unit=it["unit"], w=int(it["img"].shape[1]), h=int(it["img"].shape[0]),
                pixels=PR.enc(it["img"]), mask=PR.enc(mask, "u1"), types=[dict(name="streak", color="#ff3b30", n=len(it["streaks"]))],
                defects=[["streak", r[1], r[0], r[3], r[4], it["unit"]] for r in sorted(it["streaks"], key=lambda r: -r[4])[:1500]],
                defects_more=max(0, len(it["streaks"]) - 1500), numbers=[["streaks", str(len(it["streaks"]))], ["reads", str(it["rows"])]])))
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
