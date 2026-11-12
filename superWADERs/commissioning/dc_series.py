"""Dark current over a series of images (run_superWADERs --dc-series): how the dark current of every CCD changes
with time, as numbers, a PDF report and an interactive page.

Per image and CCD (from the native fits of the run, native_pipeline.process_image):
  DC fit       lambda of the dark-current fit [e-/pix], with its error
  DC 0e-/1e-   from the counts of 0 e- and 1 e- pixels (autoconfig, independent of the fit)
  rate         DC / effective exposure, in e-/pix/day; effective exposure = EXPOSURE + half the readout (the mean
               time a pixel waits on the CCD; rows read later wait longer)
  row fit      the dark charge per row (from the 0e- fraction) against the row's waiting time
               (EXPOSURE + row x readout time / rows): slope = dark-current rate (e-/pix/day), intercept = charge that
               does not grow with time (made during the readout: clock-induced / serial-register charge)
Per CCD over the series: the rates against time since the first image, a fit rate(t) = plateau + A exp(-t / tau)
(5 or more images) and flags: rate above --dc-limit, still falling at the end of the series.
Outputs (out_dir/dc_series/): dc_series.csv, dc_report.pdf (summary table, one page per module with its CCDs, one
page per CCD), index.html + one interactive page per CCD (packet_report's page).
"""
import datetime
import math
import os

import numpy as np

DAY = 86400.0


def _time(s):
    try:
        return datetime.datetime.fromisoformat(str(s))
    except (TypeError, ValueError):
        return None


def collect(records, meta_of):
    """Per CCD {(acm, ext): [point dict, ...]} sorted by time. records: [(image path, native summary)];
    meta_of(path) -> header dict (upper-case keys)."""
    out = {}
    for path, s in records:
        m = meta_of(path)
        t0, t1 = _time(m.get("DATEINI")), _time(m.get("DATEEND"))
        exp = float(m.get("EXPOSURE", 0) or 0)
        tread = (t1 - t0).total_seconds() if t0 and t1 else float("nan")
        acm = m.get("ACMID")
        for ext, c in s["ccds"].items():
            if "gain" not in c:
                continue
            n = int(ext[3:])
            g = c["gain"]
            fl = c["fits"]["final"]
            dc, dce = fl["lam"][0] / g, (fl["lam"][1] or 0.0) / g
            teff = exp + tread / 2 if math.isfinite(tread) else exp
            p = dict(image=os.path.basename(path), time=t0, exposure=exp, readout=tread, dc=dc, dc_err=dce,
                     lam01=c.get("lam01"), noise=c["noise_e"], gain=g, resolved=bool(c.get("resolved")),
                     rate=dc / teff * DAY if teff > 0 else float("nan"),
                     rate_err=dce / teff * DAY if teff > 0 else float("nan"), ccd=m.get(f"CH{n - 1}") or f"ch{n - 1}")
            lr = np.array(c.get("lam_rows") or [], float)
            if lr.size > 5 and math.isfinite(tread) and tread > 0:
                tw = exp + np.arange(lr.size) * tread / lr.size
                ok = np.isfinite(lr)
                if ok.sum() > 5:
                    b, a = np.polyfit(tw[ok], lr[ok], 1)
                    p.update(row_wait=tw.tolist(), lam_rows=lr.tolist(), row_rate=b * DAY, row_intercept=a)
            out.setdefault((int(acm) if acm is not None else 0, n), []).append(p)
    for v in out.values():
        v.sort(key=lambda p: (p["time"] or datetime.datetime.min))
    return out


def settle_fit(t_h, r):
    """rate(t) = plateau + A exp(-t / tau) for 5 or more points; returns (plateau, A, tau [h]) or None."""
    from scipy.optimize import curve_fit
    ok = np.isfinite(r)
    if ok.sum() < 5:
        return None
    t_h, r = np.asarray(t_h)[ok], np.asarray(r)[ok]
    try:
        p, _ = curve_fit(lambda t, c, a, tau: c + a * np.exp(-t / tau), t_h, r,
                         p0=[r[-1], r[0] - r[-1], max(t_h[-1] / 3, 0.1)], maxfev=20000)
        ok_fit = p[0] >= 0 and 0 < p[2] < 10 * max(t_h[-1] - t_h[0], 1e-3)     # physical plateau, tau within reach
        return tuple(float(v) for v in p) if ok_fit else None
    except (RuntimeError, ValueError):
        return None


def summarise(pts, limit=None):
    """Numbers and flags of one CCD's series."""
    t0 = pts[0]["time"]
    th = np.array([((p["time"] - t0).total_seconds() / 3600 if p["time"] and t0 else np.nan) for p in pts])
    r = np.array([p["rate"] for p in pts])
    rr = np.array([p.get("row_rate", np.nan) for p in pts])
    fit = settle_fit(th, r)
    falling = len(r) >= 3 and np.all(np.diff(r[-3:]) < 0)
    over = limit is not None and np.nanmedian(r[-3:]) > limit
    return dict(t_h=th, rate=r, row_rate=rr, fit=fit, falling=bool(falling), over=bool(over),
                last=float(r[-1]), median=float(np.nanmedian(r)), n=len(pts))


def write_csv(series, mods_of, path):
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["module", "acm", "ext", "ccd", "image", "dateini", "exposure_s", "readout_s", "dc_fit_e_per_pix",
                    "dc_fit_err", "dc_0e1e_e_per_pix", "rate_e_per_pix_per_day", "row_fit_rate_e_per_pix_per_day",
                    "row_fit_intercept_e_per_pix", "noise_e", "gain", "resolved"])
        for (acm, ext), pts in sorted(series.items()):
            for p in pts:
                w.writerow([mods_of(acm, ext, p["ccd"]), acm, ext, p["ccd"], p["image"], p["time"], p["exposure"],
                            round(p["readout"], 1), f"{p['dc']:.6g}", f"{p['dc_err']:.3g}",
                            "" if p["lam01"] is None else f"{p['lam01']:.6g}", f"{p['rate']:.6g}",
                            f"{p.get('row_rate', float('nan')):.6g}", f"{p.get('row_intercept', float('nan')):.6g}",
                            f"{p['noise']:.4f}", f"{p['gain']:.5f}", p["resolved"]])


def write_pdf(series, sums, mods_of, path, limit=None):
    """dc_report.pdf: summary table, one page per module (its CCDs' rates vs time), one page per CCD."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams["text.usetex"] = False
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.05, 0.95, "Dark current over the series (run_superWADERs --dc-series)", fontsize=15, weight="bold")
        rows = []
        for (acm, ext), pts in sorted(series.items()):
            s = sums[(acm, ext)]
            fit = s["fit"]
            flags = ", ".join(x for x, on in (("above limit", s["over"]), ("still falling", s["falling"])) if on) or "-"
            rows.append([mods_of(acm, ext, pts[0]["ccd"]), str(acm), f"EXT{ext}", pts[0]["ccd"], str(s["n"]),
                         f"{s['median']:.3g}", f"{s['last']:.3g}",
                         f"{np.nanmedian(s['row_rate']):.3g}" if np.isfinite(s["row_rate"]).any() else "-",
                         f"{fit[0]:.3g} (tau {fit[2]:.2g} h)" if fit else "-", flags])
        ax = fig.add_axes([0.03, 0.05, 0.94, 0.85])
        ax.axis("off")
        tb = ax.table(cellText=rows or [["-"] * 10], colLabels=["module", "ACM", "ext", "CCD", "images",
                      "median rate", "last rate", "row-fit rate", "plateau (fit)", "flags"], loc="upper left",
                      cellLoc="left")
        tb.auto_set_font_size(False)
        tb.set_fontsize(8)
        tb.scale(1, 1.3)
        fig.text(0.05, 0.02, "rates in e-/pix/day: DC / (exposure + readout/2); row-fit: slope of the dark charge per row "
                 "against the row's waiting time" + (f"; limit {limit} e-/pix/day" if limit else ""), fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)
        mods = {}
        for (acm, ext), pts in series.items():
            mods.setdefault(mods_of(acm, None, None), []).append((acm, ext))
        for mod, keys in sorted(mods.items()):
            fig, ax = plt.subplots(figsize=(11, 7))
            for acm, ext in sorted(keys):
                s = sums[(acm, ext)]
                ax.errorbar(s["t_h"], s["rate"], yerr=[p["rate_err"] for p in series[(acm, ext)]], fmt="o-", ms=4,
                            label=f"EXT{ext} {series[(acm, ext)][0]['ccd']}")
            if limit:
                ax.axhline(limit, color="#c0392b", ls="--", lw=1, label="limit")
            ax.set(xlabel="hours since the first image", ylabel="dark current [e-/pix/day]", title=f"{mod}: dark current vs time")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)
        for (acm, ext), pts in sorted(series.items()):
            s = sums[(acm, ext)]
            fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
            name = f"{mods_of(acm, ext, pts[0]['ccd'])} ACM {acm} EXT{ext} {pts[0]['ccd']}"
            a = ax[0, 0]
            a.errorbar(s["t_h"], [p["dc"] for p in pts], yerr=[p["dc_err"] for p in pts], fmt="o-", ms=4, label="fit")
            a.plot(s["t_h"], [p["lam01"] if p["lam01"] is not None else np.nan for p in pts], "s", ms=4, label="0e-/1e- counts")
            a.set(xlabel="hours since the first image", ylabel="dark charge [e-/pix]", title="dark charge per image")
            a.legend(fontsize=8)
            a = ax[0, 1]
            a.plot(s["t_h"], s["rate"], "o-", ms=4, label="DC / (exposure + readout/2)")
            if np.isfinite(s["row_rate"]).any():
                a.plot(s["t_h"], s["row_rate"], "s--", ms=4, label="slope over the rows")
            if s["fit"]:
                tt = np.linspace(0, np.nanmax(s["t_h"]), 200)
                c0, a0, tau = s["fit"]
                a.plot(tt, c0 + a0 * np.exp(-tt / tau), ":", label=f"plateau {c0:.3g}, tau {tau:.2g} h")
            if limit:
                a.axhline(limit, color="#c0392b", ls="--", lw=1)
            a.set(xlabel="hours since the first image", ylabel="e-/pix/day", title="dark-current rate")
            a.legend(fontsize=8)
            a = ax[1, 0]
            cm = plt.get_cmap("viridis")
            for k, p in enumerate(pts):
                if "lam_rows" in p:
                    a.plot(np.array(p["row_wait"]) / 3600, p["lam_rows"], ".", ms=2, color=cm(k / max(len(pts) - 1, 1)))
            a.set(xlabel="time the row waited [h]", ylabel="dark charge of the row [e-/pix]",
                  title="dark charge per row (colour: image order)")
            a = ax[1, 1]
            a.axis("off")
            txt = [f"images: {s['n']}", f"median rate: {s['median']:.4g} e-/pix/day", f"last rate: {s['last']:.4g} e-/pix/day",
                   f"row-fit rate (median): {np.nanmedian(s['row_rate']):.4g} e-/pix/day" if np.isfinite(s["row_rate"]).any() else "row fit: -",
                   f"intercept (charge not growing with time), median: {np.nanmedian([p.get('row_intercept', np.nan) for p in pts]):.3g} e-/pix",
                   f"settling fit: plateau {s['fit'][0]:.4g} e-/pix/day, tau {s['fit'][2]:.3g} h" if s["fit"] else "settling fit: - (fewer than 5 images or no convergence)",
                   f"flags: {'still falling' if s['falling'] else ''} {'above limit' if s['over'] else ''}".strip(),
                   "", "first / last image:", pts[0]["image"], pts[-1]["image"]]
            a.text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=9)
            fig.suptitle(name)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def report(records, meta_of, mods_of, out_dir, limit=None, autoconfig_args=(), fit_cards=None):
    """--dc-series: csv, PDF and interactive pages. fit_cards: {(image file name, ext number): fit card} already
    made (campaign records); without it the cards are made here (one more fit per image). Returns (summary lines,
    overview rows: one per CCD with its numbers, flags and page)."""
    import packet_report as PR
    fits_of = dict(fit_cards or {})  # (image, ext) -> the dark-current fit card (histogram + model, numbers, profiles)
    for path, _ in ([] if fit_cards is not None else records):
        try:
            per_ext, _ = PR.image_data(path, autoconfig_args)
        except Exception as e:                       # the page is still made without this image's fit plots
            print(f"  dc-series: no fit plots for {os.path.basename(path)} ({type(e).__name__}: {e})")
            continue
        for n, (summ, _, _) in per_ext.items():
            fits_of[(os.path.basename(path), n)] = summ
    d = os.path.join(out_dir, "dc_series")
    os.makedirs(d, exist_ok=True)
    series = collect(records, meta_of)
    if not series:
        return ["dark-current series: no fitted CCD"], []
    sums = {k: summarise(v, limit) for k, v in series.items()}
    write_csv(series, mods_of, os.path.join(d, "dc_series.csv"))
    write_pdf(series, sums, mods_of, os.path.join(d, "dc_report.pdf"), limit)
    lines = [f"dark-current series (--dc-series; {d}/: dc_report.pdf, dc_series.csv, index.html):"]
    modules = {}                     # module -> [(acm, ext, ccd, tabs, numbers, flags)]: one page per module
    for (acm, ext), pts in sorted(series.items()):
        s = sums[(acm, ext)]
        ccd, mod = pts[0]["ccd"], mods_of(acm, ext, pts[0]["ccd"])
        flags = [x for x, on in (("above limit", s["over"]), ("still falling", s["falling"])) if on]
        nums = [["images", str(s["n"])], ["median rate", f"{s['median']:.4g} e-/pix/day", s["over"]],
                ["last rate", f"{s['last']:.4g} e-/pix/day", s["falling"]],
                ["settling fit", f"plateau {s['fit'][0]:.4g} e-/pix/day, tau {s['fit'][2]:.3g} h" if s["fit"] else "-"]]
        enc = PR.enc
        th = np.nan_to_num(s["t_h"])
        ser = lambda y, name, style="points": dict(x=enc(th, "f8"), y=enc(np.array(y, float)), name=name, style=style)
        nan = float("nan")
        plots = [dict(title="dark-current rate vs time", xlabel="hours since the first image", ylabel="e-/pix/day", logy=True,
                      series=[ser(s["rate"], "DC / (exposure + readout/2)", "line")]
                      + ([ser(s["row_rate"], "slope of the dark charge over the rows", "line")] if np.isfinite(s["row_rate"]).any() else [])),
                 dict(title="dark charge per image", xlabel="hours since the first image", ylabel="e-/pix",
                      series=[ser([p["dc"] for p in pts], "dark-current fit", "line"),
                              ser([p["lam01"] if p["lam01"] is not None else nan for p in pts], "0e-/1e- pixel counts")]),
                 dict(title="charge not growing with time (row-fit intercept)", xlabel="hours since the first image", ylabel="e-/pix",
                      series=[ser([p.get("row_intercept", nan) for p in pts], "intercept", "line")]),
                 dict(title="read noise per image", xlabel="hours since the first image", ylabel="e-",
                      series=[ser([p["noise"] for p in pts], "noise", "line")]),
                 dict(title="gain per image", xlabel="hours since the first image", ylabel="ADU/e-",
                      series=[ser([p["gain"] for p in pts], "gain", "line")]),
                 dict(title="effective exposure per image", xlabel="hours since the first image", ylabel="hours",
                      series=[ser([(p["exposure"] + p["readout"] / 2) / 3600 for p in pts], "exposure + readout/2", "line")])]
        rowplot = [dict(title="dark charge per row against the time the row waited (one series per image)",
                        xlabel="time the row waited [h]", ylabel="dark charge of the row [e-/pix]",
                        series=[dict(x=enc(np.array(p["row_wait"]) / 3600, "f8"), y=enc(np.array(p["lam_rows"], float)),
                                     name=p["image"][-28:], style="points") for p in pts if "lam_rows" in p])]
        head = ["image", "start (DATEINI)", "exposure [s]", "readout [s]", "DC fit [e-/pix]", "+-", "DC 0e-/1e- [e-/pix]",
                "rate [e-/pix/day]", "row-fit rate [e-/pix/day]", "row-fit intercept [e-/pix]", "gain [ADU/e-]", "noise [e-]", "peaks resolved"]
        fmt = lambda v, f="{:.4g}": "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f.format(v)
        rows = [[p["image"], str(p["time"])[:19], fmt(p["exposure"], "{:.0f}"), fmt(p["readout"], "{:.0f}"), fmt(p["dc"]),
                 fmt(p["dc_err"], "{:.2g}"), fmt(p["lam01"]), fmt(p["rate"]), fmt(p.get("row_rate")), fmt(p.get("row_intercept")),
                 fmt(p["gain"], "{:.4f}"), fmt(p["noise"], "{:.3f}"), "yes" if p["resolved"] else "no"] for p in pts]
        method = [
            "Every number comes from the native fits of this run (run_superWADERs: the same pedestal, dark-current fit and gain "
            "choice of gain_dc_split, one per image and CCD; their details are in native_summary.json and summary.txt of the run).",
            "DC fit: lambda of the dark-current fit (Gaussian electron peaks of width sigma, Poisson weights of mean lambda) on "
            "the pedestal-subtracted pixels of the active area, divided by the gain: dark charge per pixel [e-/pix], with the fit "
            "error. DC 0e-/1e-: from the numbers of 0 e- and 1 e- pixels (autoconfig, independent of the fit).",
            "Times from the image header: DATEINI / DATEEND (start and end of the readout) and EXPOSURE. Every pixel collects "
            "charge during the exposure and while it waits to be read: rows read later wait longer. Rate = DC / (EXPOSURE + "
            "readout time / 2), the mean time on the CCD, x 86400 s -> e-/pix/day.",
            "Row fit: the dark charge of each row (from the fraction of its pixels with 0 e-: lambda = -ln P(0)) against the "
            "time that row waited (EXPOSURE + row x readout time / rows). Slope = dark-current rate (independent of the "
            "exposure bookkeeping); intercept = charge that does not grow with time (made during the readout: clock-induced "
            "or serial-register charge).",
            "Settling fit (5 or more images): rate(t) = plateau + A exp(-t / tau); accepted only with plateau >= 0 and tau < 10 x "
            "the time span. Flags: 'still falling' when the last three rates decrease; 'above limit' when the median of the last "
            "three rates exceeds --dc-limit.",
            "Note: images of different types (exposure, skips, binning) are mixed in the series as given; their rates are "
            "comparable, their dark charge per image is not."]
        log = [f"{p['image']}: DATEINI {p['time']}, exposure {p['exposure']} s, readout {p['readout']:.0f} s, DC {p['dc']:.4g} "
               f"+- {p['dc_err']:.2g} e-/pix, rate {p['rate']:.4g} e-/pix/day" for p in pts]
        modules.setdefault(mod, []).append((acm, ext, ccd,
                             [dict(id="dc", label="Rates", kind="plots", data=dict(title="Dark current over the series",
                                                                                   numbers=nums, plots=plots), log=log),
                              dict(id="fits", label="Fits", kind="summary",
                                   data=[fits_of[(p["image"], ext)] for p in pts if (p["image"], ext) in fits_of],
                                   log=[f"{p['image']}: dark-current fit of the image (native engine: pedestal, comb fit, gain "
                                        f"choice of gain_dc_split)" for p in pts]),
                              dict(id="rows", label="Per row", kind="plots", data=dict(title="Dark charge per row", plots=rowplot),
                                   log=[f"{p['image']}: row fit slope {fmt(p.get('row_rate'))} e-/pix/day, intercept "
                                        f"{fmt(p.get('row_intercept'))} e-/pix" for p in pts]),
                              dict(id="data", label="Data", kind="plots", data=dict(title="The numbers of every image",
                                                                                   table=dict(head=head, rows=rows), plots=[]), log=log),
                              dict(id="method", label="Method", kind="plots", data=dict(title="Where these numbers come from",
                                                                                       text=method, plots=[]), log=[])],
                                            nums, flags))
        lines.append(f"  {mod} ACM {acm} EXT{ext} {ccd}: {s['n']} images, median {s['median']:.4g}, last {s['last']:.4g} e-/pix/day"
                     + (f", plateau {s['fit'][0]:.4g} (tau {s['fit'][2]:.3g} h)" if s["fit"] else "") + (f"  [{', '.join(flags)}]" if flags else ""))
    ov = []
    for mod, items in sorted(modules.items()):
        rid, name, ccds, tabs = PR.module_page(mod, [it[:4] for it in items])
        PR.write_page(os.path.join(d, rid + "_dc.html"), rid + "_dc", f"{name} · dark current",
                      "dark current over the series (run_superWADERs --dc-series)", tabs, ccds, back="index.html")
        for acm, ext, ccd, _, _, _ in items:
            s, pts = sums[(acm, ext)], series[(acm, ext)]
            fin = lambda v: float(v) if v is not None and np.isfinite(v) else None
            ri = [p.get("row_intercept") for p in pts if p.get("row_intercept") is not None and np.isfinite(p.get("row_intercept"))]
            ov.append(dict(module=mod, acm=acm, ext=f"EXT{ext}", ccd=ccd, page=rid + "_dc.html", falling=s["falling"],
                           over=s["over"], when=str(pts[-1]["time"])[:19],
                           values=dict(images=s["n"], rate_median=fin(s["median"]), rate_last=fin(s["last"]),
                                       plateau=fin(s["fit"][0]) if s["fit"] else None, tau_h=fin(s["fit"][2]) if s["fit"] else None,
                                       row_intercept_median=float(np.median(ri)) if ri else None,
                                       span_h=fin(s["t_h"][-1]) if len(s["t_h"]) else None,
                                       noise_median=float(np.median([p["noise"] for p in pts]))),
                           history=[[p["image"], str(p["time"])[:19], p["dc"], p["rate"], p.get("row_rate"), p["noise"]] for p in pts]))
    return lines, ov


FIELDS = {"images": ("images", ""), "rate_median": ("dark current (median)", "e-/pix/day"),
          "rate_last": ("dark current (last image)", "e-/pix/day"), "plateau": ("settling-fit plateau", "e-/pix/day"),
          "tau_h": ("settling time constant", "h"), "row_intercept_median": ("readout charge (row-fit intercept, median)", "e-/pix"),
          "span_h": ("time span of the series", "h"), "noise_median": ("read noise (median)", "e-")}


def overview(camp, rows, crit, path, back, limit=None):
    """dc_series/index.html: the module x CCD overview of the dark-current series. Returns the view."""
    import campaign as CA
    sec = CA.section(crit, "dc_series")
    for r in rows:
        flags = []
        if r.pop("falling", False):
            flags.append(("falling", "warn", "dark current still falling at the end of the series (last three rates "
                                             "decrease)", "still falling"))
        if r.pop("over", False):
            flags.append(("over_limit", "fail", f"dark current above --dc-limit {limit} e-/pix/day (median of the last "
                                                f"three images)", "above --dc-limit"))
        r["verdict"], r["reasons"], r["levels"] = CA.judge(r["values"], sec, flags)
        r["links"] = [["module page", r.pop("page")]]
        h = r.pop("history")
        r["history"] = dict(head=["image", "start", "DC [e-/pix]", "rate [e-/pix/day]", "row-fit rate", "noise [e-]"],
                            rows=h) if len(h) > 1 else None
    view = dict(name="dark-current series", ccds=rows)
    CA.write_overview(path, dict(
        title="Dark current over the series", sub=CA.sub_line(camp), back=back, criteria=sec,
        fields={k: dict(label=a, unit=b) for k, (a, b) in FIELDS.items()},
        sections=[["Dark current", ["rate_median", "rate_last", "plateau", "tau_h", "row_intercept_median"]],
                  ["Series", ["images", "span_h", "noise_median"]]],
        cell=["rate_last", "rate_median", "plateau", "images"], views=[view],
        columns=["images", "span_h", "rate_median", "rate_last", "plateau", "tau_h", "row_intercept_median", "noise_median"],
        definitions=[("rate", "Dark charge of the fit / (exposure + half the readout) of each image, in e-/pix/day."),
                     ("settling fit", "rate(t) = plateau + A exp(-t / tau) with 5 or more images (shown when physical)."),
                     ("row-fit intercept", "Charge that does not grow with the time a row waits: made during the readout."),
                     ("module page", "Rates, every image's fit, the dark charge per row, all numbers and the method "
                                     "(dc_series/<module>_..._dc.html); dc_report.pdf and dc_series.csv hold the same.")],
        flag_labels={"falling": "still falling", "over_limit": "above --dc-limit"}, csv_name="dc_series_overview"))
    return view
