"""Markdown tables from the benchmark CSVs (results/images.csv, events.csv, fakes.csv) -> results/tables.md (wip/lowdc).

    python summarize.py
"""
import os
import numpy as np
import pandas as pd
import metrics as MT

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
FEW = 20.0           # e-: unmatched objects below this charge are few-electron fakes


def _pe(k, n):
    n = max(n, 1); p = k / n
    return f"{p:.2f}±{np.sqrt(max(p * (1 - p), 1.0 / n) / n):.2f}"


def eff_table(ev, configs, col="found", label="A_GMX"):
    rows = []
    for c in configs:
        e = ev[(ev.config == c) & (ev.kind == "fe")]
        if e.empty:
            continue
        r = {"config": c}
        for n in range(1, 11):
            g = e[e.n_e == n]
            r[f"{n}e"] = _pe(g[col].sum(), len(g)) if len(g) else ""
        r["N/n_e"] = int(len(e) / 10)
        rows.append(r)
    return f"\n**{label}** (found = at least one object owns the event's electrons)\n\n" + MT.md_table(pd.DataFrame(rows)) + "\n"


def sigma_table(ev, configs):
    rows = []
    bins = [(0.2, 0.5), (0.5, 0.8), (0.8, 1.21)]
    for c in configs:
        e = ev[(ev.config == c) & (ev.kind == "fe")]
        for n in (2, 3, 4, 5):
            r = {"config": c, "n_e": n}
            for lo, hi in bins:
                g = e[(e.n_e == n) & (e.sigma >= lo) & (e.sigma < hi)]
                r[f"sig {lo}-{hi:.1f}"] = _pe(g.found.sum(), len(g)) if len(g) else ""
            g = e[e.n_e == n]
            r["n_pix=1 frac"] = f"{(g.n_pix == 1).mean():.2f}"
            r["eff(n_pix=1)"] = _pe(g[g.n_pix == 1].found.sum(), (g.n_pix == 1).sum())
            r["eff(n_pix>1)"] = _pe(g[g.n_pix > 1].found.sum(), (g.n_pix > 1).sum())
            rows.append(r)
    return MT.md_table(pd.DataFrame(rows)) + "\n"


def response_table(ev, configs):
    e = ev[ev.config.isin(configs) & (ev.kind == "fe") & (ev.found == 1)]
    rows = []
    for n in range(1, 11):
        g = e[e.n_e == n]
        if g.empty:
            continue
        d = g.q - n
        rows.append(dict(n_e=n, N=len(g), mean_q=f"{g.q.mean():.2f}", exact=f"{(d.abs() < 0.5).mean():.2f}",
                         low=f"{(d < -0.5).mean():.2f}", high=f"{(d > 0.5).mean():.2f}", split=f"{(g.n_obj > 1).mean():.3f}",
                         lowE_cls=f"{g.lowE.mean():.2f}"))
    return MT.md_table(pd.DataFrame(rows)) + "\n"


def key(c):
    return c.split("_", 1)[1]           # s.._l..


def fake_table(img, fk, groups):
    """groups: label -> list of configs pooled (fake area = all their images)"""
    rows = []
    for lab, cs in groups.items():
        im = img[img.config.isin(cs)]
        if im.empty:
            continue
        mp = im.mpix.sum()
        f = fk[fk.config.isin(cs) & (fk.near_big == 0)]
        a = f[(f.src == "agmx") & (f.q < FEW)]
        c = f[(f.src == "cand") & (f.clean == 1)]
        r = dict(group=lab, images=len(im), Mpix=f"{mp:.1f}",
                 agmx_fakes=len(a), agmx_per_Mpix=f"{len(a) / mp:.3f}±{np.sqrt(max(len(a), 1)) / mp:.3f}",
                 agmx_1px=int((a.npix == 1).sum()), agmx_q2=int(((a.q > 1.5) & (a.q < 2.5)).sum()),
                 cand_clean_per_Mpix=f"{len(c) / mp:.2f}", cand_exp_per_Mpix=f"{im.cand_exp_clean.sum() / mp:.2f}")
        r["cand_clean_nexp<0.05_per_Mpix"] = f"{(c.n_exp < 0.05).sum() / mp:.3f}"
        r["thr_T"] = f"{im.thr_T.mean():.1f}±{im.thr_T.std():.1f}"
        rows.append(r)
    return MT.md_table(pd.DataFrame(rows)) + "\n"


def lam_table(img, configs):
    rows = []
    for c in configs:
        im = img[img.config == c]
        if im.empty:
            continue
        rows.append(dict(config=c, lam_true=f"{im.lam_true.mean():.1e}", agmx_lam=f"{im.agmx_lam.median():.1e}",
                         bg_lam_mean=f"{im.bg_lam_mean.mean():.1e}", bg_lam_p90=f"{im.bg_lam_p90.mean():.1e}",
                         det_cons=f"{im.det_lam_cons.mean():.1e}±{im.det_lam_cons.std():.0e}",
                         det_occ=f"{im.det_lam_occ.mean():.1e}±{im.det_lam_occ.std():.0e}",
                         noise=f"{im.agmx_noise.mean():.3f}", s_true=im.s_true.iloc[0],
                         hotcol_flag=f"{im.hotcol_found.mean():.2f}"))
    return MT.md_table(pd.DataFrame(rows)) + "\n"


def runtime_table(img):
    img = img.assign(geom=img.H.astype(str) + "x" + img.W.astype(str), kind=img.config.str.split("_").str[0])
    g = img.groupby(["geom", "kind"])[["t_bg", "t_agmx", "t_fe"]].mean().round(2)
    g["n"] = img.groupby(["geom", "kind"]).size()
    return MT.md_table(g.reset_index()) + "\n"


def artifacts(fk, img):
    out = []
    for c in ("art_s0.10_l1e-05", "base_s0.10_l1e-05"):
        f = fk[(fk.config == c) & (fk.src == "agmx")]
        n = (img.config == c).sum()
        out.append(f"- {c} ({n} images): A_GMX unmatched objects {len(f)}; at x < 2 (first column) {int((f.x < 2).sum())} "
                   f"(cls: {f[f.x < 2].cls.value_counts().to_dict()}); elsewhere {int((f.x >= 2).sum())} "
                   f"(q: {sorted(f[f.x >= 2].q.round(1).tolist())[:40]})")
    for c in [c for c in img.config.unique() if c.startswith("hotcol")]:
        f = fk[(fk.config == c) & (fk.src == "agmx") & (fk.near_big == 0)]
        hc = f[(f.x - 640).abs() < 1.5]
        fc = fk[(fk.config == c) & (fk.src == "cand") & (fk.clean == 1)]
        out.append(f"- {c}: A_GMX fakes {len(f)}, of them in the hot column {len(hc)} (q {sorted(hc.q.round(1).tolist())[:15]}); "
                   f"clean candidates unmatched {len(fc)} (in hot col {int(((fc.x - 640).abs() < 1.5).sum())})")
    return "\n".join(out) + "\n"


def main():
    img = pd.read_csv(os.path.join(OUT, "images.csv"))
    ev = pd.read_csv(os.path.join(OUT, "events.csv"))
    fk = pd.read_csv(os.path.join(OUT, "fakes.csv"))
    cfg = sorted(img.config.unique())
    grid = [c for c in cfg if not c.startswith(("blank", "l2"))]
    order = lambda cs: sorted(cs, key=lambda c: (c.split("_")[0], float(c.split("_s")[1].split("_")[0]), float(c.split("_l")[1])))
    md = ["# A_GMX low-dark-current benchmark: tables (generated by summarize.py)\n"]
    md.append("## Efficiency vs n_e\n")
    md.append(eff_table(ev, order(grid) + order([c for c in cfg if c.startswith("l2")])))
    md.append(eff_table(ev, order(grid) + order([c for c in cfg if c.startswith("l2")]), col="cand_any",
                        label="fe_candidates: any candidate (2-20 e-, <= 9 px) owns the event"))
    md.append("\n## Efficiency vs diffusion sigma (A_GMX)\n\n")
    md.append(sigma_table(ev, order([c for c in grid if c.startswith("base")])))
    md.append("\n## Charge response of found events (A_GMX; base configs pooled)\n\n")
    md.append(response_table(ev, [c for c in grid if c.startswith("base")]))
    md.append("\n## Fakes per Mpix per image (unmatched, q < 20 e-, not near a big deposit)\n\n")
    groups = {}
    for c in cfg:
        kind, k = c.split("_", 1)
        if kind in ("base", "blank"):
            groups.setdefault(f"plain {k}", []).append(c)
        elif kind == "l2":
            groups.setdefault(f"L2 100x500 {k}", []).append(c)
        else:
            groups.setdefault(c, []).append(c)
    md.append(fake_table(img, fk, dict(sorted(groups.items()))))
    md.append("\n## Dark charge / noise estimates vs truth (per-image mean; agmx_lam = median of bg.lam)\n\n")
    md.append(lam_table(img, order(cfg)))
    md.append("\n## Runtime per image [s] (single thread)\n\n")
    md.append(runtime_table(img))
    md.append("\n## Artefacts\n\n")
    md.append(artifacts(fk, img))
    for F in ("0.08",):
        fe = os.path.join(OUT, f"proto_box_events_F{F}.csv")
        if os.path.exists(fe):
            pe = pd.read_csv(fe); pi = pd.read_csv(os.path.join(OUT, f"proto_box_images_F{F}.csv"))
            pf = pd.read_csv(os.path.join(OUT, f"proto_box_fakes_F{F}.csv"))
            md.append(f"\n## Prototype counting-box finder (proto_box.py, F = {F} per Mpix nominal; same images)\n")
            md.append(eff_table(pe, order([c for c in pe.config.unique() if not c.startswith("blank")]), label="proto_box"))
            rows = []
            for lab, cs in sorted(groups.items()):
                im = pi[pi.config.isin(cs)]
                if im.empty:
                    continue
                mp = im.mpix.sum(); f = pf[pf.config.isin(cs) & (pf.near_big == 0)]
                rows.append(dict(group=lab, Mpix=f"{mp:.1f}", fakes=len(f), per_Mpix=f"{len(f) / mp:.3f}±{np.sqrt(max(len(f), 1)) / mp:.3f}",
                                 q_occ=f"{im.q_occ.mean():.1e}", lam_fit=f"{im.lam_fit.mean():.1e}±{im.lam_fit.std():.0e}"))
            md.append("\n" + MT.md_table(pd.DataFrame(rows)) + "\n")
    with open(os.path.join(OUT, "tables.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
