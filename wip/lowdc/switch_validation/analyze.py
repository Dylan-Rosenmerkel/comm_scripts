"""Tables of the switch validation -> results/tables.md (sections: benchmark, decisions, halo, boundary, L2, DM09).

    python analyze.py [--versions v0,v1]
"""
import os
import argparse
import json

import numpy as np
import pandas as pd

import common as C
import metrics as MT
from bench_switch import load

EFF_BINS = ((2, 2), (3, 3), (4, 4), (5, 99))
NOISY = ("_s0.20", "_s0.25", "_l0.01", "_s0.17")


def grp_of(config):
    return "base_" + config[len("blank_"):] if config.startswith("blank_") else config


def eff_str(e):
    out = []
    for lo, hi in EFF_BINS:
        x = e[(e.n_e >= lo) & (e.n_e <= hi)]
        out.append(f"{x.found.mean():.2f}" if len(x) else "-")
    return "/".join(out)


def bayes_standalone(prior="physics", F=0.08):
    """the stand-alone Bayes records at the group operating point (bayes/evaluate.bayes_operating, + >= 5 e-)"""
    d = os.path.join(C.BAYES, "results", prior)
    if not os.path.exists(os.path.join(d, "events.csv")):
        return {}
    ev = pd.read_csv(os.path.join(d, "events.csv")); ob = pd.read_csv(os.path.join(d, "objects.csv"))
    im = pd.read_csv(os.path.join(d, "images.csv"))
    out = {}
    for g in ev.group.unique():
        o = ob[ob.group == g].sort_values("p_ev", ascending=False)
        mp = im[im.group == g].mpix.sum()
        cum = np.cumsum(o.p0.values) / mp
        k = int(np.searchsorted(cum, F, side="right"))
        t = o.p_ev.values[k - 1] if k > 0 else 1.0
        sel = o.iloc[:k]
        nf = int(((sel.owner == -1) & (sel.q_sum < 20)).sum())
        e = ev[(ev.group == g) & (ev.kind == "fe")].assign(found=lambda x: (x.p_ev >= t).astype(int))
        ok = e[e.found == 1]
        exact = float((np.abs(ok.N_mean - ok.n_e) < 0.5).mean()) if len(ok) else np.nan
        out[g] = dict(eff=eff_str(e[e.n_e >= 2]), fakes=f"{nf} ({nf / mp:.2f})", exact=exact, mpix=mp)
    return out


def runs_of(v, main):
    """(version, tag) columns: A_GMX alone and every mode of the main version, single-image only for the others"""
    return [("agmx", "A_GMX alone")] if v == main else []


def tags_of(v, main):
    return ["agmx", "single", "pooled"] if v == main else ["single", "pooled", "pooled_it"]


def label(v, tag):
    return "A_GMX alone" if tag == "agmx" else f"{v} {tag}"


def bench_tables(versions):
    L = []
    R = {v: load(v) for v in versions}
    main = versions[-1]
    cur = R[main]
    ev, fk, im = cur["events"], cur["fakes"], cur["images"]
    for v in versions:
        R[v]["events"]["group"] = R[v]["events"].config.map(grp_of)
        R[v]["fakes"]["group"] = R[v]["fakes"].config.map(grp_of)
        R[v]["images"]["group"] = R[v]["images"].config.map(grp_of)
    BS = bayes_standalone()
    groups = sorted(im.group.unique(), key=lambda g: (any(t in g for t in NOISY), g))
    L.append("## 1. Benchmark: efficiency 2 / 3 / 4 / >=5 e- and fakes (count, per Mpix) per group\n")
    L.append("Fakes: unmatched objects < 20 e-, not within 40 px of a big deposit, over all images of the group "
             "(base + blanks). Bayes alone: bayes/results/physics at its group operating point (0.08 / Mpix).\n")
    cols = ["group", "Mpix"] + [label(v, t) for v in versions for t in tags_of(v, main)] + ["Bayes alone"]
    rows = []
    for g in groups:
        r = {"group": g}
        mp = cur["images"][(cur["images"].group == g) & (cur["images"].tag == "agmx")].mpix.sum()
        r["Mpix"] = f"{mp:.1f}"
        for v in versions:
            E, F = R[v]["events"], R[v]["fakes"]
            for tag in tags_of(v, main):
                e = E[(E.group == g) & (E.tag == tag) & (E.kind == "fe") & (E.n_e >= 2)]
                f = F[(F.group == g) & (F.tag == tag) & (F.near_big == 0) & (F.q < 20)]
                r[label(v, tag)] = f"{eff_str(e)} ({len(f)}, {len(f) / mp:.2f})" if len(e) else "-"
        b = BS.get(g)
        r["Bayes alone"] = f"{b['eff']} ({b['fakes']})" if b else "-"
        rows.append(r)
    L.append(MT.md_table(pd.DataFrame(rows)[cols]))
    # blanks only
    L.append("\n\n### 1b. Fakes on the blank images only (count, per Mpix)\n")
    rows = []
    for g in groups:
        bl = im[(im.group == g) & im.config.str.startswith("blank_") & (im.tag == "agmx")]
        if bl.empty:
            continue
        mp = bl.mpix.sum()
        r = dict(group=bl.config.iloc[0], Mpix=f"{mp:.1f}")
        for v in versions:
            F = R[v]["fakes"]
            for tag in tags_of(v, main):
                f = F[(F.config == bl.config.iloc[0]) & (F.tag == tag) & (F.near_big == 0) & (F.q < 20)]
                r[label(v, tag)] = f"{len(f)} ({len(f) / mp:.2f})"
        rows.append(r)
    L.append(MT.md_table(pd.DataFrame(rows)))
    # energies
    L.append("\n\n### 1c. Electron counts of found events (single object): exact / low by >= 1, n_e 2..10, sparse groups\n")
    rows = []
    sparse = [g for g in groups if not any(t in g for t in NOISY)]
    for v in versions:
        E = R[v]["events"]
        for tag in tags_of(v, main):
            e = E[E.group.isin(sparse) & (E.tag == tag) & (E.kind == "fe") & (E.found == 1) & (E.n_obj == 1) & (E.n_e >= 2)]
            r = dict(run=label(v, tag))
            for n in range(2, 11):
                x = e[e.n_e == n]
                r[str(n)] = f"{(np.abs(x.q - n) < 0.5).mean():.2f}/{(x.q < n - 0.5).mean():.2f}" if len(x) else "-"
            ee = E[E.group.isin(sparse) & (E.tag == tag) & (E.kind == "fe") & (E.found == 1) & (E.n_e >= 2)]
            r["split"] = f"{(ee.n_obj > 1).mean():.3f}"
            rows.append(r)
    L.append(MT.md_table(pd.DataFrame(rows)))
    # 1-e- events
    L.append("\n\n### 1d. 1-e- events found (sparse groups; the Bayesian model has no N = 1 events)\n")
    rows = []
    for v in versions:
        E = R[v]["events"]
        for tag in tags_of(v, main):
            e = E[E.group.isin(sparse) & (E.tag == tag) & (E.kind == "fe") & (E.n_e == 1)]
            rows.append(dict(run=label(v, tag), eff_1e=f"{e.found.mean():.2f}", n=len(e)))
    L.append(MT.md_table(pd.DataFrame(rows)))
    # big deposits (busy)
    L.append("\n\n### 1e. Big deposits (busy group): found, mean q / true\n")
    rows = []
    for v in versions:
        E = R[v]["events"]
        for tag in tags_of(v, main):
            e = E[(E.tag == tag) & (E.kind != "fe")]
            rows.append(dict(run=label(v, tag), n=len(e), found=f"{e.found.mean():.3f}",
                             q_ratio=f"{(e.q / e.n_e).mean():.4f}"))
    L.append(MT.md_table(pd.DataFrame(rows)))
    # runtime
    L.append("\n\n### 1f. Runtime per image [s] (median; single thread, 2 workers)\n")
    rows = []
    for g in groups:
        a = im[(im.group == g) & (im.tag == "agmx")]
        s = im[(im.group == g) & (im.tag == "single")]
        p = im[(im.group == g) & (im.tag == "pooled")]
        rows.append(dict(group=g, A_GMX=f"{a.t_switch.median():.2f}", switch_single=f"{s.t_switch.median():.2f}",
                         switch_pooled=f"{p.t_switch.median():.2f}", prepare_run_per_image=f"{p.t_prep.median():.2f}"))
    L.append(MT.md_table(pd.DataFrame(rows)))
    return L, R


def on_table(R, main):
    """forced Bayes ('on') vs A_GMX alone and the rule ('single') where the rule says A_GMX"""
    E, F, I = R[main]["events"], R[main]["fakes"], R[main]["images"]
    on = I[I.tag == "on"]
    if on.empty:
        return []
    L = ["\n\n### 1g. Above the upper cut: forced Bayes (fewe='on') vs the rule (single) vs A_GMX alone\n"]
    rows = []
    for g in sorted(on.group.unique()):
        mp = I[(I.group == g) & (I.tag == "agmx")].mpix.sum()
        r = dict(group=g, f_c=f"{on[on.group == g].f_c.median():.2e}")
        for tag in ("agmx", "single", "on"):
            e = E[(E.group == g) & (E.tag == tag) & (E.kind == "fe") & (E.n_e >= 2)]
            f = F[(F.group == g) & (F.tag == tag) & (F.near_big == 0) & (F.q < 20)]
            r[tag] = f"{eff_str(e)} ({len(f)}, {len(f) / mp:.2f})"
        r["t_on"] = f"{on[on.group == g].t_switch.median():.1f}"
        rows.append(r)
    L.append(MT.md_table(pd.DataFrame(rows)))
    return L


def decision_tables(R, versions):
    L = ["\n\n## 2. Regime decisions per group (main version, single-image mode; pooled in brackets)\n"]
    im = R[versions[-1]]["images"]
    rows = []
    for g in sorted(im.group.unique()):
        s = im[(im.group == g) & (im.tag == "single")]
        p = im[(im.group == g) & (im.tag == "pooled")]
        n = len(s)
        blocks = s.nb_bayes.sum() + s.nb_both.sum() + s.nb_agmx.sum()
        r = dict(group=g, n_img=n, s=f"{s.s.median():.3f}", f_c=f"{s.f_c.median():.2e}",
                 used=f"{s.used.sum()}/{n}", blocks_bayes_both_agmx=f"{s.nb_bayes.sum() / blocks:.2f}/{s.nb_both.sum() / blocks:.2f}/{s.nb_agmx.sum() / blocks:.2f}",
                 lam=f"{s.lam.median():.1e} ({p.lam.median():.1e})", only3=f"{s.only3.sum()} ({p.only3.sum()})",
                 identical_when_unused=f"{int(((s.used == 0) & (s.identical == 1)).sum())}/{int((s.used == 0).sum())}")
        for v in versions:
            if v != versions[-1]:
                sv = R[v]["images"][(R[v]["images"].group == g) & (R[v]["images"].tag == "single")]
                r[f"{v} used"] = f"{int(sv.used.sum())}/{len(sv)}"
        rows.append(r)
    L.append(MT.md_table(pd.DataFrame(rows)))
    return L


def halo_tables():
    fn = os.path.join(C.OUT, "halo_events.csv")
    if not os.path.exists(fn):
        return []
    ev = pd.read_csv(fn); fk = pd.read_csv(os.path.join(C.OUT, "halo_fakes.csv")); bg = pd.read_csv(os.path.join(C.OUT, "halo_big.csv"))
    L = ["\n\n## 3. Halo hand-over (small events 2-10 e- near Fe-55 / muons)\n"]
    e = ev[ev.kind == "fe"].copy()
    bins = [0, 4, 7, 10, 13, 16, 20, 25, 32, 45]
    e["dbin"] = pd.cut(e.d_pix, bins)
    for cfg in sorted(e.config.unique()):
        L.append(f"\n### {cfg}: efficiency (2-10 e-) vs distance to the nearest big-deposit pixel; finder of the found\n")
        x = e[e.config == cfg]
        t = x.pivot_table(index="dbin", columns="tag", values="found", aggfunc="mean", observed=True).round(2)
        t.insert(0, "n", x[x.tag == "agmx"].groupby("dbin", observed=True).size())
        L.append(MT.md_table(t.reset_index().astype(str)))
        x2 = x[(x.found == 1) & (x.n_obj == 1)]
        t = x2.assign(exact=(np.abs(x2.q - x2.n_e) < 0.5)).pivot_table(index="tag", values="exact", aggfunc="mean").round(3)
        L.append("\n\nexact electron count of found events, all distances:\n\n" + MT.md_table(t.reset_index().astype(str)))
    f = fk[(fk.q < 20)].copy()
    f["dbin"] = pd.cut(f.d_big, [0, 5, 10, 15, 20, 30, 40, 1e9])
    L.append("\n\n### Unmatched objects < 20 e- vs distance to the nearest big-deposit pixel (all images)\n")
    t = f.pivot_table(index="dbin", columns="tag", values="q", aggfunc="count", observed=True).fillna(0).astype(int)
    L.append(MT.md_table(t.reset_index().astype(str)))
    L.append("\n\n### A_GMX clusters > 20 e- unchanged (fraction of images), agmx blocks per image, f_c vs f_c of all pixels\n")
    t = bg.groupby("tag").agg(big_same=("big_same", "mean"), nb_agmx=("nb_agmx", "mean"), f_c=("f_c", "median"),
                              f_c_all=("f_c_all", "median")).reset_index()
    L.append(MT.md_table(t.astype(str)))
    big = ev[ev.kind != "fe"]
    t = big.groupby("tag").agg(found=("found", "mean"), q_ratio=("q", lambda q: (q / big.loc[q.index, "n_e"]).mean())).reset_index()
    L.append("\n\nbig deposits found and q / true:\n\n" + MT.md_table(t.astype(str)))
    return L


def boundary_tables():
    fn = os.path.join(C.OUT, "boundary_events.csv")
    if not os.path.exists(fn):
        return []
    ev = pd.read_csv(fn)
    L = ["\n\n## 4. Block boundaries (events within 1.5 px of a 100-px block edge)\n"]
    rows = []
    ref = ev[ev.tag == "agmx"].set_index(["config", "img", "ev"]).found
    for (cfg, tag), x in ev[ev.n_e >= 2].groupby(["config", "tag"]):
        xi = x.set_index(["config", "img", "ev"])
        mixed = x.finder.fillna("").str.contains(r"\+")
        lost = ((ref.reindex(xi.index).values == 1) & (xi.found.values == 0)).sum()
        rows.append(dict(config=cfg, run=tag, n=len(x), eff=f"{x.found.mean():.3f}", eff_low_noise=f"{x[x.s_here < 0.15].found.mean():.3f}",
                         eff_high_noise=f"{x[x.s_here >= 0.15].found.mean():.3f}" if (x.s_here >= 0.15).any() else "-",
                         split=int((x.n_obj > 1).sum()), double_agmx_bayes=int(mixed.sum()), lost_vs_agmx=int(lost)))
    L.append(MT.md_table(pd.DataFrame(rows)))
    fk = pd.read_csv(os.path.join(C.OUT, "boundary_fakes.csv"))
    t = fk[fk.q < 20].groupby(["config", "tag"]).size().rename("fakes").reset_index()
    L.append("\n\nunmatched objects < 20 e-:\n\n" + MT.md_table(t.astype(str)))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", default="v0,v1", help="comma-separated; the last one is the main version")
    a = ap.parse_args()
    vers = a.versions.split(",")
    L, R = bench_tables(vers)
    L += on_table(R, vers[-1])
    L += decision_tables(R, vers)
    L += halo_tables()
    L += boundary_tables()
    open(os.path.join(C.OUT, "tables.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
