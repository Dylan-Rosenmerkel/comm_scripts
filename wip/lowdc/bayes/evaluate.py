"""Tables and ROC figures for the Bayesian finder vs A_GMX and proto_box (wip/lowdc/bayes).

    python evaluate.py  -> results/tables.md, figs/*.png

Fake definition as in ../RESULTS.md: unmatched object (no true electron in its pixels, no event centre within 2 px),
not within 40 px of a big deposit; per Mpix of ALL images of the group (base + blank pooled = 'plain').
Bayesian score: log J1 (Bayes factor of one event with N ~ U{2..20} vs background, per unit event rate) - the ROC
is swept over it; the model's own operating point uses the posterior P(event) with the EM-fitted R_N and the
analytic expected fake count sum_{selected} P(no event | data).
"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOWDC = os.path.dirname(HERE)
sys.path.insert(0, LOWDC)
import metrics as MT  # noqa: E402

RES = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figs")
F_STAR = (0.01, 0.08, 1.0)
COL = {"bayes physics": "#2a78d6", "bayes bench": "#eb6834", "A_GMX": "#1baf7a", "proto_box": "#eda100"}


def _grp_of(config):
    """benchmark config -> group name (blank_x pooled with base_x)"""
    return "base_" + config[len("blank_"):] if config.startswith("blank_") else config


# ------------------------------------------------------------------------------------------------ loaders
def load_bayes(prior):
    d = os.path.join(RES, prior)
    if not os.path.exists(os.path.join(d, "events.csv")):
        return None
    ev = pd.read_csv(os.path.join(d, "events.csv"))
    ob = pd.read_csv(os.path.join(d, "objects.csv"))
    im = pd.read_csv(os.path.join(d, "images.csv"))
    nu = pd.read_csv(os.path.join(d, "null.csv")) if os.path.exists(os.path.join(d, "null.csv")) else pd.DataFrame()
    gj = json.load(open(os.path.join(d, "groups.json")))
    return dict(ev=ev, ob=ob, im=im, nu=nu, gj=gj)


def load_points(kind):
    """operating points of A_GMX / proto_box: list of (F_nominal, events df, fakes df, images df)"""
    out = []
    if kind == "A_GMX":
        ev = pd.read_csv(os.path.join(LOWDC, "results", "events.csv"))
        fk = pd.read_csv(os.path.join(LOWDC, "results", "fakes.csv"))
        fk = fk[fk.src == "agmx"]
        im = pd.read_csv(os.path.join(LOWDC, "results", "images.csv"))
        out.append((0.08, ev, fk, im))
        for F in (0.01, 1.0):
            fn = os.path.join(RES, f"agmx_F{F:g}_events.csv")
            if os.path.exists(fn):
                out.append((F, pd.read_csv(fn), pd.read_csv(os.path.join(RES, f"agmx_F{F:g}_fakes.csv")),
                            pd.read_csv(os.path.join(RES, f"agmx_F{F:g}_images.csv"))))
    else:
        out.append((0.08, pd.read_csv(os.path.join(LOWDC, "results", "proto_box_events_F0.08.csv")),
                    pd.read_csv(os.path.join(LOWDC, "results", "proto_box_fakes_F0.08.csv")),
                    pd.read_csv(os.path.join(LOWDC, "results", "proto_box_images_F0.08.csv"))))
        for F in (0.01, 1.0, 10.0):
            fn = os.path.join(RES, f"proto_F{F:g}_events.csv")
            if os.path.exists(fn):
                out.append((F, pd.read_csv(fn), pd.read_csv(os.path.join(RES, f"proto_F{F:g}_fakes.csv")),
                            pd.read_csv(os.path.join(RES, f"proto_F{F:g}_images.csv"))))
    return out


def point_stats(ev, fk, im, group):
    """efficiency per n_e and fakes / Mpix of one operating point of a baseline on one group"""
    ev = ev.assign(group=ev.config.map(_grp_of)); fk = fk.assign(group=fk.config.map(_grp_of))
    im = im.assign(group=im.config.map(_grp_of))
    e = ev[(ev.group == group) & (ev.kind == "fe")]
    f = fk[(fk.group == group) & (fk.near_big == 0)]
    if "q" in f:
        f = f[f.q < 20]
    imm = im[im.group == group]
    if e.empty or imm.empty:
        return None
    mp = imm.mpix.sum()
    return dict(mpix=mp, n_fake=len(f), fake_rate=len(f) / mp, eff=e.groupby("n_e").found.mean().to_dict(),
                n=e.groupby("n_e").size().to_dict())


# ------------------------------------------------------------------------------------------------ Bayes ROC
def bayes_roc(B, group, ne_list=(2, 3, 4)):
    """thresholds on log J1 -> (fake rate, efficiencies); returns DataFrame"""
    ev = B["ev"][(B["ev"].group == group) & (B["ev"].kind == "fe")]
    ob = B["ob"][B["ob"].group == group]
    mp = B["im"][B["im"].group == group].mpix.sum()
    fakes = np.sort(ob[(ob.owner == -1) & (ob.q_sum < 20)].logJ1.values)[::-1]
    thr = np.unique(np.r_[np.linspace(-2, 40, 211), fakes + 1e-6])
    rows = []
    for t in thr:
        r = dict(thr=t, n_fake=int((fakes >= t).sum()), fake_rate=(fakes >= t).sum() / mp)
        for n in ne_list:
            e = ev[ev.n_e == n]
            r[f"eff{n}"] = float((e.logJ1 >= t).mean()) if len(e) else np.nan
        e = ev[ev.n_e >= 5]
        r["eff5+"] = float((e.logJ1 >= t).mean()) if len(e) else np.nan
        rows.append(r)
    return pd.DataFrame(rows), mp


def eff_at(roc, F, mp, col):
    """best efficiency with measured fakes <= F x Mpix (the threshold just above the highest allowed fake);
    returns (eff, threshold, n_fake)"""
    ok = roc[roc.n_fake <= F * mp]
    if ok.empty:
        return np.nan, np.nan, np.nan
    i = ok[col].idxmax()
    # prefer the highest threshold among equal-efficiency points (conservative)
    best = ok[ok[col] >= ok.loc[i, col] - 1e-12]
    j = best.thr.idxmax()
    return ok.loc[i, col], best.loc[j, "thr"], int(best.loc[j, "n_fake"])


def bayes_operating(B, group, F):
    """the model's own operating point: P(event) >= t with t such that sum_{selected} P(no event) / Mpix = F
    (posterior expected fake rate); returns t, efficiencies, measured fakes"""
    ob = B["ob"][B["ob"].group == group]
    ev = B["ev"][(B["ev"].group == group) & (B["ev"].kind == "fe")]
    mp = B["im"][B["im"].group == group].mpix.sum()
    s = ob.sort_values("p_ev", ascending=False)
    cum = np.cumsum(s.p0.values) / mp
    k = int(np.searchsorted(cum, F, side="right"))
    t = s.p_ev.values[k - 1] if k > 0 else 1.0
    sel = s.iloc[:k]
    nf = int(((sel.owner == -1) & (sel.q_sum < 20)).sum())
    eff = {n: float((ev[ev.n_e == n].p_ev >= t).mean()) for n in (2, 3, 4)}
    return dict(t=t, n_sel=k, pred=float(cum[k - 1] * mp) if k else 0.0, n_fake=nf, mp=mp, **{f"eff{n}": v for n, v in eff.items()})


# ------------------------------------------------------------------------------------------------ main
def main():
    os.makedirs(FIGS, exist_ok=True)
    Bs = {p: load_bayes(p) for p in ("physics", "bench")}
    Bs = {k: v for k, v in Bs.items() if v is not None}
    base = {k: load_points(k) for k in ("A_GMX", "proto_box")}
    md = ["# Bayesian finder: generated tables (evaluate.py)\n"]
    P = Bs["physics"]
    groups = sorted(P["gj"])

    # 1. efficiency at matched measured fake rates
    md.append("## 1. Efficiency at matched (measured) fake rates\n")
    md.append("Measured fakes per Mpix over all images of the group (base + blank). Bayes: threshold on log J1 = the "
              "highest efficiency with <= F x Mpix measured fakes (with < 1 allowed fake this is 'above the highest "
              "fake', optimistic by construction - see the analytic operating point in section 2). Baselines: their "
              "operating points with measured fake rate <= F (A_GMX: count_fakes 0.01 / 0.08 / 1; proto_box: F = "
              "0.01 / 0.08 / 1 / 10); '-' = no point qualifies or not run.\n")
    rows = []
    for g in groups:
        for F in F_STAR:
            r = dict(group=g, F=F)
            for pk, B in Bs.items():
                roc, mp = bayes_roc(B, g)
                r["Mpix"] = round(mp, 1)
                for n in ("2", "3", "4"):
                    e, t, nf = eff_at(roc, F, mp, f"eff{n}")
                    r[f"{pk[:4]} {n}e"] = f"{e:.2f}" if np.isfinite(e) else "-"
                r[f"{pk[:4]} thr"] = f"{eff_at(roc, F, mp, 'eff2')[1]:.1f}"
            for bk, pts in base.items():
                best = None
                for Fn, ev, fk, im in pts:
                    st = point_stats(ev, fk, im, g)
                    if st is None or st["fake_rate"] > F:
                        continue
                    if best is None or st["eff"].get(2, 0) + st["eff"].get(3, 0) > best["eff"].get(2, 0) + best["eff"].get(3, 0):
                        best = dict(st, Fn=Fn)
                for n in (2, 3, 4):
                    r[f"{bk[:5]} {n}e"] = f"{best['eff'].get(n, np.nan):.2f}" if best else "-"
            rows.append(r)
    t1 = pd.DataFrame(rows)
    md.append(MT.md_table(t1) + "\n")

    # 2. the model's own operating point (posterior expected fakes) vs measured / null
    md.append("## 2. Analytic false-alarm control: posterior expected fakes vs measured vs off-image null\n")
    md.append("Selection P(event) >= t with t set so that sum_selected P(no event | data) = F x Mpix. 'meas' = unmatched "
              "selected objects; 'null' = shuffled-image neighbourhoods with P(event) >= t per Mpix of shuffled images "
              "(2 permutations, accepted events removed before shuffling).\n")
    rows = []
    for pk, B in Bs.items():
        for g in groups:
            for F in F_STAR:
                op = bayes_operating(B, g, F)
                nu = B["nu"]
                nmp = B["im"][B["im"].group == g].mpix.sum()
                if len(nu):
                    nug = nu[nu.group == g]
                    reps = max(nug.rep.nunique(), 1) if len(nug) else 2
                    nnull = int((nug.p_ev >= op["t"]).sum())
                else:
                    reps, nnull = 2, 0
                rows.append(dict(prior=pk, group=g, F=F, t=f"{op['t']:.3g}", sel=op["n_sel"], pred=round(op["pred"], 2),
                                 meas=op["n_fake"], null=f"{nnull}/{reps}x", eff2=round(op["eff2"], 2), eff3=round(op["eff3"], 2),
                                 eff4=round(op["eff4"], 2)))
    md.append(MT.md_table(pd.DataFrame(rows)) + "\n")

    # 3. energy response
    md.append("## 3. Electron-count response (events with P(event) > 0.5)\n")
    for pk, B in Bs.items():
        ev = B["ev"]
        e = ev[(ev.kind == "fe") & (ev.p_ev > 0.5) & (ev.n_e >= 2) & ~ev.group.str.startswith(("tail", "art", "busy"))]
        tab = e.groupby("n_e").apply(lambda d: pd.Series(dict(n=len(d), N_mean=round(d.N_mean.mean(), 2),
                                                               exact=round((d.N_map == d.n_e).mean(), 3),
                                                               low1=round((d.N_map <= d.n_e - 1).mean(), 3),
                                                               high1=round((d.N_map >= d.n_e + 1).mean(), 3),
                                                               split=round((d.n_obj50 >= 2).mean(), 3)))).reset_index()
        md.append(f"prior {pk} (base/hotcol/grad/l2 groups):\n\n" + MT.md_table(tab) + "\n")
        for gg in ("tail", "art", "busy"):
            e2 = ev[(ev.kind == "fe") & (ev.p_ev > 0.5) & (ev.n_e >= 2) & ev.group.str.startswith(gg)]
            if len(e2):
                md.append(f"- {pk} {gg}: exact {np.mean(e2.N_map == e2.n_e):.3f}, low {np.mean(e2.N_map < e2.n_e):.3f}, "
                          f"high {np.mean(e2.N_map > e2.n_e):.3f} (n = {len(e2)})\n")
        md.append("\n")

    # 4. sigma / depth
    md.append("## 4. Diffusion sigma and depth recovery (events with P(event) > 0.5, n_e >= 4)\n")
    sys.path.insert(0, HERE)
    import finder as FD
    for pk, B in Bs.items():
        ev = B["ev"]
        e = ev[(ev.kind == "fe") & (ev.p_ev > 0.5) & (ev.n_e >= 4)].copy()
        e["sbin"] = pd.cut(e.sigma, [0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        tab = e.groupby(["sbin"], observed=True).apply(lambda d: pd.Series(dict(n=len(d), sig_post_mean=round(d.sig_mean.mean(), 3),
                                                                                 sig_post_sd=round(d.sig_mean.std(), 3),
                                                                                 z_true=round(np.nanmean(FD.z_of_sigma(d.sigma)), 0),
                                                                                 z_post=round(d.z_mean.mean(), 0)))).reset_index()
        tab["sbin"] = tab.sbin.astype(str)
        rr = np.corrcoef(e.sigma, e.sig_mean)[0, 1]
        md.append(f"prior {pk}: corr(true sigma, posterior mean) = {rr:.3f}\n\n" + MT.md_table(tab) + "\n")

    # 5. background
    md.append("## 5. Dark charge: pooled fit vs truth\n")
    md.append("lam_true = mean simulated rate (dark + serial + hot); +1e = plus the 1-e- event density (indistinguishable); "
              "lam_fit = pooled model mean; lam_init = single-image fallback (median over images, weak prior). A_GMX "
              "detection_bg conservative median from ../results/images.csv.\n")
    agi = pd.read_csv(os.path.join(LOWDC, "results", "images.csv"))
    agi["group"] = agi.config.map(_grp_of)
    rows = []
    for g in groups:
        s = P["gj"][g]
        im = P["im"][P["im"].group == g]
        lt = s["lam_true_mean"]; l1 = s["lam_1e_events"]
        a = agi[agi.group == g]
        rows.append(dict(group=g, lam_true=f"{lt:.2e}", **{"+1e": f"{lt + l1:.2e}"}, lam_fit=f"{s['lam_mean_fit']:.2e}",
                         ratio=round(s["lam_mean_fit"] / (lt + l1), 2), lam0=f"{s['lam0']:.2e}", beta=round(s["beta"], 2),
                         lam_init_med=f"{s['lam_init_median']:.2e}",
                         init_p16_p84=f"{im.lam_init.quantile(0.16):.1e}..{im.lam_init.quantile(0.84):.1e}",
                         agmx_det_cons=f"{a.det_lam_cons.median():.1e}" if len(a) else "-",
                         hot_cols=",".join(str(h["c"]) for h in s["hot_cols_fit"] if h["P"] > 0.5) or "-",
                         R_fit=f"{sum(s['R_N']):.2e}", R_true=f"{s['R_true']:.2e}"))
    md.append(MT.md_table(pd.DataFrame(rows)) + "\n")

    # 6. runtime
    md.append("## 6. Runtime per image (single thread)\n")
    rt = P["im"].groupby("group").agg(n=("img", "size"), mpix=("mpix", "first"), n_cand=("n_cand", "mean"), n_nb=("n_nb", "mean"),
                                      t_noise=("t_noise", "mean"), t_find=("t_find", "mean")).reset_index()
    rt["t_bg_per_img"] = [P["gj"][g]["t_bg"] / P["gj"][g]["n_img"] for g in rt.group]
    rt["t_total"] = (rt.t_noise + rt.t_find + rt.t_bg_per_img).round(3)
    for c in ("n_cand", "n_nb"):
        rt[c] = rt[c].round(0)
    for c in ("t_noise", "t_find", "t_bg_per_img"):
        rt[c] = rt[c].round(3)
    md.append("t_find = candidates + evidences of both passes (pass 2 repeats the evidences with the pooled lam map).\n\n"
              + MT.md_table(rt) + "\n")

    # 7. noise asymmetry (sim)
    md.append("## 7. Noise symmetry check (simulation; counts beyond +-a, |v| < 0.45 e-)\n")
    rows = []
    for g in groups:
        for a in P["gj"][g]["asym"]:
            if a["a"] < 0.45:
                rows.append(dict(group=g, a=round(a["a"], 3), n_pos=a["n_pos"], n_neg=a["n_neg"],
                                 z=round((a["n_pos"] - a["n_neg"]) / np.sqrt(max(a["n_pos"] + a["n_neg"], 1)), 2)))
    asy = pd.DataFrame(rows)
    if len(asy):
        md.append(MT.md_table(asy.groupby("group").apply(lambda d: d.iloc[len(d) // 2]).reset_index(drop=True)) + "\n")

    open(os.path.join(RES, "tables.md"), "w").write("\n".join(md))
    print("\n".join(md))

    # figures: ROC per key group
    key = [g for g in ("base_s0.06_l1e-05", "base_s0.10_l1e-05", "base_s0.15_l1e-05", "base_s0.15_l0.0001",
                       "l2_s0.06_l1e-05", "l2_s0.10_l1e-05") if g in groups]
    for ne in (2, 3):
        fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
        for ax, g in zip(axs.ravel(), key):
            for pk, B in Bs.items():
                roc, mp = bayes_roc(B, g)
                x = np.maximum(roc.fake_rate.values, 0.5 / mp)
                ax.step(x, roc[f"eff{ne}"].values, where="post", color=COL[f"bayes {pk}"], lw=2, label=f"Bayes ({pk} sigma prior)")
            for bk, pts in base.items():
                xs, ys = [], []
                for Fn, ev, fk, im in pts:
                    st = point_stats(ev, fk, im, g)
                    if st is not None:
                        xs.append(max(st["fake_rate"], 0.5 / st["mpix"])); ys.append(st["eff"].get(ne, np.nan))
                if xs:
                    o = np.argsort(xs)
                    ax.plot(np.array(xs)[o], np.array(ys)[o], "o-", color=COL[bk], ms=8, lw=1.5, label=bk)
            ax.axvspan(1e-3, 0.5 / mp, color="0.9", lw=0)
            ax.set_xscale("log"); ax.set_xlim(1e-3, 100); ax.set_ylim(0, 1.02)
            ax.set_title(g, fontsize=10); ax.grid(alpha=0.3, lw=0.5)
            ax.set_xlabel("measured fakes per Mpix"); ax.set_ylabel(f"efficiency, {ne} e- events")
        axs.ravel()[0].legend(fontsize=8, loc="lower right")
        fig.suptitle(f"ROC, {ne}-electron events (grey band: below 0.5 fake in the group's Mpix, plotted at 0.5 fake)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, f"roc_{ne}e.png"), dpi=110)
        plt.close(fig)
    # efficiency vs n_e at the analytic operating point F = 0.08
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
    for ax, g in zip(axs.ravel(), key):
        for pk, B in Bs.items():
            op = bayes_operating(B, g, 0.08)
            ev = B["ev"][(B["ev"].group == g) & (B["ev"].kind == "fe")]
            eff = ev.assign(f=ev.p_ev >= op["t"]).groupby("n_e").f.mean()
            ax.plot(eff.index, eff.values, "o-", color=COL[f"bayes {pk}"], label=f"Bayes {pk} (P>={op['t']:.2g})")
        for bk, pts in base.items():
            for Fn, ev, fk, im in pts:
                if Fn == 0.08:
                    st = point_stats(ev, fk, im, g)
                    if st:
                        ax.plot(list(st["eff"].keys()), list(st["eff"].values()), "s--", color=COL[bk], label=f"{bk} (F 0.08)")
        ax.set_title(g, fontsize=10); ax.set_xlabel("electrons n_e"); ax.set_ylabel("efficiency"); ax.grid(alpha=0.3, lw=0.5)
    axs.ravel()[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "eff_vs_ne.png"), dpi=110); plt.close(fig)
    # energy response scatter
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    e = P["ev"][(P["ev"].kind == "fe") & (P["ev"].p_ev > 0.5) & (P["ev"].n_e >= 2)]
    m = e.groupby("n_e").N_mean.agg(["mean", "std"])
    ax[0].errorbar(m.index, m["mean"], m["std"], fmt="o", color=COL["bayes physics"], label="Bayes posterior-mean N")
    ag = pd.read_csv(os.path.join(LOWDC, "results", "events.csv"))
    ag = ag[(ag.kind == "fe") & (ag.found == 1) & (ag.n_e >= 2) & ag.config.str.startswith("base")]
    m2 = ag.groupby("n_e").q.agg(["mean", "std"])
    ax[0].errorbar(m2.index + 0.15, m2["mean"], m2["std"], fmt="s", color=COL["A_GMX"], label="A_GMX counting charge")
    ax[0].plot([1, 10], [1, 10], color="0.5", lw=1)
    ax[0].set_xlabel("true electrons"); ax[0].set_ylabel("measured electrons (mean +- sd)"); ax[0].legend(fontsize=8)
    e4 = e[e.n_e >= 4]
    ax[1].scatter(e4.sigma, e4.sig_mean, s=6, alpha=0.4, color=COL["bayes physics"], label="physics prior")
    if "bench" in Bs:
        eb = Bs["bench"]["ev"]; eb = eb[(eb.kind == "fe") & (eb.p_ev > 0.5) & (eb.n_e >= 4)]
        ax[1].scatter(eb.sigma, eb.sig_mean, s=6, alpha=0.4, color=COL["bayes bench"], label="bench prior")
    ax[1].plot([0, 1.3], [0, 1.3], color="0.5", lw=1)
    ax[1].set_xlabel("true diffusion sigma [px]"); ax[1].set_ylabel("posterior mean sigma [px]"); ax[1].legend(fontsize=8)
    ax[1].set_title("n_e >= 4")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "energy_sigma.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
