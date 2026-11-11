"""Before / after comparison of Bayesian-finder benchmark runs (same seeds) -> markdown table (wip/lowdc/bayes).

    python compare.py --tags v2_before,v2_after [--F 0.08,1] [--out results/v2/compare.md]

Per group and nominal fake rate F (analytic operating point: P(event) >= t with sum_selected (1 - P(event)) = F x
Mpix, evaluate.bayes_operating): efficiency for 2 / 3 / 4 e-, predicted / measured / off-image-null fakes, pooled lam
vs truth (+ 1-e- events), R, runtime per image. Also log J1 >= 12 (rate-free score) efficiencies and fakes.
"""
import argparse, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate as EV  # noqa: E402
sys.path.insert(0, os.path.dirname(HERE))
import metrics as MT  # noqa: E402


def rows_for(tag, F_list, groups=None):
    B = EV.load_bayes(tag)
    if B is None:
        return []
    out = []
    for g in sorted(B["gj"]):
        if groups and g not in groups:
            continue
        s = B["gj"][g]
        im = B["im"][B["im"].group == g]
        lt = s["lam_true_mean"] + s["lam_1e_events"]
        tpi = (im.t_noise + im.t_find).mean() + s["t_bg"] / s["n_img"]
        ev = B["ev"][(B["ev"].group == g) & (B["ev"].kind == "fe")]
        ob = B["ob"][B["ob"].group == g]
        mp = im.mpix.sum()
        f12 = int(((ob.owner == -1) & (ob.q_sum < 20) & (ob.logJ1 >= 12)).sum())
        e12 = "/".join(f"{(ev[ev.n_e == n].logJ1 >= 12).mean():.2f}" for n in (2, 3, 4))
        for F in F_list:
            op = EV.bayes_operating(B, g, F)
            nu = B["nu"]
            nug = nu[nu.group == g] if len(nu) else nu
            reps = max(nug.rep.nunique(), 1) if len(nug) else 0
            nnull = (nug.p_ev >= op["t"]).sum() / reps if reps else np.nan
            out.append(dict(tag=tag, group=g, F=F, Mpix=round(mp, 1),
                            eff=f"{op['eff2']:.2f}/{op['eff3']:.2f}/{op['eff4']:.2f}",
                            pred=round(op["pred"], 1), meas=op["n_fake"], null=round(nnull, 1) if reps else "-",
                            lam_ratio=round(s["lam_mean_fit"] / lt, 2), R_ratio=round(sum(s["R_N"]) / max(s["R_true"], 1e-30), 2),
                            R_pt=f"{sum(s['R_pt']):.1e}" if s.get("R_pt") else "-",
                            eff_lJ12=e12, fakes_lJ12=f12, t_img=round(tpi, 2)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", default="v2_before,v2_after")
    ap.add_argument("--F", default="0.08,1")
    ap.add_argument("--groups", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    F_list = [float(x) for x in a.F.split(",")]
    gs = set(filter(None, a.groups.split(",")))
    rows = []
    for t in a.tags.split(","):
        rows += rows_for(t, F_list, gs)
    df = pd.DataFrame(rows).sort_values(["group", "F", "tag"], kind="stable")
    md = MT.md_table(df)
    print(md)
    if a.out:
        os.makedirs(os.path.dirname(os.path.join(HERE, a.out)), exist_ok=True)
        open(os.path.join(HERE, a.out), "w").write(md + "\n")


if __name__ == "__main__":
    main()
