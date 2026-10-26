"""Am-241 lines (runs_29img, EXT1 / EXT3 / EXT4, 29 images) with A_GMX before and after the 2026-10-25 changes
(satellite limit + quality flags): every line fitted alone (Gaussian + linear background) without / with the flag cuts,
and the constrained Np L fit on EXT4.
  python am241_lines_flags.py        (cluster)      python am241_lines_flags.py fit"""
import glob, importlib.util, json, os, sys
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import am241_29img_recluster as RC
import gmx as NEW
OUT = os.environ.get("LINES_OUT", os.path.join(HERE, "clusters_lines_flags"))      # LINES_OUT: another folder
OLD_PATH = os.path.expanduser("~/Documents/comm_scripts_data/session_2026-10-23_clustering/clustering_2026-10-25/excess/gmx_before_flags.py")
LINES = [("Np La", 13.95), ("Np Lb", 17.75), ("Np Lg", 20.78), ("gamma 26.3", 26.34), ("gamma 59.5", 59.54)]


def old_module():
    spec = importlib.util.spec_from_file_location("gmx_old", OLD_PATH); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def job(a):
    stem, ext = a
    fn = os.path.join(OUT, f"{stem}_ext{ext}.npz")
    if os.path.exists(fn): return
    Q, valid = RC.load(stem, ext); bg = NEW.AM.background_map(Q, valid)
    new = [c for c in NEW.general_cluster(Q, valid, bg) if np.isfinite(c["E"])]
    old = [c for c in old_module().general_cluster(Q, valid, bg) if np.isfinite(c["E"])]
    np.savez(fn, **{f"new_{k}": np.array([c.get(k, np.nan) for c in new], float) for k in ("E", "sxy", "blend", "ped", "peakfrac", "chi2")},
             **{f"old_{k}": np.array([c.get(k, np.nan) for c in old], float) for k in ("E", "sxy", "blend")})
    print(stem[-12:], ext, len(old), len(new), flush=True)


def fit():
    import am241_paths; am241_paths.setup()    # runs_29img analysis + the repo's amfit.py
    from amfit import Model, summary
    import am_test as AT
    res = {}
    for ext in (1, 3, 4):
        fs = sorted(glob.glob(os.path.join(OUT, f"*_ext{ext}.npz")))
        if not fs: continue
        D = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in np.load(fs[0]).files}
        S = {"old A_GMX (sxy<0.7, blend<0.02)": (D["old_E"], (D["old_sxy"] < 0.7) & (D["old_blend"] < 0.02)),
             "new A_GMX (sxy<0.7, blend<0.02)": (D["new_E"], (D["new_sxy"] < 0.7) & (D["new_blend"] < 0.02))}
        base = (D["new_sxy"] < 0.7) & (D["new_blend"] < 0.02)
        S["new + ped<10"] = (D["new_E"], base & ~(D["new_ped"] >= 10))
        S["new + peakfrac>0.95"] = (D["new_E"], base & (D["new_peakfrac"] > 0.95))
        S["new + ped<10 + peakfrac>0.95"] = (D["new_E"], base & ~(D["new_ped"] >= 10) & (D["new_peakfrac"] > 0.95))
        print(f"\n=== EXT{ext} ({len(fs)} images): each line fitted alone -> peak/E0 (width eV, counts); 59.5 tail/peak = N(60.5-64)/N(59-60)")
        res[ext] = {}
        for name, (E, m) in S.items():
            e = E[m]; row = {}
            for ln, E0 in LINES:
                r = AT.fit_line(e, E0)
                row[ln] = None if r is None else dict(ratio=r[0] / E0, err=r[1] / E0, width=1000 * r[2])
            tail = np.sum((e > 60.5) & (e < 64)) / max(np.sum((e > 59) & (e < 60)), 1)
            line = f"  {name:34s} n {m.sum():6d}  tail {tail:.2f}  " + "  ".join(
                f"{ln}: " + (f"{row[ln]['ratio']:.4f} ({row[ln]['width']:.0f})" if row[ln] else "-") for ln, _ in LINES)
            if ext == 4:
                rr = Model("Ag").fit(e, 0.985); s = summary(rr)
                Z = np.sqrt(sum((s[g][0] / s[g][1]) ** 2 for g in ("La", "Lb", "Lg", "g26") if s[g][1] > 0))
                row["lepy"] = dict(k=s["k"][0], devndf=rr["devndf"], Z=Z, **{g: s[g][0] / s[g][1] for g in ("Lb", "Lg", "g26", "Ag K") if s[g][1] > 0})
                line += f"  | Lepy k {s['k'][0]:.4f} dev/ndf {rr['devndf']:.2f} Z {Z:.1f}"
            row["n"] = int(m.sum()); row["tail"] = float(tail)
            res[ext][name] = row
            print(line, flush=True)
    json.dump(res, open(os.path.join(HERE, "..", "docs", "aGMX", "lines_flags.json"), "w"), indent=1, default=float)
    # LaTeX table for the A_GMX document
    L = ["\\begin{table}[H]\\centering\\scriptsize", "\\begin{adjustbox}{max width=\\linewidth}", "\\begin{tabular}{llc" + "c" * len(LINES) + "cc}", "\\toprule",
         "CCD & selection & clusters & " + " & ".join(n.replace("gamma", "$\\gamma$") for n, _ in LINES) + " & 59.5 tail & $k$ (EXT4) \\\\", "\\midrule"]
    for ext, rows in res.items():
        for name, row in rows.items():
            cells = [f"{row[n]['ratio']:.4f} ({row[n]['width']:.0f})" if row.get(n) else "--" for n, _ in LINES]
            k = f"{row['lepy']['k']:.4f}" if "lepy" in row else ""
            L.append(f"EXT{ext} & {name.replace('_', '-').replace('<', '$<$').replace('>', '$>$')} & {row['n']} & " + " & ".join(cells) + f" & {row['tail']:.2f} & {k} \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"; L += ["\\end{tabular}", "\\end{adjustbox}", "\\caption*{Every Am-241 line fitted alone: peak / true energy (width in eV); Np L$\\beta$ and L$\\gamma$ are groups of lines, so their centres are not expected at exactly 17.75 / 20.78\\,keV (the constrained fit's $k$ is the check there).}", "\\end{table}"]
    open(os.path.join(HERE, "..", "docs", "aGMX", "lines_flags_table.tex"), "w").write("% generated by clustering_tests/am241_lines_flags.py\n" + "\n".join(L) + "\n")
    contamination_table(AT, Model, summary)


def contamination_table(AT, Model, summary):
    """docs/aGMX/contam_table.tex: the crowded CCD (EXT4) with WADERs and with A_GMX + the pedestal cuts: clusters,
    59.5 keV tail/peak, constrained-fit k, dev/ndf, combined Z and the line significances"""
    import am241_paths
    fs = sorted(glob.glob(os.path.join(OUT, "*_ext4.npz")))
    D = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("new_E", "new_sxy", "new_blend", "new_ped")}
    cwd = os.getcwd(); os.chdir(am241_paths.AN)
    from scan import load as wload
    dW = wload(4); os.chdir(cwd)
    base = (D["new_sxy"] < 0.7) & (D["new_blend"] < 0.02)
    sel = [("WADERs ($\\sigma_{xy}<0.7$)", dW["Energy"][dW["sxy"] < 0.7]),
           ("\\AGMX{} ($\\sigma_{xy}<0.7$, blend $<0.02$)", D["new_E"][base])]
    for c in (10, 5):
        sel.append((f"\\quad + ped $<{c}$\\,\\el/pix", D["new_E"][base & ~(D["new_ped"] >= c)]))
    MAIN = ("La", "Lb", "Lg", "g26")
    rows = []
    for lab, e in sel:
        r = Model("Ag").fit(e, 0.985); st = summary(r)
        z = {g: st[g][0] / st[g][1] for g in ("La", "Lb", "Lg", "g26", "Ag K") if st[g][1] > 0}
        tail = np.sum((e > 60.5) & (e < 64)) / max(np.sum((e > 59) & (e < 60)), 1)
        Z = np.sqrt(sum(z[g] ** 2 for g in MAIN if g in z))
        n = f"{e.size:,}".replace(",", "\\,")
        rows.append(f"{lab} & {n} & {tail:.2f} & {st['k'][0]:.4f} & {r['devndf']:.2f} & {Z:.1f} & "
                    + " / ".join(f"{z.get(g, float('nan')):.1f}" for g in ("Lb", "Lg", "g26", "Ag K")) + " \\\\")
        print("contam:", rows[-1])
    L = ["% generated by clustering_tests/am241_lines_flags.py (contamination_table)", "\\begin{table}[H]\\centering\\small",
         "\\begin{adjustbox}{max width=\\linewidth}", "\\begin{tabular}{lcccccc}", "\\toprule",
         "selection (crowded CCD, 29 images) & clusters & 59.5 tail/peak & $k$ & dev/ndf & $Z$ & L$\\beta$ / L$\\gamma$ / $\\gamma$26.3 / Ag K \\\\",
         "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}"]
    open(os.path.join(HERE, "..", "docs", "aGMX", "contam_table.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        fit(); sys.exit()
    os.makedirs(OUT, exist_ok=True)
    stems = sorted({os.path.basename(f).split("_EXT")[0] for f in glob.glob(os.path.join(RC.G, "*_EXT4_waders.fits.bz2"))})
    with Pool(int(os.environ.get("NPROC", 5))) as p:
        p.map(job, [(s, e) for s in stems for e in (4, 3, 1)], chunksize=1)
