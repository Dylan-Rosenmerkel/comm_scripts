"""Straight-track finder v2 on lower-background images: package ("new", from regress.py) vs package + muons2 ("newmu").
Subset: 2 CCDs (first module image) per gn_lowcharge_study run + 4 Am-241 EXT4 images. -> reg/<tag>_newmu.npz
(and _new.npz if regress.py has not made it)."""
import os, sys, time, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as HX, regress as RG
AM, AD = HX.AM, HX.AD
OUT = RG.OUT

def subset():
    J = [j for j in RG.jobs() if not j[4].startswith("L2_") and "High_Temp" not in j[4]]
    out, seen = [], {}
    for j in J:
        if j[0] == "am":
            if sum(1 for o in out if o[0] == "am") < 4 and int(j[2][-2:].strip("_")) % 7 == 0: out.append(j)
            continue
        g = j[4][:25]
        if seen.get(g, 0) < 2 and j[3] in (1, 4): out.append(j); seen[g] = seen.get(g, 0) + 1
    return out

def run_job(j):
    import bgv3, xsplit, streaks, muons2
    kind, run, stem, ext, tag = j
    if all(os.path.exists(f"{OUT}/{tag}_{v}.npz") for v in ("newx", "newmu")): return
    t0 = time.time(); fr = 0.0; nmu = 0
    try:
        if kind == "am":
            sys.path.insert(0, os.path.expanduser("~/Documents/Software/comm_scripts/clustering_tests"))
            import am241_29img_recluster as RC
            Q, valid = RC.load(stem, ext); gc = True
        else:
            Q, valid, gc = RG.load_run(run, stem, ext)
        bg = AM.background_map(Q, valid); b3, rep = bgv3.regional_v3(Q, valid, bg)
        for v in ("newx", "newmu"):
            fn = f"{OUT}/{tag}_{v}.npz"
            if os.path.exists(fn): continue
            t1 = time.time(); nmu = 0
            xsplit.install(HX.GX); streaks.install(AD)
            if v == "newmu":
                tk, mu, mq = muons2.find_muons(Q, valid, b3); nmu = len(mu); fr = float(tk[valid].mean())
                muons2.install(AD)
            cl, info = AD.adaptive_cluster(Q, valid, b3, engine="gmx", return_info=True, return_pixels=True)
            muons2.uninstall(AD); streaks.uninstall(AD); xsplit.uninstall(HX.GX)
            cl = [c for c in cl if np.isfinite(c.get("E", np.nan)) and "_pix" in c]
            cl, nl = HX.link_tracks(cl)
            tab = {k: np.array([c.get(k, np.nan) for c in cl], float) for k in ("E", "sxy", "x", "y", "blend", "ped")}
            np.savez_compressed(fn, cls=np.array([c.get("cls", "") for c in cl]), regime=np.array([c.get("regime", "") for c in cl]), gaincorr=gc,
                                sec=time.time() - t1, rep=json.dumps(rep), n_linked=nl, npix=np.array([c["_pix"][0].size for c in cl]),
                                n_mu=nmu, tube_frac=fr if v == "newmu" else 0.0, **tab)
        print(f"{tag}: {nmu} finder tracks, tube fraction {fr:.4f}, {time.time() - t0:.0f} s", flush=True)
    except Exception as e:
        import traceback; print(f"{tag}: FAILED {e!r} {traceback.format_exc()[-400:]}", flush=True)

if __name__ == "__main__":
    J = subset(); print(len(J), "jobs", flush=True)
    from multiprocessing import get_context
    with get_context("spawn").Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4, maxtasksperchild=2) as p:
        list(p.imap_unordered(run_job, J, chunksize=1))
    print("ALLDONE", flush=True)
