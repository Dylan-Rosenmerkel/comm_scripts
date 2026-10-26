"""Wide test of one method: same images, same injected objects (same seeds) as wide_test.py, one method only.

    python wide_run.py adaptive        -> results_wide_adaptive/      (adaptive.py, engine 'mhc')
    python wide_run.py GMX             -> results_wide_gmx/           (gmx.py)
    python wide_run.py adaptive+GMX    -> results_wide_adaptive_gmx/  (A_GMX: adaptive.py, engine 'gmx')

WIDE_OUT overrides the output folder, DATASET reruns one data set, NPROC sets the processes (default 6).
The method keys and folders are those of the original wide_adaptive.py / wide_gmx.py / wide_adaptive_gmx.py,
so the saved results stay comparable. Compare with summ_wide.py output of wide_test.py.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUTDIR = {"adaptive": "results_wide_adaptive", "GMX": "results_wide_gmx", "adaptive+GMX": "results_wide_adaptive_gmx"}


def make_methods(method):
    import adaptive as AD
    import gmx as GX
    run = {"adaptive": lambda Q, valid, sw, bg: AD.adaptive_cluster(Q, valid, bg, sigma=sw),
           "GMX": lambda Q, valid, sw, bg: GX.general_cluster(Q, valid, bg),
           "adaptive+GMX": lambda Q, valid, sw, bg: AD.adaptive_cluster(Q, valid, bg, sigma=sw, engine="gmx")}[method]
    return lambda Q, valid, sw, bg: {method: lambda: run(Q, valid, sw, bg)}


# at module level, not under __main__: the worker processes (spawn) re-import this file with the same sys.argv,
# and need the same method and output folder
METHOD = sys.argv[1] if len(sys.argv) == 2 and sys.argv[1] in OUTDIR else None
if METHOD:
    os.environ.setdefault("WIDE_OUT", OUTDIR[METHOD])
    import wide_test as W                              # reads WIDE_OUT at import
    W.methods = make_methods(METHOD)

if __name__ == "__main__":
    if METHOD is None:
        sys.exit(__doc__)
    from multiprocessing import Pool
    os.makedirs(W.OUT, exist_ok=True)
    inv = W.inventory()
    if os.environ.get("DATASET"):                      # rerun one data set only
        inv = [a for a in inv if os.environ["DATASET"] in a[0]]
    with Pool(int(os.environ.get("NPROC", 6)), maxtasksperchild=4) as p:
        p.map(W.job, inv, chunksize=1)
