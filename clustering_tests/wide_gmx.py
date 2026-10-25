"""Wide test of gmx.py (GMX): same images, same injected objects (same seeds) as wide_test.py, method 'GMX'
only -> results_wide_gmx/. Compare with summ_wide.py output of wide_test.py."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("WIDE_OUT", "results_wide_gmx")
import wide_test as W
import gmx as GX


def methods(Q, valid, sw, bg):
    return {"GMX": lambda: GX.general_cluster(Q, valid, bg)}


W.methods = methods

if __name__ == "__main__":
    from multiprocessing import Pool
    os.makedirs(W.OUT, exist_ok=True)
    inv = W.inventory()
    if os.environ.get("DATASET"):                      # rerun one data set only
        inv = [a for a in inv if os.environ["DATASET"] in a[0]]
    with Pool(int(os.environ.get("NPROC", 6)), maxtasksperchild=4) as p:
        p.map(W.job, inv, chunksize=1)
