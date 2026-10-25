"""Wide test of adaptive.py: same images, same injected objects (same seeds) as wide_test.py, method 'adaptive+GMX'
only -> results_wide_adaptive/. Compare with summ_wide.py output of wide_test.py."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("WIDE_OUT", "results_wide_adaptive_gmx")
import wide_test as W
import adaptive as AD


def methods(Q, valid, sw, bg):
    return {"adaptive+GMX": lambda: AD.adaptive_cluster(Q, valid, bg, sigma=sw, engine="gmx")}


W.methods = methods

if __name__ == "__main__":
    from multiprocessing import Pool
    os.makedirs(W.OUT, exist_ok=True)
    inv = W.inventory()
    if os.environ.get("DATASET"):                      # rerun one data set only
        inv = [a for a in inv if os.environ["DATASET"] in a[0]]
    with Pool(int(os.environ.get("NPROC", 6)), maxtasksperchild=4) as p:
        p.map(W.job, inv, chunksize=1)
