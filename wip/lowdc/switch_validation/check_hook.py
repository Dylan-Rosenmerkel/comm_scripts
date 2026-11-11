"""Checks of the fewe hook in clustering_tests/adaptive.py:

  1. fewe='off' is bit-identical to adaptive.py before the hook (scratchpad backup adaptive.py.before_fewe), with and
     without return_pixels, on sparse / busy / crowded-mode / unmodelled images
  2. adaptive_cluster(fewe='auto') == agmx_switch.apply(A_GMX's fewe='off' output) (what the benchmark evaluates),
     single-image and with a pooled lam / R_N
  3. return_pixels=False with fewe='auto': same clusters without '_pix'

    python check_hook.py  -> prints a table; results/check_hook.csv
"""
import os
import copy
import importlib.util
import warnings

import numpy as np
import pandas as pd

import common as C

BACKUP = os.path.join(os.path.dirname(C.SCRATCH), "adaptive.py.before_fewe")


def load_backup():
    import importlib.machinery
    spec = importlib.util.spec_from_file_location("adaptive_before_fewe", BACKUP,
                                                  loader=importlib.machinery.SourceFileLoader("adaptive_before_fewe", BACKUP))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def images():
    import sim
    cf = C.configs()
    for name in ("base_s0.15_l1e-05", "busy_s0.15_l0.0001", "base_s0.06_l0.0001", "art_s0.10_l1e-05", "base_s0.25_l1e-05"):
        Q, valid, truth, p, rate = C.simulate(name, 0, cf[name][2])
        yield name, Q, valid
    # crowded mode (dark charge 0.5 e-/pix, Fe-55) and unmodelled (read noise 1.5 e-)
    for name, par in (("crowded_l0.5", dict(lam=0.5, s=0.15, n_fe55=6, n_muon=2)), ("unmodelled_s1.5", dict(lam=1e-3, s=1.5))):
        rng = np.random.default_rng(C.B._seed(name, 0))
        Q, valid, truth, p, rate = sim.simulate(rng, **par)
        yield name, np.where(valid, Q, 0.0), valid


def main():
    warnings.filterwarnings("ignore")
    import adaptive as AD
    import agmx_switch as SW
    OLD = load_backup()
    E1 = AD.AM.E_PER_E
    rows = []
    for name, Q, valid in images():
        bg = AD.AM.background_map(Q, valid)
        r = dict(image=name)
        for rp in (True, False):
            a, ia = OLD.adaptive_cluster(Q, valid, bg, engine="gmx", return_pixels=rp, return_info=True)
            b, ib = AD.adaptive_cluster(Q, valid, bg, engine="gmx", return_pixels=rp, return_info=True, fewe="off")
            r[f"off_identical_pix{int(rp)}"] = int(C.same_clusters(a, b) and set(ia) == set(ib))
        r.update(regime=ia.get("regime"), crowded=int(bool(ia.get("crowded_mode", False))))
        off, info = AD.adaptive_cluster(Q, valid, bg, engine="gmx", return_pixels=True, return_info=True, fewe="off")
        on, ion = AD.adaptive_cluster(Q, valid, bg, engine="gmx", return_pixels=True, return_info=True, fewe="auto")
        ref, iref = SW.apply(Q, valid, copy.deepcopy(off), {k: v for k, v in info.items()}, E1, "auto")
        r["auto_eq_apply"] = int(C.same_clusters(on, ref))
        on2 = AD.adaptive_cluster(Q, valid, bg, engine="gmx", return_pixels=False, fewe="auto")
        r["nopix_eq"] = int(C.same_clusters(on2, [{k: v for k, v in c.items() if k != "_pix"} for c in on]))
        fw = ion.get("fewe", {})
        r.update(used=fw.get("used"), reason=fw.get("reason", ""), f_c=fw.get("f_c", np.nan), n_bayes=fw.get("n_bayes", 0),
                 n_off=len(off), n_auto=len(on))
        if fw.get("used"):
            lam_maps, R_N = SW.prepare_run([(Q, valid)])
            onp, _ = AD.adaptive_cluster(Q, valid, bg, engine="gmx", return_pixels=True, return_info=True, fewe="auto",
                                         fewe_lam=lam_maps[0], fewe_RN=R_N)
            refp, _ = SW.apply(Q, valid, copy.deepcopy(off), dict(info), E1, "auto", lam_maps[0], R_N)
            r["pooled_eq_apply"] = int(C.same_clusters(onp, refp))
        print(r, flush=True)
        rows.append(r)
    os.makedirs(C.OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "check_hook.csv"), index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
