import sys, time, numpy as np
sys.path.insert(0, "."); import harness as HX, bgv3, muons
if __name__ == "__main__":
    Q, valid = HX.load(9, 1); bg, _ = bgv3.regional_v3(Q, valid, HX.AM.background_map(Q, valid))
    t = time.time(); tk, mu = muons.find_muons(Q, valid, bg); print(f"muon finder: {len(mu)} tracks in {time.time() - t:.0f} s; lengths median {np.median([m['length'] for m in mu]):.0f}, max {max(m['length'] for m in mu):.0f}; pixels {tk.sum()}")
    np.save("mu_taken_9_1.npy", tk)
    lab = np.zeros(Q.shape, np.int32)
    for k, m in enumerate(mu, 1): lab[m["_pix"][0], m["_pix"][1]] = k
    np.save("mu_lab_9_1.npy", lab)
