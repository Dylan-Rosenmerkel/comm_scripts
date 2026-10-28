"""Split clusters at true track crossings (scratch). A merged cluster is split when its clouds contain two straight lines
(RANSAC on cloud centres, each >= min_len px long, >= 3 clouds) at >= ang_min degrees that cross with both lines
continuing >= side px on both sides of the crossing point (an X, not a kink, vertex or delta ray). Clouds go to the
line they lie closest to (perpendicular distance). Applied after merge_graph and again after island_arbitration."""
import numpy as np

def _lines(mu, A, sig, tol=2.0, min_len=20.0, min_n=3, max_lines=4, rng=np.random.default_rng(0)):
    idx = np.arange(len(mu)); L = []
    rem = idx.copy()
    for _ in range(max_lines):
        if rem.size < min_n: break
        best = None
        P = rem if rem.size <= 40 else rng.choice(rem, 40, replace=False)
        for a in range(len(P)):
            for b in range(a + 1, len(P)):
                p, q = mu[P[a]], mu[P[b]]; u = q - p; n = np.linalg.norm(u)
                if n < 4: continue
                u = u / n
                d = np.abs((mu[rem, 0] - p[0]) * u[1] - (mu[rem, 1] - p[1]) * u[0])
                inl = rem[d < tol + 0.5 * sig[rem]]
                if inl.size < min_n: continue
                sc = A[inl].sum()
                if best is None or sc > best[0]: best = (sc, inl)
        if best is None: break
        inl = best[1]
        c = np.average(mu[inl], 0, weights=A[inl]); C = np.cov((mu[inl] - c).T, aweights=A[inl]); u = np.linalg.eigh(C)[1][:, 1]
        t = (mu[inl] - c) @ u
        if t.max() - t.min() < min_len: rem = np.setdiff1d(rem, inl); continue
        L.append((c, u, t.min(), t.max())); rem = np.setdiff1d(rem, inl)
    return L

def _cross(l1, l2, ang_min=20.0, side=8.0):
    (c1, u1, a1, b1), (c2, u2, a2, b2) = l1, l2
    if abs(u1 @ u2) > np.cos(np.radians(ang_min)): return False
    M = np.array([u1, -u2]).T
    try: s, t = np.linalg.solve(M, c2 - c1)
    except np.linalg.LinAlgError: return False
    return (s - a1 >= side) and (b1 - s >= side) and (t - a2 >= side) and (b2 - t >= side)

def split_crossings(mix, lab, min_clouds=4):
    lab = np.asarray(lab).copy(); nxt = lab.max() + 1 if lab.size else 0
    a, c, b = mix.S[:, 0], mix.S[:, 1], mix.S[:, 2]
    sig = np.sqrt(np.maximum(0.5 * (a + c), 1e-6))
    nsplit = 0
    for L in np.unique(lab):
        k = np.nonzero(lab == L)[0]
        if k.size < min_clouds: continue
        lines = _lines(mix.mu[k], mix.A[k], sig[k])
        if len(lines) < 2: continue
        keep = set()
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                if _cross(lines[i], lines[j]): keep |= {i, j}
        if not keep: continue
        keep = sorted(keep)
        d = np.stack([np.abs((mix.mu[k, 0] - lines[i][0][0]) * lines[i][1][1] - (mix.mu[k, 1] - lines[i][0][1]) * lines[i][1][0]) for i in keep], 1)
        own = d.argmin(1)
        for m in range(1, len(keep)):
            lab[k[own == m]] = nxt; nxt += 1
        nsplit += len(keep) - 1
    _, lab = np.unique(lab, return_inverse=True)
    return lab, nsplit

def install(GX):
    if getattr(GX, "_xsplit", False): return
    _mg, _ia = GX.merge_graph, GX.island_arbitration
    def merge_graph(mix, *a, **k):
        return split_crossings(mix, _mg(mix, *a, **k))[0]
    def island_arbitration(lab, mix, *a, **k):
        lab2, dec = _ia(lab, mix, *a, **k)
        return split_crossings(mix, lab2)[0], dec
    GX.merge_graph, GX.island_arbitration, GX._xsplit = merge_graph, island_arbitration, True

def uninstall(GX):
    if getattr(GX, "_xsplit", False):
        GX.merge_graph = GX.merge_graph.__closure__ and _orig_of(GX.merge_graph)
        GX.island_arbitration = _orig_of(GX.island_arbitration); GX._xsplit = False

def _orig_of(f):
    for cell in f.__closure__ or ():
        if callable(cell.cell_contents) and cell.cell_contents.__name__ in ("merge_graph", "island_arbitration"): return cell.cell_contents
    return f
