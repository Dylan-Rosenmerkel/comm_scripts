"""Skipper-CCD image simulator for the ultra-low dark-current few-electron benchmark (wip/lowdc).

An image in electrons (what A_GMX clusters):

    Q[r, c] = n_dark + n_serial + n_hot + n_events + read noise + row pedestal residual

  dark charge     Poisson(lam) per active pixel; lam uniform or a linear gradient along the rows (lam at the first
                  row -> lam x (1 + grad) at the last: the exposure of a row grows with its readout time)
  serial register Poisson(lam_ser) in EVERY pixel incl. the overscan columns (charge picked up while the row waits in
                  the serial register; same rate along a row, optionally a per-row rate scatter)
  hot columns     extra Poisson rate (1e-3 .. 1e-1 e-/pix) in a few columns; hot pixels: a few pixels with 0.1..3 e-
  read noise      Gaussian sigma s; optional non-Gaussian tail: 'mix' = a fraction f_tail of pixels with k_tail x s
                  (bad samples, telegraph noise), 't' = Student-t with nu dof scaled to the same core width
  outliers        a fraction f_out of pixels shifted by +-U(out_lo, out_hi) e- (both signs; real L2 EXT4 at low skip
                  counts shows strongly negative and positive excursions)
  row pedestal    per-row offset residual N(0, ped_row) (baseline subtraction leaves a small row-wise error)
  first column    column 0 carries an extra level first_col [e-] (+ N(0, first_col_jit) per row): charge seen only in
                  the first skip(s), averaged over NDCM skips (L2 ACM 103: ~2000 / NDCM ADU, ~1.3 ADU/e-)
  events          n_e = 1..10 electrons, each electron displaced from the event centre by N(0, sigma) in y and x
                  (lateral diffusion; sigma -> 0 = point-like), binned into pixels; Fe-55-like (1600 e-,
                  sigma 0.5-1.5) and straight muon tracks (~100 e-/um, sigma growing with depth) as large deposits

simulate(...) returns (Q, valid, truth); truth is a list of dicts (kind, y, x, n_e, sigma, pix = (rows, cols, count)).
All randomness comes from the numpy Generator passed in, so a fixed seed reproduces the image.
"""
import numpy as np

PIX_UM = 15.0           # pixel pitch [um]
THICK_UM = 675.0        # DAMIC-M CCD thickness [um]
MIP_E_PER_UM = 100.0    # most-probable-ish charge of a minimum-ionising track in Si: ~3.9 MeV/cm / 3.75 eV ~ 100 e-/um


def default_params(**kw):
    """the benchmark's default image: 480 x 1280 active pixels + 64 overscan columns, lam 1e-4, s 0.15, 30 events"""
    p = dict(H=480, W=1280, n_os=64, lam=1e-4, lam_grad=0.0, lam_ser=0.0, ser_row_scatter=0.0,
             hot_cols=(), hot_col_rate=0.0, n_hot_pix=0, hot_pix_rate=(0.1, 3.0),
             s=0.15, tail="none", f_tail=0.0, k_tail=2.5, nu=5.0, ped_row=0.0,
             f_out=0.0, out_lo=1.0, out_hi=10.0, first_col=0.0, first_col_jit=0.0,
             n_events=30, ne_range=(1, 10), sig_range=(0.2, 1.2), sep=20, margin=8,
             n_fe55=0, n_muon=0)
    p.update(kw)
    return p


def _place(rng, n, H, W, margin, sep, taken):
    """n random centres (continuous) at least `sep` px from each other and from the centres in `taken`"""
    out = []
    tries = 0
    while len(out) < n and tries < 200 * max(n, 1):
        tries += 1
        y, x = rng.uniform(margin, H - margin), rng.uniform(margin, W - margin)
        if all((y - a) ** 2 + (x - b) ** 2 >= sep * sep for a, b, _ in taken + out):
            out.append((y, x, sep))
    return out


def deposit(rng, y, x, n_e, sigma, H, W):
    """electron positions of a point deposit diffused by sigma [px] -> pixel counts (rows, cols, counts) inside the image.
    Pixel (r, c) covers [r, r + 1) x [c, c + 1), so a point at (r + 0.5, c + 0.5) with sigma -> 0 lands in one pixel."""
    yy = np.floor(y + rng.normal(0.0, sigma, n_e)).astype(int) if sigma > 0 else np.full(n_e, int(np.floor(y)))
    xx = np.floor(x + rng.normal(0.0, sigma, n_e)).astype(int) if sigma > 0 else np.full(n_e, int(np.floor(x)))
    ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    key, cnt = np.unique(yy[ok] * W + xx[ok], return_counts=True)
    return key // W, key % W, cnt


def muon(rng, y, x, H, W, L=None):
    """straight track through the full thickness: projected length L [px] (5..120), random direction; charge
    MIP_E_PER_UM x 3-D path length, spread uniformly along the projection; diffusion sigma grows 0.2 -> 1.2 px with the
    depth (the end at the back of the CCD diffuses most). Returns (rows, cols, counts)."""
    L = rng.uniform(5, 120) if L is None else L
    phi = rng.uniform(0, np.pi)
    n_e = int(MIP_E_PER_UM * np.hypot(L * PIX_UM, THICK_UM))
    t = rng.uniform(0, 1, n_e)
    sig = 0.2 + 1.0 * t
    yy = np.floor(y + (t - 0.5) * L * np.sin(phi) + rng.normal(0, 1, n_e) * sig).astype(int)
    xx = np.floor(x + (t - 0.5) * L * np.cos(phi) + rng.normal(0, 1, n_e) * sig).astype(int)
    ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    key, cnt = np.unique(yy[ok] * W + xx[ok], return_counts=True)
    return key // W, key % W, cnt, float(L), float(phi)


def read_noise(rng, shape, p):
    """read noise [e-]: Gaussian core sigma s, optional tail ('mix': f_tail of the pixels have k_tail x s; 't': Student-t
    with nu dof, scaled so that its central width (MAD) equals the Gaussian one)"""
    s = p["s"]
    if p["tail"] == "mix":
        e = rng.normal(0.0, s, shape)
        bad = rng.random(shape) < p["f_tail"]
        e[bad] *= p["k_tail"]
        return e
    if p["tail"] == "t":
        from scipy import stats
        t = rng.standard_t(p["nu"], shape)
        return t * s * stats.norm.ppf(0.75) / stats.t.ppf(0.75, p["nu"])
    return rng.normal(0.0, s, shape)


def simulate(rng, **kw):
    """one image: returns Q [e-] (H x (W + n_os)), valid (active columns), truth list, the parameters used and the true
    per-pixel dark-charge rate map lam_true (dark + serial + hot columns / pixels; events excluded)"""
    p = default_params(**kw)
    H, W, n_os = p["H"], p["W"], p["n_os"]
    Wt = W + n_os
    valid = np.zeros((H, Wt), bool); valid[:, :W] = True
    # dark charge (active area only), with an optional linear gradient along the rows
    lam_r = p["lam"] * (1.0 + p["lam_grad"] * np.arange(H) / max(H - 1, 1))
    rate = np.zeros((H, Wt)); rate[:, :W] = lam_r[:, None]
    # serial register: every pixel of a row incl. overscan, optional per-row scatter of the rate
    if p["lam_ser"] > 0:
        rs = p["lam_ser"] * np.clip(1.0 + p["ser_row_scatter"] * rng.normal(size=H), 0.0, None)
        rate += rs[:, None]
    for c in p["hot_cols"]:
        rate[:, c] += p["hot_col_rate"]
    hot_pix = []
    for _ in range(p["n_hot_pix"]):
        r, c = rng.integers(0, H), rng.integers(0, W)
        rate[r, c] += rng.uniform(*p["hot_pix_rate"]); hot_pix.append((int(r), int(c)))
    N = rng.poisson(rate).astype(float)
    truth = []
    taken = []
    # large deposits first (they need room), then the few-electron events away from them
    for kind, n in (("fe55", p["n_fe55"]), ("muon", p["n_muon"])):
        for (y, x, _) in _place(rng, n, H, W, 2 * p["margin"], 3 * p["sep"], taken):
            if kind == "fe55":
                sig = rng.uniform(0.5, 1.5)
                ys, xs, cnt = deposit(rng, y, x, 1600, sig, H, W)
                truth.append(dict(kind=kind, y=y, x=x, n_e=1600, sigma=sig, pix=(ys, xs, cnt)))
            else:
                ys, xs, cnt, L, phi = muon(rng, y, x, H, W)
                truth.append(dict(kind=kind, y=y, x=x, n_e=int(cnt.sum()), sigma=np.nan, L=L, phi=phi, pix=(ys, xs, cnt)))
            taken.append((y, x, 3 * p["sep"] + (60 if kind == "muon" else 0)))
            N[ys, xs] += cnt
    lo, hi = p["ne_range"]
    for (y, x, _) in _place(rng, p["n_events"], H, W, p["margin"], p["sep"], taken):
        n_e = int(rng.integers(lo, hi + 1))
        sig = float(rng.uniform(*p["sig_range"]))
        ys, xs, cnt = deposit(rng, y, x, n_e, sig, H, W)
        truth.append(dict(kind="fe", y=y, x=x, n_e=n_e, sigma=sig, pix=(ys, xs, cnt)))
        N[ys, xs] += cnt
    Q = N + read_noise(rng, (H, Wt), p)
    if p["f_out"] > 0:
        bad = rng.random((H, Wt)) < p["f_out"]
        Q[bad] += rng.choice([-1.0, 1.0], bad.sum()) * rng.uniform(p["out_lo"], p["out_hi"], bad.sum())
    if p["ped_row"] > 0:
        Q += rng.normal(0.0, p["ped_row"], H)[:, None]
    if p["first_col"] > 0 or p["first_col_jit"] > 0:
        Q[:, 0] += p["first_col"] + rng.normal(0.0, p["first_col_jit"], H)
    p["hot_pix_list"] = hot_pix
    return Q, valid, truth, p, rate


def truth_rows(truth, image_id):
    """flat truth table rows (pixels as 'r:c:n;...')"""
    out = []
    for i, t in enumerate(truth):
        ys, xs, cnt = t["pix"]
        out.append(dict(image=image_id, ev=i, kind=t["kind"], y=t["y"], x=t["x"], n_e=t["n_e"], sigma=t["sigma"],
                        n_pix=int(len(ys)), n_e_in=int(cnt.sum()), max_pix=int(cnt.max()) if len(cnt) else 0,
                        pixels=";".join(f"{a}:{b}:{c}" for a, b, c in zip(ys, xs, cnt))))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    Q, valid, truth, p, rate = simulate(rng, n_fe55=2, n_muon=1, hot_cols=(100,), hot_col_rate=0.01, tail="mix", f_tail=1e-3)
    print(Q.shape, valid.sum(), len(truth), [t["n_e"] for t in truth][:10])
