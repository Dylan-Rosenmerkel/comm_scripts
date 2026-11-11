"""Synthetic images with controlled placement for the hand-over tests (sim.py's deposit / muon generators):

  halo_image      Fe-55 deposits and muon tracks, each with small events (2-10 e-) at a chosen distance d from the
                  deposit (Fe-55: from its centre; muon: perpendicular from a point of the track)
  boundary_image  read noise per 100 x 100 block in a checkerboard (s_lo / s_hi) or uniform, small events whose
                  centre lies within +-1.5 px of a block edge or corner

Truth dicts as sim.simulate (kind 'fe55' / 'muon' / 'fe', y, x, n_e, sigma, pix), plus d (halo) / edge (boundary).
"""
import zlib

import numpy as np

import sim

HALO_D = (5, 8, 10, 12, 15, 18, 20, 25, 30, 40)


def seed(name, i):
    return np.random.SeedSequence([zlib.crc32(name.encode()), i, 77])


def _small(rng, y, x, H, W, ne_range=(2, 10), sig_range=(0.2, 1.2)):
    n_e = int(rng.integers(ne_range[0], ne_range[1] + 1))
    sg = float(rng.uniform(*sig_range))
    ys, xs, cnt = sim.deposit(rng, y, x, n_e, sg, H, W)
    return dict(kind="fe", y=y, x=x, n_e=n_e, sigma=sg, pix=(ys, xs, cnt))


def _finish(rng, N, H, W, s_map, lam, truth):
    N = N + rng.poisson(lam, (H, W))
    for t in truth:
        ys, xs, cnt = t["pix"]
        N[ys, xs] += cnt
    Q = N + rng.normal(0.0, 1.0, (H, W)) * s_map
    return Q, np.ones((H, W), bool)


def halo_image(rng, s=0.15, lam=1e-4, H=480, W=1280, n_fe55=4, n_muon=2, k=0):
    """big deposits with two small events each, at distances from HALO_D (cycled with the image index k)"""
    truth, centres = [], []
    N = np.zeros((H, W))
    ds = list(HALO_D)
    j = k * 7
    tries = 0
    while len(centres) < n_fe55 + n_muon and tries < 5000:
        tries += 1
        kind = "fe55" if len(centres) < n_fe55 else "muon"
        L = rng.uniform(20, 80) if kind == "muon" else 0.0
        rad = L / 2 + 45
        y, x = rng.uniform(rad, H - rad) if 2 * rad < H else H / 2, rng.uniform(rad, W - rad)
        if any(np.hypot(y - a, x - b) < rad + r + 10 for a, b, r in centres):
            continue
        centres.append((y, x, rad))
        if kind == "fe55":
            sg = rng.uniform(0.5, 1.5)
            ys, xs, cnt = sim.deposit(rng, y, x, 1600, sg, H, W)
            truth.append(dict(kind=kind, y=y, x=x, n_e=1600, sigma=sg, pix=(ys, xs, cnt)))
            for side in (0, 1):
                d = ds[j % len(ds)]; j += 1
                a = rng.uniform(0, np.pi) + side * np.pi
                ev = _small(rng, y + d * np.sin(a), x + d * np.cos(a), H, W)
                ev["d"] = d; ev["near"] = "fe55"
                truth.append(ev)
        else:
            ys, xs, cnt, L, phi = sim.muon(rng, y, x, H, W, L)
            truth.append(dict(kind=kind, y=y, x=x, n_e=int(cnt.sum()), sigma=np.nan, L=L, phi=phi, pix=(ys, xs, cnt)))
            for side in (-1, 1):
                d = ds[j % len(ds)]; j += 1
                tt = rng.uniform(0.2, 0.8)
                py, px = y + (tt - 0.5) * L * np.sin(phi), x + (tt - 0.5) * L * np.cos(phi)
                ev = _small(rng, py + side * d * np.cos(phi), px - side * d * np.sin(phi), H, W)
                ev["d"] = d; ev["near"] = "muon"
                truth.append(ev)
    # distance of each small event to the nearest pixel of a big deposit
    bp = np.concatenate([np.c_[t["pix"][0], t["pix"][1]] for t in truth if t["kind"] != "fe"])
    for t in truth:
        if t["kind"] == "fe":
            t["d_pix"] = float(np.min(np.hypot(bp[:, 0] + 0.5 - t["y"], bp[:, 1] + 0.5 - t["x"])))
    Q, valid = _finish(rng, N, H, W, s, lam, truth)
    return Q, valid, truth


def boundary_image(rng, s_lo=0.10, s_hi=0.20, lam=1e-5, H=400, W=1000, pattern="checker", n_ev=40, block=100):
    """noise per block: 'checker' (s_lo / s_hi alternating), 'uniform' (s_lo); n_ev small events on block edges and
    corners (centre within +-1.5 px of the edge), >= 20 px apart"""
    bi = (np.arange(H) // block)[:, None]; bj = (np.arange(W) // block)[None, :]
    s_map = np.where((bi + bj) % 2 == 1, s_hi, s_lo) if pattern == "checker" else np.full((H, W), s_lo)
    truth = []
    tries = 0
    while len(truth) < n_ev and tries < 20000:
        tries += 1
        kind = rng.integers(0, 3)          # 0: horizontal edge, 1: vertical edge, 2: corner
        ey = block * rng.integers(1, H // block) if kind in (0, 2) else rng.uniform(8, H - 8)
        ex = block * rng.integers(1, W // block) if kind in (1, 2) else rng.uniform(8, W - 8)
        y = ey + (rng.uniform(-1.5, 1.5) if kind in (0, 2) else 0.0)
        x = ex + (rng.uniform(-1.5, 1.5) if kind in (1, 2) else 0.0)
        if any(np.hypot(y - t["y"], x - t["x"]) < 20 for t in truth):
            continue
        ev = _small(rng, y, x, H, W)
        ev["edge"] = ("h", "v", "corner")[kind]
        truth.append(ev)
    Q, valid = _finish(rng, np.zeros((H, W)), H, W, s_map, lam, truth)
    return Q, valid, truth, s_map
