"""Pooled hierarchical dark-charge model for the Bayesian finder (wip/lowdc/bayes).

    lam_ij(t) = lam0 * (1 + beta * u_i)  +  P_hotcol(j) E[h_j]  +  P_hotpix(ij) E[h_ij]      (u_i = row / (H - 1))

pooled over the T images of a group (same detector, same exposure / readout): lam0 and beta (rate rising along the
readout) by Poisson maximum likelihood on soft dark counts; hot columns and hot pixels are DISCRETE Bayesian components
that persist across images (events are transient and do not):
  hot column j   counts K_j (all rows, all images) ~ Poisson(mu_j + h E_j) with h ~ log-uniform [max(2 lam0, 1e-5), 1]
                 e-/pix vs Poisson(mu_j); prior P(hot) = pi_c (fitted, Beta(1, 1) on pi_c)
  hot pixel ij   exact per-image likelihoods G_t(mu) = sum_n Pois(n; mu) f(v_t - n) / f(v_t) over the T images:
                 hot = prod_t G_t(lam + h), h ~ log-uniform [1e-2, 10] e-/image, vs null = prod_t [(1 - q_ev) G_t(lam)
                 + q_ev Gbar_t] (q_ev = measured chance that an event deposit covers a given pixel of an image, Gbar_t
                 = mean of the 1..5-electron likelihoods); prior pi_p (fitted, Beta(1, 1e5)). Noise miscounts, dark
                 charge and events enter with their exact probabilities; only pixels that are candidates in >= 2
                 images are tested.
Soft dark counts: a candidate pixel contributes P(H0 | neighbourhood) x E[d | v, H0] (an event neighbourhood is
effectively masked: its area, < 0.1 % of the image, is not removed from the exposure). Pixels that are not candidates
contribute 0 (their posterior dark count is < r_min x lam). With one image (T = 1) hot pixels cannot be separated
from events and are not fitted; lam0 gets a weak Gamma prior (0.5 pseudo-electrons at 1e-5 e-/pix).
1-e- events are counted as dark charge: the two are indistinguishable pixel by pixel (lam = dark + 1-e- event rate).
"""
import numpy as np
from scipy import optimize, special, stats


def _plog(k, mu):
    """Poisson log-likelihood for a (soft, non-integer) count k"""
    return k * np.log(mu) - mu - special.gammaln(k + 1.0)


def _soft_counts(pixtabs, nbs):
    """per image: candidate rows, cols, soft dark counts, charged flags"""
    out = []
    for pt, nb in zip(pixtabs, nbs):
        p0 = np.ones(pt["r"].size)
        lut = {}
        for d in nb:
            if d.get("big") or d.get("halo"):
                lut[d["nb"]] = 0.0
            elif "p0" in d:
                lut[d["nb"]] = d["p0"]
        if pt["r"].size:
            p0 = np.array([lut.get(int(k), 1.0) for k in pt["nb"]])
        out.append((pt["r"], pt["c"], p0 * pt["ed0"], pt["charged"], 1.0 - p0))
    return out


def _hot_lines(K, mu, E, lam0, pi0):
    """hot lines (columns or rows): counts K ~ Poisson(mu + h E), h log-uniform [max(2 lam0, 1e-5), 1] e-/pix, vs
    Poisson(mu); prior fraction pi fitted (Beta(1, 1)). Returns (P(hot), E[h | hot], pi)."""
    hg = np.geomspace(max(2 * lam0, 1e-5), 1.0, 241)
    ok = E > 0
    lnull = _plog(K, np.maximum(mu, 1e-300))
    lh = _plog(K[:, None], mu[:, None] + hg[None, :] * E[:, None])
    lhot = special.logsumexp(lh, axis=1) - np.log(hg.size)
    pi = pi0
    for _ in range(50):
        lo = np.log(pi) + lhot - (np.log1p(-pi) + lnull)
        P = np.where(ok, special.expit(lo), 0.0)
        pi = (P.sum() + 1.0) / (ok.sum() + 2.0)
    wpost = np.exp(lh - special.logsumexp(lh, axis=1, keepdims=True))
    return P, (wpost * hg[None, :]).sum(1), pi


def fit(pixtabs, nbs, valids, a0=0.5, lam_prior=1e-5, fit_hotpix=True, pi_c0=1e-3, pi_p0=1e-5, hot_rows=False,
        blink=False, pi_b0=1e-5, blink_min=3):
    """pooled background of a group of images (same shape). Returns (list of lam maps, summary dict).
    Optional components (default off): hot_rows - rows with an excess rate (serial-register streak / glow along a
    row; same test as the hot columns), added to the lam map; blink - 'one strong pixel repeating': a pixel that in
    a fraction b (log-uniform 1e-2..0.5) of the images carries a burst of 2-8 e- (flat) and is quiet otherwise, vs
    the null (dark charge + chance event coverage q_ev) and vs the Poisson hot pixel; blinking pixels cannot be
    described by a Poisson rate, so P(blink) > 0.5 pixels are returned in summary['veto'] (to be masked)."""
    T = len(pixtabs)
    H, W = valids[0].shape
    sc = _soft_counts(pixtabs, nbs)
    expo = np.sum([v.astype(float) for v in valids], axis=0)           # images per pixel
    occ = np.zeros((H, W))
    ev_occ = 0.0
    for (r, c, s, ch, pev) in sc:
        np.add.at(occ, (r, c), ch.astype(float))
        ev_occ += float((ch * pev).sum())
    # soft dark counts are re-evaluated at the current lam map (EM fixed point, not one step: with a noise tail the
    # posterior dark count of a 1-e--like pixel depends on lam itself)
    nk = min(pt["L"].shape[1] for pt in pixtabs if pt["r"].size) if any(pt["r"].size for pt in pixtabs) else 1
    Ra = np.concatenate([pt["r"] for pt in pixtabs]).astype(int)
    Ca = np.concatenate([pt["c"] for pt in pixtabs]).astype(int)
    La = np.concatenate([pt["L"][:, :nk] for pt in pixtabs]) if Ra.size else np.zeros((0, nk))
    Wa = np.concatenate([1.0 - s_[4] for s_ in sc]) if Ra.size else np.zeros(0)        # P(H0 | neighbourhood)
    nn = np.arange(nk)

    def soft(lam_map):
        """sum over images of P(H0 | neighbourhood) E[d | v, lam_ij] per pixel"""
        out = np.zeros((H, W))
        if Ra.size:
            lam = np.maximum(lam_map[Ra, Ca], 1e-300)
            t = La + nn[None, :] * np.log(lam)[:, None] - lam[:, None] - special.gammaln(nn + 1)[None, :]
            pst = np.exp(t - special.logsumexp(t, axis=1, keepdims=True))
            np.add.at(out, (Ra, Ca), Wa * (pst * nn[None, :]).sum(1))
        return out

    u = np.arange(H) / max(H - 1, 1)
    erow = expo.sum(1)
    col_extra = np.zeros(W)
    row_extra = np.zeros(H)
    Pr, Ehr, pi_r = np.zeros(H), np.zeros(H), 0.0
    pix_extra = np.zeros((H, W))
    res = {}
    lam_row = np.full(H, lam_prior)
    for it in range(3):
        # 1) lam0, beta from the row profile, excluding the hot-column / hot-pixel excess found so far; EM iterations
        #    (soft counts at the current lam) to the fixed point
        for inner in range(200):
            cnt = soft(lam_row[:, None] + row_extra[:, None] + col_extra[None, :] + pix_extra)
            crow = cnt.sum(1) - (col_extra[None, :] * expo).sum(1) - (pix_extra * expo).sum(1) - row_extra * erow
            crow = np.clip(crow, 0, None)

            def nll(par, free_beta):
                l0 = np.exp(par[0]); b = par[1] if free_beta else 0.0
                mu = l0 * np.clip(1 + b * u, 1e-6, None) * erow
                # weak Gamma(a0, a0 / lam_prior) prior on lam0
                return -((crow * np.log(np.maximum(mu, 1e-300)) - mu).sum()
                         + (a0 - 1) * par[0] + par[0] - a0 / lam_prior * l0)
            x0 = np.log(max((crow.sum() + a0) / (erow.sum() + a0 / lam_prior), 1e-12))
            r0 = optimize.minimize(lambda p: nll(np.r_[p, 0.0], False), [x0], method="Nelder-Mead")
            r1 = optimize.minimize(lambda p: nll(p, True), [r0.x[0], 0.0], method="Nelder-Mead")
            use_beta = (r0.fun - r1.fun) > 4.5 and crow.sum() > 20      # ~3 sigma for one extra parameter
            lam0 = float(np.exp(r1.x[0] if use_beta else r0.x[0]))
            beta = float(r1.x[1]) if use_beta else 0.0
            new_row = lam0 * np.clip(1 + beta * u, 1e-6, None)
            done = np.max(np.abs(new_row - lam_row) / new_row) < 1e-4
            lam_row = new_row
            if done:
                break
        # 2) hot columns: a defect well above the dark rate, h log-uniform in [max(2 lam0, 1e-5), 1] e-/pix
        mu_c = ((lam_row + row_extra)[:, None] * expo).sum(0) + (pix_extra * expo).sum(0)
        Kc = cnt.sum(0)
        Pc, Eh, pi_c = _hot_lines(Kc, mu_c, expo.sum(0), lam0, pi_c0)
        col_extra = np.where(Pc > 1e-3, Pc * Eh, 0.0)
        # 2b) hot rows (optional): the same test along the rows (rate excess of a whole row: serial-register streak)
        if hot_rows:
            mu_r = ((lam_row[:, None] + col_extra[None, :]) * expo).sum(1) + (pix_extra * expo).sum(1)
            Pr, Ehr, pi_r = _hot_lines(cnt.sum(1), mu_r, erow, lam0, pi_c0)
            row_extra = np.where(Pr > 1e-3, Pr * Ehr, 0.0)
        # 3) hot pixels (persistence over images), exact per-image likelihoods: in image t a pixel with rate mu has
        #    G_t(mu) = sum_n Pois(n; mu) l_t(n) (l_t = f(v - n) / f(v); = e^-mu where the pixel is not a candidate).
        #    hot:  prod_t G_t(lam + h), h log-uniform 1e-2..10 e-/image
        #    null: prod_t [(1 - q_ev) G_t(lam) + q_ev Gbar_t]   (Gbar_t = mean_{n=1..5} l_t(n): an event deposit)
        #    so noise miscounts (v ~ 0.5 at s = 0.15), dark charge and events all enter with their exact probabilities
        res_hp = []
        pix_extra = np.zeros((H, W))
        if fit_hotpix and T >= 2:
            q_ev = ev_occ / max(expo.sum(), 1)
            lam_pix = lam_row[:, None] + row_extra[:, None] + col_extra[None, :]
            nk = 12
            # big deposits (and their halo) are transient by construction: their pixels do not count as occupancy
            tr = []
            for pt, nb in zip(pixtabs, nbs):
                bad = {d["nb"] for d in nb if d.get("big") or d.get("halo")}
                tr.append(np.array([int(k) not in bad for k in pt["nb"]], bool) if pt["r"].size else np.zeros(0, bool))
            keys = np.concatenate([(pt["r"] * W + pt["c"])[m] for pt, m in zip(pixtabs, tr)])
            Ls = np.concatenate([pt["L"][m, :nk] for pt, m in zip(pixtabs, tr)]) if keys.size else np.zeros((0, nk))
            uk, inv, ncand = np.unique(keys, return_inverse=True, return_counts=True)
            sel = ncand >= 2
            if sel.any():
                ent = sel[inv]
                uks = uk[sel]
                pidx = np.searchsorted(uks, keys[ent])
                Le = Ls[ent]
                rr, cc = uks // W, uks % W
                lam_p = lam_pix[rr, cc]
                Tn = expo[rr, cc]
                hh = np.geomspace(1e-2, 10.0, 81)
                n = np.arange(nk)
                mu = lam_p[:, None] + hh[None, :]                                # (npix, nh)
                # candidate entries: log G_t(mu) for every h
                lpo = n[None, None, :] * np.log(mu[pidx][:, :, None]) - mu[pidx][:, :, None] - special.gammaln(n + 1)[None, None, :]
                lg = special.logsumexp(Le[:, None, :] + lpo, axis=2)            # (nent, nh)
                L1h = np.zeros((uks.size, hh.size))
                np.add.at(L1h, pidx, lg)
                L1h += -(Tn - ncand[sel])[:, None] * mu                         # images where the pixel is quiet
                lhp = special.logsumexp(L1h, axis=1) - np.log(hh.size)
                lp0 = n[None, :] * np.log(lam_p[pidx][:, None]) - lam_p[pidx][:, None] - special.gammaln(n + 1)[None, :]
                lg0 = special.logsumexp(Le + lp0, axis=1)
                lgb = special.logsumexp(Le[:, 1:6], axis=1) - np.log(5.0)
                le0 = np.logaddexp(np.log1p(-q_ev) + lg0, np.log(max(q_ev, 1e-300)) + lgb)
                l0p = np.zeros(uks.size)
                np.add.at(l0p, pidx, le0)
                l0p += (Tn - ncand[sel]) * (np.log1p(-q_ev) - lam_p)
                npix_tot = float((expo > 0).sum())
                pi_p = pi_p0
                for _ in range(50):
                    Pp = special.expit(np.log(pi_p) + lhp - np.log1p(-pi_p) - l0p)
                    pi_p = (Pp.sum() + 1.0) / (npix_tot + 1e5)
                if blink:
                    # blinking pixel: per image (1 - b) G_t(lam) + b Gbar2_t, Gbar2_t = mean_{n=2..8} l_t(n) (a burst);
                    # quiet images: (1 - b) e^-lam (a burst would have made the pixel a candidate)
                    bb = np.geomspace(1e-2, 0.5, 25)
                    lgb2 = special.logsumexp(Le[:, 2:9], axis=1) - np.log(7.0)
                    lbt = np.logaddexp(np.log1p(-bb)[None, :] + lg0[:, None], np.log(bb)[None, :] + lgb2[:, None])
                    Lb = np.zeros((uks.size, bb.size))
                    np.add.at(Lb, pidx, lbt)
                    Lb += (Tn - ncand[sel])[:, None] * (np.log1p(-bb)[None, :] - lam_p[:, None])
                    lbl = special.logsumexp(Lb, axis=1) - np.log(bb.size)
                    # prior P(blink) = pi_b0 FIXED (not EM-updated): in a group with many events, pixels covered by
                    # events in two images are frequent and an EM-updated prior inflates itself on them
                    pi_b = pi_b0
                    lz3 = np.logaddexp.reduce(np.c_[np.log1p(-pi_p - pi_b) + l0p, np.log(pi_p) + lhp,
                                                    np.log(pi_b) + lbl], axis=1)
                    Pb = np.exp(np.log(pi_b) + lbl - lz3)
                    # a blinking pixel seen fewer than blink_min times is not vetoed (see below); it keeps the Poisson
                    # hot-pixel description (two-hypothesis P(hot)), whose raised lam still suppresses its bursts
                    Pp = np.where(ncand[sel] >= blink_min, np.exp(np.log(pi_p) + lhp - lz3), Pp)
                    res["blink_pix"] = [dict(r=int(a_), c=int(b_), n_img_cand=int(o), P=float(p))
                                        for a_, b_, o, p in zip(rr, cc, ncand[sel], Pb) if p > 0.01]
                    # veto: P(blink) > 0.5 AND the pixel fired (was a candidate) in >= blink_min images: two
                    # coincidences are common in busy groups (events), three are not
                    res["veto"] = [(int(a_), int(b_)) for a_, b_, p, o in zip(rr, cc, Pb, ncand[sel])
                                   if p > 0.5 and o >= blink_min]
                    res["pi_blink"] = float(pi_b)
                wh = np.exp(L1h - special.logsumexp(L1h, axis=1, keepdims=True))
                h_hat = (wh * hh[None, :]).sum(1)
                pix_extra[rr, cc] = np.where(Pp > 1e-3, Pp * h_hat, 0.0)
                res_hp = [dict(r=int(a_), c=int(b_), n_img_cand=int(o), T=int(t), P=float(p), rate=float(hq))
                          for a_, b_, o, t, p, hq in zip(rr, cc, ncand[sel], Tn, Pp, h_hat) if p > 0.01]
                res["pi_hotpix"] = float(pi_p); res["q_ev"] = float(q_ev)
    lam_map = lam_row[:, None] + row_extra[:, None] + col_extra[None, :] + pix_extra
    res.update(lam0=lam0, beta=beta, use_beta=bool(use_beta), n_soft=float(cnt.sum()), exposure_pix=float(expo.sum()),
               lam_mean=float((lam_map * expo).sum() / max(expo.sum(), 1)),
               hot_cols=[dict(c=int(j), P=float(Pc[j]), rate=float(Eh[j]), K=float(Kc[j]), mu=float(mu_c[j]))
                         for j in np.nonzero(Pc > 0.01)[0]],
               pi_hotcol=float(pi_c), hot_pix=res_hp,
               hot_rows=[dict(r=int(i), P=float(Pr[i]), rate=float(Ehr[i])) for i in np.nonzero(Pr > 0.01)[0]])
    res.setdefault("veto", []); res.setdefault("blink_pix", [])
    return [np.where(v, lam_map, 0.0) for v in valids], res
