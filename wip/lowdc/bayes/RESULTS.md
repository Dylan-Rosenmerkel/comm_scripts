# Bayesian few-electron finder: results (2026-11-10)

Method and how to run: `README.md`. All generated tables: `results/tables.md` (from `evaluate.py`); figures in `figs/`.
Raw records: `results/physics/` and `results/bench/` (`events.csv`, `objects.csv`, `images.csv`, `null.csv`,
`groups.json`), `results/agmx_F*.csv`, `results/proto_F*.csv`, `results/real_l2*/`.

Benchmark: the SAME images as `../bench_agmx.py` (configs and seeds reused; 1485 - 135 out-of-scope images, see
below). Groups = base + blank images of the same (s, lam) pooled (as in `../RESULTS.md` 'plain'); background and the
event-rate spectrum are fitted per group. Two sigma priors: `physics` (DAMIC diffusion law, sigma <= 1.01 px) and
`bench` (uniform 0.2-1.2 px, the simulation's own). Fakes: unmatched objects, < 20 e-, not near a big deposit, per
Mpix of all images of the group. Not run (out of scope - candidates are no longer sparse): s = 0.20 / 0.25 and
lam = 1e-2.

## 1. Headline: efficiency for 2 / 3 / 4 e- events

### 1a. At the model's own operating point (nominal 0.08 fakes / Mpix, set analytically)

Bayes: P(event) >= t with t chosen so that the posterior expected fake count sum_selected P(no event | data) =
0.08 x Mpix (no blanks, no tuning on truth). A_GMX and proto_box: their F = 0.08 runs (`../results/`). In brackets:
measured fakes (count, per Mpix) and, for Bayes, the predicted rate.

| group (Mpix) | Bayes physics prior | Bayes bench prior | A_GMX | proto_box |
|---|---|---|---|---|
| base s0.05 l1e-5 (9.2) | 0.98/1.00/1.00 (0, 0.00; pred 0.02) | 0.98/1.00/1.00 (0) | 0.61/0.93/1.00 (0) | 0.95/1.00/1.00 (0) |
| base s0.06 l3e-6 (9.2) | 1.00/1.00/1.00 (0; 0.01) | 1.00/1.00/1.00 (0) | 0.73/0.96/1.00 (0) | 0.98/1.00/1.00 (0) |
| plain s0.06 l1e-5 (46) | **1.00**/1.00/1.00 (0; 0.01) | 1.00/1.00/1.00 (0) | 0.56/0.98/0.98 (0) | 0.90/1.00/1.00 (0) |
| plain s0.10 l1e-5 (46) | **1.00**/1.00/1.00 (1, 0.02; 0.03) | 1.00/1.00/1.00 (1) | 0.62/0.90/0.98 (0) | 0.97/1.00/1.00 (0) |
| base s0.15 l3e-6 (9.2) | 1.00/1.00/1.00 (0; 0.03) | 1.00/1.00/1.00 (0) | 0.24/0.81/0.93 (0) | 0.22/0.89/0.97 (0) |
| plain s0.15 l1e-5 (55) | **0.97**/1.00/1.00 (2, 0.04; 0.07) | 0.97/1.00/1.00 (2) | 0.20/0.80/0.93 (3, 0.05) | 0.13/0.84/0.99 (1, 0.02) |
| base s0.06 l1e-4 (9.2) | 0.79/1.00/1.00 (1, 0.11; 0.08) | 0.79/1.00/1.00 (1) | 0.61/0.96/0.98 (1, 0.11) | 0.21/1.00/1.00 (0) |
| base s0.10 l1e-4 (9.2) | 0.78/0.98/1.00 (0; 0.07) | 0.78/0.98/1.00 (0) | 0.40/0.80/0.98 (2, 0.22) | 0.14/1.00/1.00 (0) |
| plain s0.15 l1e-4 (55) | **0.72**/0.99/1.00 (7, 0.13; 0.08) | 0.73/0.99/1.00 (7) | 0.12/0.82/0.94 (6, 0.11) | 0.00/0.89/0.99 (0) |
| plain s0.15 l1e-3 (55) | 0.03/**0.91**/0.99 (2, 0.04; 0.07) | 0.03/0.93/0.99 (3) | 0.00/0.65/0.85 (4, 0.07) | 0.00/0.05/0.74 (0) |
| tail s0.15 l1e-4 (9.2) | 0.14/0.87/0.98 (2, 0.22; 0.07) | 0.16/0.87/1.00 (2) | 0.04/0.51/0.82 (1, 0.11) | 0.00/0.46/0.76 (0) |
| hotcol s0.15 l1e-5 (9.2) | 1.00/1.00/1.00 (0; 0.04) | 1.00/1.00/1.00 (0) | 0.23/0.70/0.96 (3, 0.33) | 0.08/0.74/1.00 (1) |
| art s0.10 l1e-5 (9.2) | 0.79/0.95/0.95 (3, 0.33; 0.08) | 0.79/0.95/0.95 (3) | 0.42/0.72/0.98 (65, 7.1) | - |
| busy s0.15 l1e-4 (9.2) | 0.73/0.97/1.00 (0; 0.08) | 0.73/0.97/1.00 (0) | 0.24/0.72/0.96 (1) | - |
| **L2 100x500 s0.06 l1e-5** (10) | **0.99**/1.00/1.00 (1, 0.10; 0.03) | 0.99/1.00/1.00 (1) | 0.64/0.90/1.00 (9, 0.90) | 0.76/0.99/1.00 (0) |
| **L2 100x500 s0.10 l1e-5** (10) | **0.96**/1.00/1.00 (0; 0.04) | 0.96/1.00/1.00 (0) | 0.51/0.85/0.99 (7, 0.70) | 0.77/1.00/1.00 (0) |
| L2 100x500 s0.15 l1e-4 (10) | 0.78/1.00/1.00 (0; 0.08) | 0.78/1.00/1.00 (0) | 0.20/0.68/0.95 (3, 0.30) | 0.08/0.76/0.99 (0) |

Efficiency errors: 45-90 events per n_e per 15-30 images (+-0.02..0.07); the plain groups have 90-180.

### 1b. ROC (threshold swept; `figs/roc_2e.png`, `figs/roc_3e.png`; all groups in `results/tables.md` section 1)

Bayes score = log J1 (Bayes factor of one N ~ U{2..20} e- event vs background, per unit event rate; independent of
the fitted rate R). Selected points (2 e- / 3 e- efficiency, measured fakes):

| group | Bayes log J1 >= 10 | >= 12 | >= 14 | A_GMX count_fakes 0.01 / 0.08 / 1 | proto_box F 0.01 / 0.08 / 1 / 10 |
|---|---|---|---|---|---|
| plain s0.06 l1e-5 (46 Mpix) | 1.00/1.00, 0 | 1.00/1.00, 0 | 0.95/1.00, 0 | 0.51/0.93 (0), 0.56/0.98 (0), 0.76/0.98 (98 = 2.1/Mpix) | 0.56, 0.90, 1.00 (all 0 fakes), 1.00 (1) |
| plain s0.10 l1e-5 (46) | 1.00/1.00, 0 | 1.00/1.00, 0 | 1.00/1.00, 0 | 0.53/0.88 (0), 0.62/0.90 (0), 0.72/0.94 (61) | 0.45, 0.97, 1.00 (1), 1.00 (3) |
| plain s0.15 l1e-5 (55) | 0.95/1.00, 0 | 0.94/1.00, 0 | 0.90/1.00, 0 | 0.17/0.72 (2), 0.20/0.80 (3), 0.37/0.87 (18) | 0.03 (1), 0.13 (1), 0.22 (4), 0.77 (44) |
| plain s0.15 l1e-4 (55) | 0.94/0.99, 12 (0.22/Mpix) | 0.70/0.99, 6 (0.11) | 0.16/0.99, 1 (0.02) | 0.09/0.77 (6), 0.12/0.82 (6), 0.28/0.88 (23) | 0.00 (0), 0.00 (0), 0.17 (2), 0.63 (56) |
| L2 s0.06 l1e-5 (10) | 0.99/1.00, 1 | 0.99/1.00, 0 | 0.88/0.99, 0 | 0.64/0.90 (9), 0.64/0.90 (9), 0.69/0.91 (25) | 0.42, 0.76, 0.85 (0), 0.99 (1) |
| L2 s0.10 l1e-5 (10) | 0.96/1.00, 0 | 0.96/0.99, 0 | 0.78/0.99, 0 | 0.51/0.85 (7), 0.51/0.85 (7), 0.59/0.92 (20) | 0.36, 0.77, 0.88 (0), 1.00 (0) |

Matched fake rates (`tables.md` section 1, 'best efficiency with measured fakes <= F x Mpix'): at 0.01 / 0.08 / 1
per Mpix the Bayesian 2-e- efficiency is 1.00 / 1.00 / 1.00 at s0.06 and s0.10 (lam 1e-5), 0.97 / 0.97 / 0.98 at
s0.15 l1e-5, 0.01 / 0.43 / 0.98 at s0.15 l1e-4, 0.99 / 0.99 / 0.99 and 0.96 / 0.96 / 0.96 on the L2 geometry; A_GMX
has no point at <= 0.08 / Mpix in s0.15 l1e-5 / l1e-4 or on the L2 geometry (it is 0.05-0.11 and 0.7-0.9 / Mpix),
and reaches 0.37 / 0.28 / 0.64 / 0.51 at <= 1 / Mpix. Caveat: with < 1 allowed fake in the group's Mpix 'matched'
means 'above the highest fake', which is optimistic by construction for all methods; section 1a is the honest
operating point.

**Summary.** At the L2 target (s 0.06-0.10, lam <= 1e-5) 2-e- events go from 0.51-0.64 (A_GMX) and 0.76-0.97
(proto_box) to 0.96-1.00, 3-e- events from 0.85-0.98 to 1.00, with fakes at or below nominal (A_GMX: 10x above on
L2). At s = 0.15, lam <= 1e-5 the gain is largest: 2 e- 0.20 -> 0.97, 3 e- 0.80 -> 1.00. Where 2 e- are limited by
chance pairs of dark electrons (lam 1e-4) the Bayes factor still wins (0.72 vs 0.12 at s0.15, 0.78-0.79 vs
0.40-0.61 at s <= 0.10) and 3 e- reach 0.98-1.00. lam = 1e-3: 2-e- events are not detectable by anyone (pairs of
dark electrons: 0.03 at 0.04 fakes/Mpix); 3 e- 0.91 vs 0.65 (A_GMX) and 0.05 (proto_box).

## 2. False-alarm control: analytic vs measured vs data-driven null

Posterior expected fakes (sum P(no event) over the selection) vs measured (`tables.md` section 2), physics prior,
nominal 1 fake / Mpix (the regime with enough fakes to compare):

| group | predicted | measured | off-image null (per permutation) |
|---|---|---|---|
| plain s0.15 l1e-5 (55 Mpix) | 54.8 | 46 | 46.5 |
| plain s0.15 l1e-4 | 54.6 | 49 | 48.5 |
| plain s0.15 l1e-3 | 54.8 | 51 | 81 |
| L2 s0.15 l1e-4 (10 Mpix) | 9.5 | 9 | 8.5 |
| hotcol s0.15 l1e-4 | 8.6 | 7 | 5.5 |
| tail s0.15 l1e-4 | 8.6 | 8 | 11 |
| busy s0.15 l1e-4 | 8.4 | 5 | 8 |
| art s0.10 l1e-5 | 8.9 | **28** | 1 |
| art s0.10 l1e-5, v2 `point_defects` (section 13) | 8.7 | 12 | 6.5 |

- The analytic prediction is calibrated to ~10-20 % (slightly conservative) wherever the model holds; at 0.08 / Mpix
  the measured counts (0-7 per 9-55 Mpix) agree within Poisson errors except `art` (below).
- The off-image null (candidates of accepted events removed, remaining candidates permuted among the images per
  pixel location, finder rerun; 2 permutations) reproduces the measured fake counts per permutation (46.5 vs 46,
  48.5 vs 49, 8.5 vs 9); at lam 1e-3 it is 1.6x high (the permutation also scatters sub-threshold event electrons,
  which at that density add chance pairs). It does not see `art`'s fakes: they are single-pixel deposits, which a
  per-pixel permutation keeps intact (see section 7).
- Blank images alone (no events, so P(event) uses the group's fitted R): at the 0.08 / Mpix operating point 0, 0,
  2, 6, 0 fakes in 36.9 Mpix of blanks for s0.06 l1e-5, s0.10 l1e-5, s0.15 l1e-5, s0.15 l1e-4, s0.15 l1e-3.
- No 1-electron object is ever reported (N >= 2 by construction): the L2-geometry single-pixel fakes of A_GMX
  (0.7-0.9 / Mpix) are gone, and the fake rate does not depend on the image size (no blank simulations; the
  threshold comes from the posterior).

## 3. Electron-count (energy) response

Events with P(event) > 0.5, base/hotcol/grad/L2 groups (n = 900-1150 per n_e), MAP N vs true n_e:

| n_e | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| Bayes posterior mean N | 2.01 | 3.00 | 3.99 | 4.99 | 5.99 | 6.99 | 8.00 | 8.99 | 9.98 |
| Bayes exact | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.98 | 0.99 | 0.99 | 0.99 |
| Bayes low by >= 1 | 0.000 | 0.008 | 0.007 | 0.009 | 0.005 | 0.008 | 0.004 | 0.004 | 0.009 |
| A_GMX exact (`../RESULTS.md`) | 0.93 | 0.85 | 0.75 | 0.75 | 0.70 | 0.67 | 0.69 | 0.67 | 0.70 |
| A_GMX low by >= 1 | 0.04 | 0.13 | 0.22 | 0.21 | 0.26 | 0.30 | 0.28 | 0.29 | 0.27 |

No event split into two accepted objects (A_GMX: 3-9 % at 6-10 e-): the distance grouping keeps non-touching
electrons together and the exact count marginalisation assigns them. Tail: exact 0.96 (low 0.011-0.017); busy 0.98;
art 1.00. `figs/energy_sigma.png` (left).

## 4. Diffusion sigma and depth

Posterior-mean sigma vs true sigma (events n_e >= 4, P > 0.5; `figs/energy_sigma.png` right): correlation 0.85
(physics prior) / 0.86 (bench prior). Per true-sigma bin, posterior mean (physics / bench prior): 0.2-0.4 px:
0.38 / 0.39; 0.4-0.6: 0.55 / 0.55; 0.6-0.8: 0.70 / 0.73; 0.8-1.0: 0.80 / 0.87; 1.0-1.2: **0.87 / 0.97**. Per-event
resolution ~0.09-0.15 px. The physics prior stops at 1.01 px (the DAMIC law) so the benchmark's 1.0-1.2 px events
are pulled down (misspecification, as intended); detection and energies are insensitive to the prior (all
efficiencies within 0.02, section 1). Depth (physics prior): posterior-mean z 154 / 273 / 395 / 488 / 549 um vs true
86 / 221 / 390 / 574 / 671 um in the same bins - shrinkage to the prior mean at the ends, as expected with 4-10
electrons; depth is a population-level quantity at these charges.

## 5. Dark charge

Pooled model (all images of a group) vs truth (`tables.md` section 5). The truth that the data can measure is
dark charge + the 1-e- event density (indistinguishable pixel by pixel), column '+1e':

| group | true (dark + serial + hot) | + 1-e- events | fit | fit / (+1e) | single-image fallback (median, 16-84 %) | A_GMX detection lam |
|---|---|---|---|---|---|---|
| plain s0.06 l1e-5 | 1.0e-5 | 1.10e-5 | 1.06e-5 | 0.96 | 9.8e-6 (6.8e-6..1.4e-5) | 9.5e-5 |
| plain s0.10 l1e-5 | 1.0e-5 | 1.09e-5 | 1.09e-5 | 1.00 | 9.8e-6 (6.8e-6..1.4e-5) | 1.7e-4 |
| plain s0.15 l1e-5 | 1.0e-5 | 1.15e-5 | 1.09e-5 | 0.94 | 8.7e-6 (5.6e-6..1.4e-5) | 2.9e-4 |
| plain s0.15 l1e-4 | 1.0e-4 | 1.02e-4 | 1.01e-4 | 0.99 | 8.6e-5 | 3.3e-4 |
| plain s0.15 l1e-3 | 1.0e-3 | 1.00e-3 | 9.87e-4 | 0.99 | 8.0e-4 | 1.1e-3 |
| base s0.06 / s0.15 l3e-6 | 3.0e-6 | 9.5e-6 / 8.9e-6 | 9.4e-6 / 8.6e-6 | 0.99 / 0.97 | | 1e-4 / 2.6e-4 |
| L2 s0.06 / s0.10 l1e-5 | 1.0e-5 | 1.8e-5 / 1.9e-5 | 1.9e-5 / 2.0e-5 | 1.05 / 1.01 | 1.5e-5 (5e-6..2.5e-5) | 1.0e-4 / 1.7e-4 |
| grad s0.15 l1e-4 (x3 along rows) | 2.0e-4 | 2.05e-4 | 2.07e-4 (lam0 1.07e-4, beta 1.86; true 1e-4, 2) | 1.01 | | 3.2e-4 |
| hotcol s0.15 l1e-5 (+0.02 in col 640) | 2.6e-5 | 3.0e-5 | 3.4e-5, col 640 found (P = 1) | 1.11 | | 2.7e-4 |
| tail s0.15 l1e-4 | 1.0e-4 | 1.05e-4 | 1.00e-4 | 0.95 | 2.0e-5 (!) | 8.4e-4 |
| tail s0.15 l1e-5 | 1.0e-5 | 1.5e-5 | 2.8e-5 | **1.89** | 3.7e-6 (!) | 7.7e-4 |
| art s0.10 l1e-5 | 3.8e-5 | 4.2e-5 | 3.3e-5 | 0.78 | 3.0e-5 | 2.2e-4 |

- Pooled: within 0.94-1.05 of (dark + 1-e- events) in every Gaussian-noise group from 3e-6 to 1e-3 (A_GMX's
  detection lam: 9-30x high at 1e-5). The row gradient is found (beta 1.86 vs 2) and the hot column is found with
  P = 1 in all three hotcol groups (A_GMX never flagged it). In the benchmark the 1-e- events are a large part of
  the 'dark' rate at 3e-6 (x3); a real run has far fewer events, so this matters less but never vanishes.
- With the noise tail at lam 1e-5 the fit is 1.9x high: 0.5 % of pixels at 3 x s = 0.45 e- put ~150 pixels per image
  near 1 e-, against 6 dark electrons; the tail model is fitted from the negative side, and the remaining 1.9x is the
  limit of how well the tail shape (a single Gaussian component) is known. At lam 1e-4 it is unbiased.
- The single-image fallback has the expected Poisson scatter (+-40 % at 1e-5 on 0.6 Mpix) and is biased low (x0.2-0.4)
  with the noise tail (neighbourhoods of tail pixels are excluded from it); pooling fixes both.
- Hot pixels: the simulation draws new 'hot pixels' in every image (not persistent), so none can be found by
  persistence (none were, correctly: no false hot pixels in any group after the exact per-image likelihood was used;
  a first version with an occupancy count flagged 620 noise pixels at s0.15 - see section 7).
- R (events with N >= 2 per pixel) from the population EM equals the truth to < 1 % in every group (e.g. 1.47e-5 vs
  1.48e-5 at s0.15 l1e-5, 7.22e-5 vs 7.22e-5 L2) - it uses every neighbourhood with its fractional responsibility.

## 6. Runtime (single thread, M-series Mac, machine shared with other jobs)

| image | noise | candidates + evidences (2 passes) | background (per image) | total | A_GMX (`../RESULTS.md`) |
|---|---|---|---|---|---|
| 480 x 1280, s <= 0.10, lam 1e-5 | 0.17 s | 0.21 s | 0.002 s | **0.38 s** | 3.9 s |
| 480 x 1280, s <= 0.10, lam 1e-4 | 0.17 s | 0.45 s | 0.006 s | 0.63 s | 3.9 s |
| 480 x 1280, s = 0.15 | 0.17 s | 1.2-1.6 s | 0.07 s | 1.4-1.9 s | 3.9 s |
| 480 x 1280, s = 0.15 + tail | 0.17 s | 6-8.4 s | 0.1-0.3 s | 6.5-8.7 s | 3.9 s |
| 100 x 500, s <= 0.10 | 0.025 s | 0.054 s | < 1 ms | **0.08 s** | 0.57 s |
| 100 x 500, s 0.15 l1e-4 | 0.024 s | 0.15 s | 0.008 s | 0.19 s | 0.57 s |

Not included: the optional off-image null (one more evidence pass per permutation). The cost scales with the number
of candidate pixels: ~30 per 0.6 Mpix at s <= 0.10, ~2700 at s = 0.15, ~7000 with the tail (0.5 % of pixels at 3 s
are candidates) - there the method is no longer sparse and slower than A_GMX. 'pass 2' repeats all evidences with the
pooled lam; reusing pass 1 where lam barely changes would halve t_find.

## 7. Real data: L2 ACM 103 noise test (`real_l2.py`, `results/real_l2/`)

20 images of 100 x 500 (two per NDCM 100 ... 5000), column 0 masked, per extension; noise model and lam pooled
within each NDCM pair; R_N and the hot-pixel / hot-column persistence test over the 14 images with NDCM >= 500
(below 500 the noise is 0.25-0.5 e- and 20-25 % of the pixels are candidates: out of scope, listed but not used).
**Gain 1.3 ADU/e- assumed (unconfirmed); every electron number below is ADU / 1.3.**

- **Gain fit**: the profile likelihood over g = 0.7-2.2 ADU/e- is flat (Delta lnL < 0.5) for every extension
  (MLE 1.20 / 1.23 / 1.25 / 0.82, meaningless): the >= 4 s pixels at NDCM >= 1000 are 240-600 per CCD, almost all of
  them the noise tail, and only a handful are 1-e- pixels. The gain cannot be fitted from these images.
- **Noise model** (core s in e-): EXT1-3 0.147 (500) -> 0.105 (1000) -> 0.062 (3000) -> 0.048 (5000); EXT4 0.092 ->
  0.029 (in units of 1.3 ADU). A wider component with 0.3-1 % weight at 1.3 x core (the lower bound) is fitted in
  every group: the core is slightly non-Gaussian.
- **Noise symmetry**: EXT1-3 symmetric at 2.5-4 s (pos/neg 4082/4257, 753/728, 148/115, 22/14 for EXT1; |z| <= 2
  everywhere). **EXT4 is strongly asymmetric**: negative side heavier at every level (2.5 s: 4735 vs 6143,
  z = -13.5; 4 s: 314 vs 1161, z = -22) and below -5 e-: 99 pixels vs 8 above +5 e-. The symmetric noise model
  therefore over-predicts EXT4's positive tail (broad component weight 1-5e-3 at s ~ 2-20 e-), which makes the
  finder conservative there.
- **Dark charge** (soft counts, NDCM >= 500, 0.70 Mpix-images per CCD): EXT1 2.7 electrons -> 3.8e-6 e-/pix/image,
  EXT2 5.3 -> 7.6e-6, EXT3 5.6 -> 8.0e-6 (Poisson 68 % ranges roughly x0.6..x1.6); EXT4 0.05 electrons at g = 1.3
  (2.6 at g = 0.8: EXT4's charge scale is unknown). Per NDCM pair 0-4 dark electrons (most at 5000 skips, the
  longest readout). Consistent with the 'of order 1e-5 or less' in the notes.
- **Hot pixels / columns**: none (no pixel or column with P(hot) > 0.1 in any CCD). A first version (occupancy counts
  incl. big deposits and low-NDCM images) flagged a 141-pixel region in EXT1 (rows 45-56, cols 322-351); it was a big
  deposit's footprint plus noise candidates, not a defect - with big deposits treated as transient and NDCM >= 500
  it disappears.
- **Few-electron candidates**: EXT1-3 have NO neighbourhood with log J1 > 0 at NDCM >= 500 (not a single 2-e--like
  configuration). EXT4 has 12 with log J1 0.2-6.6, mostly single pixels at 1.8-2.4 'e-' (and one at 4.1, one pair
  7.7 + 3.6); with the fitted R (2.7e-8, i.e. no events) P(event) <= 2e-5, and even with R = 1e-5 (the benchmark's
  scale) the best one has posterior odds e^6.6 x 1e-5 ~ 7e-3. The off-image null gives a similar count (4 per 4
  permutations above log J1 = 5 in the two groups that have one). Big deposits (> 20 e-): 1 in EXT1, 2 in EXT4.
  Nothing in this data set is a significant few-electron event.
- EXT4 with g = 0.8 (`results/real_l2_ext4_g0.8/`): the same picture (top log J1 6.6, a pair 12.5 + 5.9 'e-').

## 8. Failure cases and limits

1. **Two dark electrons look like a 2-e- event.** At lam 1e-4 the 2-e- efficiency at 0.08 fakes / Mpix is 0.72-0.79
   (pairs of dark electrons in touching pixels; the fakes at log J1 > 9 are 2-e- pairs, 93 % in two pixels), at 1e-3
   it is 0.03. This is the information limit, not a model error (the predicted fakes match the measured ones).
   Diffuse 2-e- events (sigma > 0.8) suffer most (0.52 vs 0.95 at sigma < 0.5, s0.15 l1e-4).
2. **Non-persistent single-pixel deposits** (v2, section 13.1: with the optional point-deposit component the measured /
   predicted fakes at 1 / Mpix are 12 / 8.7 instead of 28 / 8.9) (`art`: per-image 'hot pixels' of 0.1-3 e-, excursions of 1-10 e-):
   single-pixel 2-10 e- 'events' at log J1 10-46. The broad noise component (pooled from the negative excursions)
   suppresses the excursions (A_GMX: 7.1 fakes / Mpix, Bayes 0.33 at the operating point), but a single pixel with an
   integer-like charge is a legitimate small-sigma event under the model; only per-skip data or a real persistence
   can veto it. The analytic fake prediction is 3x low there (model misspecified) and the per-pixel shuffle null
   cannot see these fakes.
3. **Noise tail at s = 0.15** (0.5 % of pixels at 0.45 e-): 2-e- efficiency 0.14 at nominal fakes (A_GMX 0.04; unchanged
   by the v2 non-parametric density, section 13.2);
   the method is no longer sparse (7000 candidates / image, 6-9 s per image) and lam at 1e-5 is 1.9x high.
4. **sigma > 1.01 px under the physics prior**: posterior sigma saturates at ~0.9 and depth at 550 um; the detection
   is unaffected (benchmark and physics prior give the same efficiencies).
5. EXT4-like one-sided noise is handled conservatively, not correctly (symmetric model).

## 9. What is still heuristic

- Candidate floor r_min = 1e-2 and linking radius 4 px (bounded evidence error, but fixed numbers); the 40-pixel cap
  for large neighbourhoods (never reached in the benchmark) and the 20-e- 'big' cut with a 15-px halo.
- Two events in one neighbourhood: only spatially separated bipartitions (3 longest single-linkage cuts); >= 3
  events not modelled. Events never shared a neighbourhood in the benchmark (>= 20 px apart), so this part is
  untested on truth.
- Noise: 3-component Gaussian scale mixture with bounds (tail <= 0.6 e-, broad >= 1 e-); per-row pedestal profiled,
  not marginalised; column offset / scale only above a 6-sigma significance cut; symmetry assumed.
- Background: hot-column / hot-pixel rate priors (log-uniform ranges) and their prior fractions; event
  neighbourhoods are masked by weight P(H0) rather than modelled; beta kept above a 2 dlnL > 9 cut.
- p(N) uniform 2..20 for the score; sigma grid 0.1-1.3 px in 0.1 steps; centre grid step min(0.25, sigma / 2).

## 10. Recommended next steps

1. Get the gain (LED / high-DC image) and rerun `real_l2.py`; with ~1e-5 e-/pix the noise-test images cannot
   calibrate it. Clarify EXT4 (amplifier, gain, the one-sided negative excursions) and, if the excursions are truly
   one-sided, fit the positive tail separately (the method takes any f(eps); only the estimation assumes symmetry).
2. Use the per-skip data (or a skip-subset consistency check) to veto single-pixel excursions - the remaining fake
   class in `art`.
3. Feed the finder's pooled lam and its no-1-e- rule into A_GMX's counting finder (or replace it below ~20 e-) and
   keep A_GMX for big deposits; the finder already flags groups > 20 e- for that hand-off.
4. A realistic event rate: rerun the benchmark with ~1 event per image (R ~ 2e-6); the BF ranking is rate-free, but
   the posterior and its fake prediction depend on R and the 30-events-per-image benchmark inflates it.
5. Speed at s = 0.15: reuse pass-1 evidences where lam is unchanged, tabulate pair neighbourhoods like the single
   pixels (exact closed form already used for one-pixel neighbourhoods).
6. Model check on real data with more images: hot-pixel persistence needs a run with many images of the same
   setting; the 2-image NDCM groups give little power.

## 11. Suggested improvements to the Bayesian method (ranked by expected gain / effort)

| # | improvement | where it is weak now (measured) | expected gain | effort |
|---|---|---|---|---|
| 1 | **Artefact vetoes**: per-skip consistency of a candidate pixel (charge present in all skips vs one sample), true persistence over many images, and an excursion component with the measured (flat 1-10 e-) shape instead of a wide Gaussian | `art`: 3 fakes in 9.2 Mpix at the operating point (0.33 / Mpix, analytic prediction 3x low); single-pixel 4-9 e- events at sigma < 0.3 are pushed to log J1 8-10 by the excursion component (4-10 e- efficiency 0.95 vs A_GMX 0.98) | fakes -> ~0.02 / Mpix; single-pixel events back to ~1.0; analytic prediction calibrated again | medium (needs per-skip data in the pipeline) |
| 2 | **Honest operating point as the default**: always select by the posterior expected-fake budget (section 1a) with R from a run-level fit and a prior on R, report the off-image null next to it; never quote 'above the highest fake' | section 1b's matched-rate numbers are optimistic when the budget is < 1 fake; P(event) depends on R, which the 30-events-per-image benchmark inflates | correct fake rates at realistic R (efficiency at fixed log J1 is rate-free, so no loss) | low |
| 3 | **Non-parametric noise tail**: replace the 3-component Gaussian mixture by a smoothed histogram / spline of the folded negative side pooled over the run (tail shape, not only width) | tail s0.15 l1e-5: pooled lam 1.9x HIGH (2.8e-5 vs 1.5e-5); single-image fallback 0.2-0.4x LOW there; 2-e- efficiency 0.14 at nominal fakes | lam within ~20 % (the tail at lam 1e-4 is already unbiased), a few % on 2 e- | low-medium |
| 4 | **Rate-aware 2-e- reporting at high lam**: at lam >= ~3e-4 report only N >= 3 per event and 2-e- events only as a population rate (EM) | lam 1e-3: 2 e- 0.03 at 0.04 fakes / Mpix; s0.15 l1e-4: 0.72 at 0.13 / Mpix - pred = meas, i.e. the information limit (chance dark pairs), not a model error | no efficiency gain possible; avoids a catalogue dominated by chance pairs | low |
| 5 | **Fit the diffusion law from the event population** (A, b of sigma^2 = -A ln(1 - bz) by EM over the per-event sigma posteriors, like R_N) | physics prior saturates at 1.01 px: posterior sigma 0.87 vs 0.97 (bench prior) for true 1.0-1.2 px; depth 550 vs 670 um. Detection insensitive (efficiencies within 0.02 between priors) | unbiased sigma / depth at the population level | low-medium |
| 6 | **Speed at s = 0.15**: reuse pass-1 evidences where lam changed < 1 %, closed form for two-pixel neighbourhoods (one-pixel ones already tabulated), skip the 2-event term for weak groups | 1.4-1.9 s per 0.6 Mpix (A_GMX 3.9 s); 6-9 s with the noise tail (7000 candidates / image) | ~2x | low |
| 7 | **Low-frequency noise on real data**: along-row baseline (running trimmed mean, ~50 px) as a nuisance, offset uncertainty folded into the noise width, residual autocorrelation check per row; one-sided tail option for EXT4 | L2: sigma x sqrt(N) grows with N; EXT4 asymmetric (z = -13 .. -22); the current per-row offset cannot follow a drift inside a row | fewer half-electron offsets in drifting rows; EXT4 usable instead of merely conservative | low (offsets) / medium (one-sided tail) |
| 8 | **Gain**: joint fit of g with lam using all candidates of a run, plus an external calibration image | profile likelihood flat (Delta lnL < 0.5 over 0.7-2.2 ADU / e-) on the noise-test images | only with data containing hundreds of 1-e- pixels | low code, needs data |
| 9 | **Population N-electron rate**: posterior uncertainty on R_N (bootstrap / Gamma posterior), joint R_N x sigma spectrum, explicit R_1 + lam degeneracy (one parameter, reported as such) | R (N >= 2) already within 1 % of the truth in every group; no uncertainty reported | error bars for rate spectra; sub-threshold events used | low |
| 10 | **Multi-event neighbourhoods**: simulate close event pairs (< 8 px) to test the bipartition approximation; extend to >= 3 events by birth-death MCMC if needed | untested (benchmark events >= 20 px apart) | unknown; relevant for real data with clustered activity | medium |

## 12. When to hand an image / region to the Bayesian finder (switching criterion)

**Quantities, all measurable before clustering in ~0.2 s per image:**
- s: core read noise from the negative side of the zero peak (`noise.fit_negative`), pooled over the run setting;
- f_c: the soft-candidate fraction, pixels with max_{n>=1} f(v - n) / f(v) > 1e-2 (`noise.candidates`). It
  combines read noise, its tail and lam in one number and directly measures sparsity: the mean number of other
  candidates within the 4-px linking radius is ~50 f_c;
- lam: pooled dark charge (`background.fit`, or the single-image fallback);
- big deposits: candidate groups above 20 e- (already flagged by the finder) and A_GMX's core fill.

Measured f_c (per pixel): s 0.05-0.10, lam <= 1e-4: 5e-5 - 3e-4; s 0.15: 4.4e-3 (lam <= 1e-4), 5.4e-3 (lam 1e-3);
s 0.10 + tail: 2.3e-3; s 0.15 + tail: 1.1-1.2e-2. For Gaussian noise f_c ~ P(eps > 0.5 - 4.6 s^2) + lam:
s = 0.15 -> 4e-3, 0.16 -> 8e-3, 0.17 -> 1.5e-2.

**Rule (per run setting for s and lam, refined per region for f_c and big deposits):**

| condition | finder for groups <= 20 e- | evidence on both sides of the cut |
|---|---|---|
| f_c <= 6e-3 (s <~ 0.155 Gaussian, lam <= 1e-3) | **Bayes** (A_GMX for big deposits only) | inside: Bayes >= A_GMX in every tested group, 2 e- +0.2..+0.8 (e.g. s0.15 l1e-5 0.97 vs 0.20; L2 s0.06-0.10 0.96-0.99 vs 0.51-0.64), 3 e- +0.03..+0.3, fakes at or below nominal (A_GMX 10x above on L2), energies 99 % exact vs 67-93 %, 2-10x faster |
| 6e-3 < f_c <= 1.5e-2 (noise tail, s ~ 0.16-0.17) | **Bayes in addition to A_GMX** (Bayes catalogue, A_GMX as cross-check) if 5-10 s per 0.6 Mpix is acceptable | measured only for s0.15 + tail (f_c 1.1e-2): Bayes still better (lam 1e-4: 2 e- 0.14 vs 0.04, 3 e- 0.87 vs 0.51, fakes 0.22 vs 0.11 / Mpix; s0.10 + tail: 0.61 vs 0.29 at 2 e-) but 2x slower than A_GMX and lam at 1e-5 is 1.9x high |
| f_c > 1.5e-2 (s >= ~0.17, lam >~ 1e-2, glow, crowded core fill >= 0.10) or A_GMX regime 'unmodelled' | **A_GMX only** | neighbourhoods start to percolate (> 0.75 other candidates per linking disk); not benchmarked here (s 0.20 / 0.25, lam 1e-2 were out of scope) - the cut is a sparsity argument, not a measured crossover |
| lam >= 3e-4 (any s) | Bayes, but catalogue N >= 3; 2-e- events only as a population rate | expected chance dark pairs ~ lam^2 x (pixel pairs within a neighbourhood): measured 2-e- fakes 0.13 / Mpix at lam 1e-4 with eff 0.72, eff 0.03 at 0.04 / Mpix at lam 1e-3; 3 e- stays 0.91-0.99. The cut 3e-4 is interpolated between the two measured points |
| groups > 20 e- (Fe-55, muons, glow) | **A_GMX** (crowded mode if its core fill >= 0.10); Bayes skips a 15-px halo around them | busy (3 Fe-55 + 1 muon per image): combined = Bayes small events 2 / 3 / 4 e- 0.71 / 0.98 / 1.00 (log J1 >= 12, 0 fakes in 9.2 Mpix) vs A_GMX alone 0.24 / 0.72 / 0.96 (0.11 fakes / Mpix); A_GMX finds 60 / 60 big deposits. The simulation keeps events >= 60 px (Fe-55) / 120 px (muons) from big deposits, so the halo hand-over itself is untested |

**Combination on one image.** (1) Noise model and candidates (cheap) -> f_c per 100 x 100 block and per image;
(2) candidate groups > 20 e- -> A_GMX (big deposits, tracks, crowded mode), with a 15-px halo mask; (3) blocks with
f_c <= 6e-3 outside the halo -> Bayesian finder for every group <= 20 e-, using the run-pooled lam and R_N;
(4) blocks above 1.5e-2 -> A_GMX's counting finder; in between -> both, Bayes catalogue. A_GMX's counting finder is
then not needed in the sparse regime; its few-electron output there is strictly worse on every benchmark number.
`art` (first column, excursions, transient hot pixels): Bayes for the small groups gives 2 / 3 e- 0.77 / 0.95 with 0
fakes at log J1 >= 12 vs A_GMX 0.42 / 0.72 with 7.1 fakes / Mpix (< 20 e-; plus 32 unmatched objects >= 20 e-); A_GMX remains better only for single-pixel 4-10 e- deposits (0.98 vs 0.95, item 1 of
section 11).

**Granularity.** s, the noise shape and lam per run setting (pooled over its images: lam per 0.6-Mpix image at 1e-5
is +-40 %, pooled < 10 %); f_c and big deposits per image and per region (they change locally: glow, hot regions,
tracks); the hot-pixel / hot-column model per run (persistence needs many images). A per-image switch on s alone
would be noisy and miss local glow; a per-run switch alone would miss tracks and glow.

## Caveats

- The simulation and the model share their assumptions (Poisson dark charge, Gaussian lateral diffusion,
  pixel-integrated, symmetric noise): part of the gain over A_GMX is 'home advantage'; the tail / art variants and
  the real L2 data (EXT4) are the only tests of misspecification.
- The fitted event rate is used in P(event); the benchmark has 30 events per 0.6 Mpix image, far more than real
  data, which makes P(event) larger than it would be in a real run at the same Bayes factor. Thresholds on log J1
  (section 1b) are rate-free.
- Fake counts are small (0-7 at the operating points); rates below ~0.02 / Mpix are upper limits. The A_GMX ROC
  points at count_fakes 0.01 and 1 exist only for the key configurations (base / blank s0.06, s0.10, s0.15 at lam
  1e-5 and s0.15 at 1e-4, L2 s0.06 / s0.10).
- Machine-load: the runs were done with 2 worker processes (plus one single-process run at a time); other jobs of the
  user were running concurrently, so the times are upper bounds.
- Diffusion law parameters: A = 285 um^2, b = 8.2e-4 / um from DAMIC at SNOLAB (PRL 125, 241803 (2020),
  arXiv:2007.15622, surface-muon calibration of its 675-um CCDs); the functional form from DAMIC, PRD 94, 082006
  (2016), arXiv:1607.07410. I could not find published DAMIC-M (670 um, L2 module) values; they are parameters.

## 13. v2 (2026-11-11): artefact vetoes, non-parametric noise tail, ceilings

Everything below is OPTIONAL (defaults unchanged: a default run reproduces v1 exactly - checked: results/v2/before
equals results/physics on every compared group, and a default-vs-v1 regression on 7 images agrees to 1e-14 absolute).
API additions: README 'v2 additions'. Re-benchmark (same seeds, physics prior, 2 shuffles): `results/v2/before`
(defaults) vs `results/v2/after` (`--point 1 --noise empirical --hot-rows 1 --blink 1`), full table
`results/v2/compare.md` (`compare.py`). Not changed: agmx_switch.py (not edited; it calls only default paths).

### 13.1 Artefacts (A1)

**Diagnosis of the v1 `art` fakes** (truth-tagged, `art` at 1 fake / Mpix, 28 fakes): 18 transient 'hot pixels' (the
simulation draws 3 new ones per image, integer Poisson charge 0.1-3 e- mean - indistinguishable from a compact
event pixel by pixel), 9 positive excursions with integer-like values, 1 other; all single pixels. The excursion
density itself was already right to ~x2 (the broad component, w 1.2e-5 at 5.7 e-, vs the true flat 1-10 e-).

**Point-deposit component** (`point_defects=True`): N >= 2 e- in one pixel without diffusion, rate spectrum R_pt(N)
learned by the population EM, kept only if it raises the population log-likelihood by > 5 (model selection; without
it R_pt absorbed compact real events in l2_s0.06 and cut 2-e- efficiency 0.99 -> 0.90 - first attempt, discarded).
dlnL = 7.1 in `art`; 0.0-1.4 in every other group (R_pt = 0 there, results identical to before).

| art s0.10 l1e-5 (9.2 Mpix) | eff 2/3/4 e- @0.08 | fakes meas / pred @0.08 | eff @1 | fakes meas / pred / shuffle-null @1 |
|---|---|---|---|---|
| v1 (before) | 0.79/0.95/0.95 | 3 / 0.7 | 1.00/0.98/0.98 | 28 / 8.9 / 1.0 |
| v2 after | 0.79/0.95/0.95 | 1 / 0.7 | 0.83/0.98/0.95 | 12 / 8.7 / 6.5 |

Remaining 12 at F = 1: 6 excursions, 5 transient hot pixels, 1 other. The prediction is now within ~1.5 sigma; the
price is that compact single-pixel 2-e- events become ambiguous (P(event) ~ 0.5) when transient point defects exist
(2-e- at F = 1: 1.00 -> 0.83); at the 0.08 operating point nothing changes except the fakes (3 -> 1).

**Excursions / asymmetry**: the excursion hypothesis is the noise density (v = n + eps with eps from the tail,
continuous, competing with every n). New: positive-side scales c_tail (e_safe..1 e-) and c_exc (> 1 e-) fitted on the
positive pixel values with a free integer-comb occupation (`noise.fit_excursion_scale`), used when the core symmetry
test (|z| > 4) or the scales' log BF (> 3) demand it. On the simulation the auto mode chose symmetric in all 19
groups (correct). A first version included pixels above 12 e- and declared Fe-55 / muon pixels positive excursions
in `busy` (c_exc = 100, 2-e- 0.73 -> 0.49): fixed by fitting only v <= 11.5 e-; final `busy` identical to before.
**Ringing**: real L2 EXT4 negative excursions (146 pixels < -3 e-, NDCM >= 500) have median neighbours 0.00 e- (mean
<= 0.13 e- at x+1): no serial-direction signature, so no ringing component was built.

**Persistence** (`hot_rows`, `blink`, `mask_cols`): local test config `persist_s0.10_l1e-05` (20 persistent Poisson
hot pixels 0.02-1 e-/image, 5 blinking pixels with a 2-8 e- burst in 20 % of the images, one hot row 2e-3 e-/pix).
Hot row found (P = 1, row 200); 17-19 / 20 hot pixels found (v1 component); blinking pixels: those firing >= 3 times
are vetoed (1 of 5 here), those firing twice are described as Poisson hot pixels (the v1 behaviour, which already
suppressed them). Fakes 3 / 12 (0.08 / 1 per Mpix) before and after: no gain on this config - all remaining fakes
are blinking pixels that fired 1-2 times in 15 images, which no persistence test can separate from events. False
vetoes in the event-dense benchmark groups: 0-7 pixels per group (after requiring >= 3 firings; 9-21 before that).
Column-0 masking (`mask_cols=(0,)`) is an option; not benchmarked (no `art` fake was in column 0).

**Skip-resolved test** (`skips.py`, `results/v2/skip_study.md`): constant level vs spike / step, exact Gaussian
marginals. Simulated skip series, sigma1 = 3.3 e- per sample, threshold at 1e-3 false vetoes of real-charge pixels:
- white noise: a glitch in one sample (spike) and charge present only in the first skips (column-0-like): 100 %
  vetoed at a = 1-5 e- in the mean, M = 100-3000; a level change at a random skip: 39 / 71 / 94 % (a = 1 / 2 / 5 e-,
  M = 100), 85 / 96 / 99 % (M = 1000), 94 / 99 / 100 % (M = 3000).
- with 1/f noise (rms 0.5-1 sigma1) a white-noise model vetoes 23-100 % of REAL charge (unusable); a polynomial
  drift does not fix it; the 1/f covariance model (`cov=skips.cov_1f(...)`, in practice measured from empty pixels)
  restores the false-veto rate (threshold logBF 0.7-1.8) with spikes / first-skip 97-100 % and steps 14-99 %.
- Projection for `art` (not measured on per-skip data - the simulation has none): the 6 remaining excursion fakes at
  F = 1 (9 in v1) would be removed if excursions are glitches or first-skip charge, ~85-99 % of them if they are
  random level changes (white noise, M >= 1000); the 5 transient hot pixels are real charge and are not affected.
  So F = 1 fakes 12 -> ~6, i.e. the skip test can halve them; the per-skip reader is needed to confirm.

### 13.2 Non-parametric noise tail (A2)

`noise.fit_empirical`: semi-parametric density = scale mixture x exp(B-spline correction in asinh(|e|/s)), smoothness
chosen by the Laplace evidence, posterior sd per bin (a Dirichlet-histogram variant `method='dp'` was noisier, 4-10 %
in the tail, and is not used). Results:
- tail_* (simulated tail is exactly a 2-Gaussian mixture, i.e. the v1 family): no change - eff 0.14/0.87/0.98
  (tail s0.15 l1e-4 @0.08) and lam 1.92x (1.89x before) at s0.15 l1e-5. On a noise-only test the mixture's tail is
  already within 2 % of the truth up to 1.1 e-; the 1.9x lam bias at 1e-5 needs a tail density known to ~1 % near
  1 e- (tail pixels near 1 e- outnumber the 6 dark electrons per image ~20:1), at the level of the fit's statistical
  error with 15 images. Not fixed. (No oracle-noise lam run was made to prove this.)
- art: excursion density closer to the true flat 1-10 e- (log f at 8 e- -14.93 vs -15.10, true -14.40); no
  measurable change in fakes beyond the point-deposit effect.
- Real L2 (`results/real_l2_v2_before|after`): EXT1-3 symmetric (auto chose symmetric). EXT4 asymmetric: fitted
  positive/negative probability beyond 1.5 e- 0.02-0.3 (e.g. NDCM 3000: 8.4e-6 vs 4.8e-4; beyond 5 e- 1.2e-7 vs
  7.1e-6), c_exc 0.01-0.3. Consequences: EXT4 candidates with log J1 > 0: 12 -> 18, top log J1 6.6 -> 9.2, top P(event)
  2e-5 -> 4e-4 (R fitted 2.8e-8); EXT4 lam 0.5-2.5e-5 per NDCM group (was 0.5-0.6e-5). Off-image null in EXT4: 4-8 per 4
  permutations above log J1 5 vs 1-2 in the data. Still no significant few-electron event. Whether EXT4 is now
  'correct' cannot be checked without truth; the model is no longer conservative-by-construction there.
- runtime: + ~5 s per group for the asymmetry decision (9 x 9 scale grid on <= 1e4 pixels).

### 13.3 Ceilings and population sensitivity (A3)

**Finder vs oracle on the same images** (`ceiling.py oracle`; oracle = true lam map + 1-e- density, true Gaussian
noise, true sigma U(0.2, 1.2), true N spectrum; `results/v2/oracle_*.csv`):

| group | variant | 2/3/4 e- at <= 4 measured fakes (0.08 / Mpix) | 2/3/4 e- at analytic op. point (meas. fakes) |
|---|---|---|---|
| base s0.15 l1e-4 (55 Mpix) | finder | 0.43/0.99/1.00 | 0.72/0.99/1.00 (7) |
| | true background, finder prior | 0.46/0.99/1.00 | 0.60/0.99/1.00 (6) |
| | oracle | 0.46/0.99/1.00 | 0.73/0.99/1.00 (8) |
| base s0.15 l1e-3 | finder | 0.03/0.98/1.00 | 0.03/0.91/0.99 (2) |
| | true background, finder prior | 0.03/0.98/1.00 | 0.01/0.88/0.98 (1) |
| | oracle | 0.03/0.94/1.00 | 0.03/0.93/0.99 (3) |

Gap split: background estimation ~0.03, prior ~0, i.e. **the finder is at the oracle ceiling** within the
statistics (4 allowed fakes; +-0.05). The remaining loss is fundamental. Analytic noiseless limit (Neyman-Pearson on
exact dark-charge patterns, `results/v2/pairs.md`): 2 e- at 0.08 / Mpix: 1.00 (lam <= 3e-5), 0.84 (1e-4), 0.45 (2e-4),
0.26 (3e-4), 0.13 (5e-4), 0.03 (1e-3); 3 e-: 1.00 up to 3e-4, 0.88 at 1e-3, 0.33 at 3e-3. At lam 1e-4 the read noise
s = 0.15 costs 0.84 -> ~0.5-0.7 (2 e-); at 1e-3 the finder equals the noiseless limit (0.03), 3 e- 0.91 vs 0.88
(consistent within errors).

**Oracle (s, lam) grid** (`results/v2/grid_small.md`, only 28 Mpix of null per point = 2 allowed fakes at 0.08 /
Mpix, so +-0.2 scatter; a larger grid was set up but NOT run): 2 e- at 0.08 / Mpix ~1.0 at lam 1e-5 for s 0.06-0.15;
0.2-0.9 at 1e-4; 0.02-0.2 at 3e-4; ~0 at 1e-3; 3 e- >= 0.98 up to 3e-4 and 0.81-0.87 at 1e-3.

**Population R_N** (`ceiling.py pop`, `results/v2/pop_*`; physics prior, fitted background, bootstrap of blank
neighbourhoods + Poisson-injected matched event neighbourhoods, 40 replicas):
- no events (60 blanks, 37 Mpix): R2 = 0.089 / Mpix at lam 1e-4 (positive, ~1.5 sd of the estimator: boundary
  effect of a rate fitted at zero) and 4e-4 / Mpix at lam 1e-3; R3 = 0 in both.
- lam 1e-4, sd of the 2-e- rate per Mpix (EM vs counting at 0.08 fakes / Mpix with the fake rate subtracted): E = 10
  Mpix-images: 0.18-0.31 vs 0.28-0.54 (counting 1.2-1.8x worse); E = 100: 0.05-0.12 vs 0.08-0.16 (1.4-1.6x). Bias of
  the EM +0.06-0.08 at rho = 0.1, -0.08..-0.14 at rho = 1. 3 e-: both equal (ratio 1.0-1.2), unbiased.
- lam 1e-3: 2 e- counting is useless (eff 0.03; sd 0.7-2.9 / Mpix), the EM has sd 0.03-0.54 but is biased LOW by
  ~35 % at rho = 1 (0.66 vs 1.0) and finds ~0 at rho = 0.1 - not understood (possibly incomplete EM convergence at
  200 iterations, or the flat N-spectrum prior); 3 e- EM and counting equal and unbiased within 5-10 %.

**Recommendations (numbers from the analytic limit and the oracle runs above; 2 e-, 0.08 fakes / Mpix):**
- efficiency 0.9 needs lam <~ 8e-5 e-/pix/image (noiseless), and s <= ~0.12 to stay near it (s = 0.15 at 1e-4: 0.72
  at the operating point vs 0.84 noiseless); 0.5 needs lam <~ 1.8e-4 (noiseless) / ~1.2e-4 at s = 0.15. Above ~3e-4
  2-e- events are only a population quantity (use the EM rate, with the caveat above).
- skips: s ~ 3.3 e- / sqrt(M); dark charge picked up during readout grows with M. Going from s 0.15 to 0.10 (M x2.25)
  is worth it only while lam stays <~ 1e-4: the 2-e- efficiency drops much faster with lam (x2 lam at 1e-4: 0.84 ->
  0.45) than it gains with s.
- binning 2x2 does not help 2-e- events at fixed lam per pixel (noiseless limit 0.84 -> 0.72 at 1e-4, 0.03 -> 0.017
  at 1e-3): chance pairs scale with lam^2 per binned pixel.
- shorter readout per image (smaller images / fewer rows per readout) lowers the readout-accumulated lam, the
  dominant lever for 2 e-; image size itself does not change fake rates per Mpix.

### 13.4 Re-benchmark summary (v2 after vs before; `results/v2/compare.md`)

Identical efficiencies (2/3/4 e- at 0.08 and at 1 / Mpix) and fakes in base s0.06 / s0.10 / s0.15 (l1e-5, 1e-4,
1e-3), hotcol_* (4 groups), l2_* (3), busy, tail_* (4) - within +-0.02 (l2_s0.15 2 e- 0.78 -> 0.80); lam ratios
unchanged (tail s0.15 l1e-5 still 1.9x); runtime per image -10 % to +5 %. Changes only in `art` (above) and
`persist` (no change in fakes). Off-image null per permutation within the Poisson scatter of before.

### 13.5 Status, defaults, next steps

- Defaults unchanged (all v2 behaviour OFF). Recommended to switch on: `point_defects=True` (only active when the
  data demand it; improved art, neutral elsewhere) and `noise='empirical'` for real data with asymmetric noise (EXT4).
  `hot_rows` / `blink`: neutral on the benchmark, useful only for runs with many images; keep optional.
- Untested / unfinished: skip test on real per-skip data (needs the reader); column-0 masking benchmark; the larger
  oracle grid (code ready: `ceiling.py grid`, 200 blanks per point); EM bias at lam 1e-3 and the positive R2 on blanks
  at 1e-4; an oracle-noise lam run for the tail bias; the population study uses 40 bootstrap replicas only.
- Next: per-skip reader -> `skips.SkipModel` with a measured 1/f covariance on candidate pixels (expected to halve
  the art fakes); EM convergence / prior check of R_2 at high lam; run `ceiling.py grid` for the (s, lam) map.
