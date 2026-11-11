# A_GMX few-electron finding at ultra-low dark current: simulation benchmark (2026-11-10)

Folder `wip/lowdc/` (self-contained; nothing outside it was changed). The full generated tables are in
`results/tables.md` (from `summarize.py`), the analytic limits in `results/analytic.md`; raw records are in
`results/images.csv` (one row per image), `events.csv` (one row per injected deposit), `fakes.csv` (one row per
unmatched A_GMX cluster or fe_candidates candidate), `proto_box_*.csv` (prototype).

| file | what |
|---|---|
| `sim.py` | image generator in electrons: Poisson dark charge (uniform / row gradient), serial-register charge (incl. overscan), hot columns / pixels, Gaussian read noise with optional tail (`mix`: fraction of pixels with k x s; `t`: Student-t), rare large excursions of both signs (`f_out`), row-pedestal residual, first-column artefact (`first_col`, L2: ~2000/NDCM ADU), events of 1-10 e- with diffusion sigma 0.2-1.2 px, Fe-55 (1600 e-) and muon tracks; truth = per-deposit pixel lists |
| `metrics.py` | matching (pixel overlap weighted by true electrons, fallback centroid <= 2 px; objects within 40 px of a big deposit counted apart), efficiency / rate errors, markdown tables |
| `bench_agmx.py` | runs `background_map -> adaptive_cluster(engine='gmx', return_pixels=True) -> fe_candidates.candidates` exactly as `src/agmx_root.process` (noise < 0.35), plus diagnostics (`detection_bg` both modes, `count_threshold`); fixed seed per (config, image); resumable; `--procs` workers, each single-threaded |
| `summarize.py` | the tables |
| `analytic.py` | per-pixel null rates and the smallest detectable window charge for a given fake rate |
| `proto_box.py` | QUICK PROTOTYPE only (to support the ranking): window-count finder with an exact analytic null |

Setup: 480 x 1280 active pixels (+64 overscan, masked), 30 events per image (n_e uniform 1..10, sigma uniform 0.2..1.2
px, >= 20 px apart), 15-30 images per configuration (45-90 events per n_e), plus 60 event-free images for the main
blank configurations, and the L2 ACM 103 geometry (100 x 500, 4 events per image, 200 images per configuration).
Fake area: all images (events cover < 1 % of the area); "plain" = base + blank configurations pooled.
Variants: `tail` = 0.5 % of pixels with 3 x the read noise (symmetric); `hotcol` = one column with +0.02 e-/pix;
`art` = first column 1.5 +- 0.3 e-, 1e-5 pixels with +-1..10 e- excursions, serial-register 2e-5 e-/pix, 3 hot
pixels (0.1-3 e-), row pedestal 0.01 e-; `busy` = + 3 Fe-55 and 1 muon per image; `grad` = dark charge rising x3 along
the rows. 1485 images in total; ~32 min wall time (2 then 4 worker processes).

## 1. Efficiency vs n_e (A_GMX, fixed nominal rate 0.08 fakes/Mpix)

Selected rows (all configurations: `results/tables.md`). Errors are binomial.

| config (s [e-], lam [e-/pix]) | 1e | 2e | 3e | 4e | 5e | 6e | >=7e |
|---|---|---|---|---|---|---|---|
| s0.05 l1e-5 | 0.00 | 0.61±0.07 | 0.93±0.04 | 1.00 | 1.00 | 1.00 | 1.00 |
| s0.06 l3e-6 | 0.00 | 0.73±0.06 | 0.96±0.03 | 1.00 | 1.00 | 1.00 | 1.00 |
| s0.06 l1e-5 | 0.00 | 0.56±0.08 | 0.98±0.02 | 0.98 | 1.00 | 1.00 | 1.00 |
| s0.06 l1e-4 | 0.00 | 0.61±0.09 | 0.96±0.03 | 0.98 | 1.00 | 1.00 | 1.00 |
| s0.10 l1e-5 | 0.02 | 0.62±0.08 | 0.90±0.04 | 0.98 | 1.00 | 1.00 | 1.00 |
| s0.10 l1e-4 | 0.00 | 0.40±0.07 | 0.80±0.06 | 0.98 | 1.00 | 1.00 | 1.00 |
| s0.15 l3e-6 | 0.00 | 0.24±0.06 | 0.81±0.05 | 0.93 | 1.00 | 1.00 | 1.00 |
| s0.15 l1e-5 | 0.00 | 0.20±0.04 | 0.80±0.04 | 0.93 | 0.98 | 1.00 | 1.00 |
| s0.15 l1e-4 | 0.00 | 0.12±0.03 | 0.82±0.04 | 0.94 | 0.99 | 1.00 | 1.00 |
| s0.15 l1e-3 | 0.00 | 0.00 | 0.65±0.05 | 0.85 | 1.00 | 0.99 | 1.00 |
| s0.15 l1e-2 | 0.00 | 0.00 | 0.01 | 0.38±0.05 | 0.69 | 0.90 | 0.97-1.00 |
| s0.20 l1e-5 | 0.00 | 0.07 | 0.28±0.07 | 0.76 | 0.93 | 0.98 | 1.00 |
| s0.25 l1e-5 | 0.00 | 0.17 | 0.05±0.04 | 0.66 | 0.77 | 0.95 | 0.96-1.00 |
| tail s0.15 l1e-5 | 0.00 | 0.04 | 0.65±0.07 | 0.91 | 0.95 | 1.00 | 1.00 |
| tail s0.15 l1e-4 | 0.00 | 0.04 | 0.51±0.08 | 0.82 | 0.98 | 1.00 | 1.00 |
| tail s0.20 l1e-4 | 0.00 | 0.00 | 0.19 | 0.55 | 0.88 | 0.92 | 1.00 |
| hotcol s0.15 l1e-5 | 0.00 | 0.23 | 0.70±0.07 | 0.96 | 1.00 | 1.00 | 1.00 |
| art s0.10 l1e-5 | 0.00 | 0.42 | 0.72±0.07 | 0.98 | 0.98 | 1.00 | 1.00 |
| busy s0.15 l1e-4 (Fe-55 + muon) | 0.02 | 0.24 | 0.72±0.07 | 0.96 | 0.98 | 0.98 | 1.00 |
| L2 100x500 s0.06 l1e-5 | 0.05 | 0.64±0.06 | 0.90±0.03 | 1.00 | 1.00 | 1.00 | 1.00 |
| L2 100x500 s0.10 l1e-5 | 0.13 | 0.51±0.06 | 0.85±0.04 | 0.99 | 0.99 | 1.00 | 1.00 |

(The non-zero 1-e- "efficiencies" in the L2 rows are dark-electron fakes landing next to the event, see section 3.)

Efficiency vs diffusion (A_GMX, s0.10 l1e-5): 2 e- 0.64 / 0.88 / 0.50 for sigma 0.2-0.5 / 0.5-0.8 / 0.8-1.2 px;
3 e- 1.00 / 0.90 / 0.81. At s0.15 l1e-4: 3 e- 0.97 / 0.86 / 0.65, 4 e- 1.00 / 1.00 / 0.84. 2-e- events whose
electrons share one pixel are always found at s <= 0.10 (eff 1.00), at s0.15 l1e-4 only 0.53.

**2-e- events by topology** (A_GMX found / fe_candidates any candidate / prototype):

| config | 2 e- touching | 2 e- not touching | 3 e- touching | 3 e- not touching |
|---|---|---|---|---|
| s0.06 l1e-5 | 0.63 / 1.00 / 1.00 (n=30) | 0.36 / 0.00 / 0.64 (n=11) | 1.00 / 1.00 / 1.00 | 0.90 / 0.70 / 1.00 |
| s0.10 l1e-5 | 0.74 / 1.00 / 1.00 (n=27) | 0.38 / 0.00 / 0.92 (n=13) | 0.97 / 1.00 / 1.00 | 0.75 / 0.88 / 1.00 |
| s0.15 l1e-5 | 0.27 / 1.00 / 0.17 (n=64) | 0.00 / 0.00 / 0.00 (n=23) | 0.96 / 1.00 / 1.00 | 0.47 / 0.81 / 0.50 |
| s0.15 l1e-4 | 0.18 / 1.00 / 0.00 (n=57) | 0.03 / 0.00 / 0.00 (n=33) | 0.99 / 1.00 / 1.00 | 0.41 / 0.89 / 0.59 |

## 2. Charge response of found events (A_GMX, base configurations pooled)

| n_e | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| mean measured [e-] | 2.01 | 2.89 | 3.76 | 4.74 | 5.64 | 6.58 | 7.55 | 8.54 | 9.60 |
| exact (\|q - n_e\| < 0.5) | 0.93 | 0.85 | 0.75 | 0.75 | 0.70 | 0.67 | 0.69 | 0.67 | 0.70 |
| low by >= 1 e- | 0.04 | 0.13 | 0.22 | 0.21 | 0.26 | 0.30 | 0.28 | 0.29 | 0.27 |
| split into >= 2 clusters | 0 | 0.003 | 0.010 | 0.027 | 0.046 | 0.069 | 0.067 | 0.075 | 0.090 |

Every few-electron cluster is cls `low-E` (counting energies). 25-30 % of 4-10 e- events lose about one electron:
the counting energy sums only the counted pixels of the group / the pixels the mixture owns with responsibility >
0.5 (gmx.py:850-853, `_counting_pix` l.1104ff); an electron that diffused into a non-touching pixel is left out
(and either lost or a separate group). Diffuse events split in up to 9 % of cases at 10 e-.

## 3. Fakes per Mpix per image (unmatched, < 20 e-, not near a big deposit)

| group | Mpix | A_GMX fakes | A_GMX / Mpix | thr_T | fe_cand clean obs / exp per Mpix |
|---|---|---|---|---|---|
| plain s0.06 l1e-5 | 46 | 0 | < 0.022 (1 sigma) | 14.5±0.4 | 0.00 / 0.06 |
| plain s0.10 l1e-5 | 46 | 0 | < 0.022 | 14.6±0.5 | 0.00 / 0.19 |
| plain s0.15 l1e-5 | 55 | 3 | 0.054±0.031 | 14.9±0.6 | 0.72 / 1.86 |
| plain s0.15 l1e-4 | 55 | 6 | 0.109±0.044 | 15.0±0.7 | 0.87 / 2.16 |
| plain s0.15 l1e-3 | 55 | 4 | 0.072±0.036 | 15.6±0.9 | 8.45 / 7.82 |
| plain s0.15 l1e-2 | 55 | 5 | 0.090±0.040 | 16.6±0.9 | 434 / 440 |
| tail s0.15 l1e-4 | 9.2 | 1 | 0.11±0.11 | 15.5±0.9 | 9.66 / 5.60 |
| hotcol s0.15 l1e-5 | 9.2 | 3 (all in the hot column) | 0.33±0.19 | 14.7±0.4 | 0.87 / 1.71 |
| art s0.10 l1e-5 | 9.2 | 65 | 7.1±0.9 | 14.8±0.5 | 0.43 / 0.45 |
| **L2 100x500 s0.06 l1e-5** | 10 | **9 (all single 1-e- pixels)** | **0.90±0.30** | 14.1±2.2 | 0.00 / 0.49 |
| **L2 100x500 s0.10 l1e-5** | 10 | **7 (all single 1-e- pixels)** | **0.70±0.26** | 14.5±2.6 | 0.00 / 0.84 |
| L2 100x500 s0.15 l1e-4 | 10 | 3 | 0.30±0.17 | 16.0±2.1 | 1.40 / 2.98 |

- On 480 x 1280 images with Gaussian noise the fake rate is at the nominal 0.08/Mpix for s = 0.15 (0.05-0.11) and
  far below it for s <= 0.10 (0 in 46 Mpix): the finder is conservative exactly where 2-e- events are detectable.
- **L2 geometry: A_GMX reports single dark electrons (charge 1, one pixel) as events**, 0.7-0.9 per Mpix
  (~1 per 25 images), 10x nominal. Cause: (a) `detection_bg` measures lam from the isolated 1-e- pixels in rows
  r +- 30 (gmx.py:130ff); at 1e-5 e-/pix a 61 x 500 strip holds ~0.3 dark electrons, so lam_q = 0 in many rows and is
  floored at 1e-6 (gmx.py:169); (b) `count_threshold` (gmx.py:1079ff) on a 0.05 Mpix image has F = 0.004 expected
  fakes, n_exceed = floor(F x 50) = 0, so the threshold is the single highest T of 50 blanks simulated at that lam:
  median thr_T = 13.8155 = -ln(1e-6), i.e. exactly a 1-e- pixel at the floor. A real dark electron in such a row has
  T >= thr (sig 0.00-3.5 in fakes.csv) and is kept. The threshold scatters by +-2.2 in T (x9 in fake rate) between
  images. Nothing in the code forbids a 1-electron group.
- `art`: first column -> 35 objects in 15 images (20 cls `artifact-column`, 15 cls `low-E` not flagged); the
  +-1..10 e- excursions and hot pixels -> ~4 fake few-electron events per image (q 2-6 e-, 1 px). Expected (no
  shape information in one pixel) but they must be masked upstream (hot pixels persist across images; outliers can be
  vetoed with the per-skip data or a multi-image comparison).
- `hotcol` (0.02 e-/pix): never flagged (`hot_columns_as_dark` needs > 0.05 e-/pix excess, gmx.py:173 `min_excess`);
  at lam 1e-5 such a column is 2000x the dark rate and gives the only fakes of those images.
- Negative excursions, row-pedestal residual 0.01 e-, serial-register charge 2e-5 and a x3 row gradient gave no
  visible extra fakes. Big deposits (`busy`) did not reduce the small-event efficiency beyond the errors.
- fe_candidates' expected clean counts are calibrated at s >= 0.15, lam >= 1e-3 (8.5 obs / 7.8 exp), but
  over-predict 2.6x at s0.15 l1e-5 (0.72 / 1.86) and >= 5x at s <= 0.10 (0 observed where 8.7 expected in 46 Mpix),
  and under-predict with the noise tail (9.7 / 5.6): its blanks use `detection_bg(mode='occupancy')` lam, which is
  biased at s = 0.15 (section 4), and Gaussian noise.

## 4. Dark-charge and noise estimates vs truth (per-image mean)

| config | lam_true | bg.lam median | bg.lam mean | bg.lam p90 | detection_bg conservative | detection_bg occupancy |
|---|---|---|---|---|---|---|
| blank s0.06 l1e-5 | 1e-5 | 6.7e-6 | 1.9e-4 | 6.0e-4 | 9.6e-5 (x10) | 7.3e-6 |
| blank s0.10 l1e-5 | 1e-5 | 6.8e-6 | 3.2e-4 | 1.0e-3 | 1.7e-4 (x17) | 1.2e-5 |
| blank s0.15 l1e-5 | 1e-5 | 3.2e-5 | 4.7e-4 | 1.5e-3 | 3.0e-4 (x30) | 7.8e-5 (x8) |
| base s0.10 l1e-4 | 1e-4 | 1.0e-6 (x0.01) | 3.4e-4 | 1.1e-3 | 2.1e-4 | 7.4e-5 |
| blank s0.15 l1e-4 | 1e-4 | 4.8e-5 | 5.1e-4 | 1.6e-3 | 3.3e-4 | 1.2e-4 |
| blank s0.15 l1e-3 | 1e-3 | 8.4e-4 | 1.0e-3 | 2.4e-3 | 1.1e-3 | 8.7e-4 |
| blank s0.15 l1e-2 | 1e-2 | 9.6e-3 | 9.6e-3 | 1.1e-2 | 1.0e-2 | 1.0e-2 |
| base s0.20 l1e-5 | 1e-5 | 1.8e-4 | 7.1e-4 | 2.1e-3 | 6.0e-4 | 2.4e-4 |
| tail s0.15 l1e-5 | 1e-5 | 5.5e-4 | 8.1e-4 | 2.1e-3 | 7.7e-4 | 6.2e-4 |

Read noise is estimated to < 1 % in all configurations. Dark charge:
- `background_map` (agent_methods.py:283ff): the per-row comb fit sees 0.01-0.1 dark electrons per 1280-px row, so
  its median is noise (x0.01-x5); the 2-D correction (l.398-418: mean residual in 60 x 128 cells) has a noise of
  s / sqrt(7680) = 1.1e-3 (s = 0.10) to 1.7e-3 (s = 0.15) e-/pix, and `lam_map = max(lam_s + dmap, floor)` turns the
  positive half into dark charge: mean 2-5e-4, p90 0.6-1.5e-3 at every true lam <= 1e-4 (grows with s, not with lam).
- `detection_bg` conservative (`lam = max(lam, bg.lam)`, gmx.py:168), which the A_GMX counting finder uses, inherits
  those spikes: 10-30x the truth at 1e-5. Verified on single events: touching 2-e- pairs missed at s0.10 l1e-5 sit
  where bg.lam = 1-3e-3 (T = 10.5-12.4 < thr 14.5); found pairs where lam_det ~ 1e-5 have T = 21-26.
- `detection_bg` occupancy mode is unbiased for s <= 0.10 but at s >= 0.15 it subtracts the modelled rounding
  miscounts (P(e > 0.5) = 4.3e-4 at s = 0.15, 6.2e-3 at 0.20), which are 40-600x the dark charge: a 1 % error in s
  moves the result by ~7e-5. With a noise tail it absorbs the tail miscounts as dark charge (x60).
- Symmetric-noise alternative (checked on 10 blank images): lam = [N(n=+1) - N(n=-1)] / N_pix is unbiased (s0.15
  l1e-5: 9.6e-6 +- 4.8e-5 per image; tail s0.15 l1e-5 event-free: -1e-6) because the miscounts are equal on both
  sides; its statistical error per 0.6 Mpix image (4.8e-5 at s = 0.15, 4.5e-6 at s = 0.10) shows that lam ~ 1e-5 can
  only be measured over many images.

## 5. Runtime (single thread, M-series Mac)

| image | background_map | adaptive_cluster | fe_candidates | total |
|---|---|---|---|---|
| 480 x 1280 (0.61 Mpix) | 2.05 s | 1.27 s | 0.57 s | 3.9 s |
| 100 x 500 (0.05 Mpix) | 0.42 s | 0.11 s | 0.04 s | 0.57 s |

(`count_threshold`'s 50 blanks and fe_candidates' 20 blanks are a large part of the clustering and catalogue time.)

## 6. Analytic limits (results/analytic.md)

Per-pixel null P(n = 1) = lam + P(e > 0.5): s <= 0.10 -> ~lam (miscounts 3e-7 at s = 0.10, < 1e-16 at 0.06);
s = 0.15 -> 4.3e-4 (miscounts dominate below lam ~ 4e-4); s = 0.20 -> 6.2e-3; s = 0.25 -> 2.3e-2. Smallest window
charge detectable at 0.08 false windows/Mpix (window 1/2/3/5 px): s <= 0.10, lam <= 1e-5: 2/2/2/3 e- -> **any 2
electrons within a 3 x 3 window are significant**; lam 1e-4: 2/3/3/3; s0.15 lam <= 1e-4: 2/3/3/4 (2 e- only in one
pixel); s0.20: 3/4/5/6. A 1-e- event is never distinguishable from a dark electron (only as a rate excess).
A_GMX's statistic for two touching 1-e- pixels with the TRUE lam: T = 21.6 (s0.10 l1e-5) vs thr ~14.6 -> always
kept; with the lam it actually uses (up to 3e-3) T drops to 10.5-12.4.

## 7. What fails and why (summary)

1. **2-e- events at s <= 0.10, lam <= 1e-5: 0.56-0.73 found where ~0.95 is reachable.** Two causes: (a) touching
   pairs lost (26-37 %) to the inflated detection dark charge (2-D correction noise of `background_map`
   -> `detection_bg` conservative max, gmx.py:168); (b) non-touching pairs (25-35 % of 2-e- events at sigma 0.2-1.2)
   cannot form a group (COUNT_LINK = 1, gmx.py:1046; fe_candidates `_groups` link=1) and are found only 36-38 %.
2. **3-e- events: 0.80-0.98** - losses are diffuse events (sigma > 0.8: 0.65-0.81 at s0.10-0.15), same causes.
3. **Single 1-e- pixels reported as events on small (L2-size) images**, 0.7-0.9/Mpix, 10x nominal (lam floor 1e-6
   + highest-of-50-blanks threshold, gmx.py:169 / 1079-1101).
4. **Fake rate is far from nominal in both directions** (0 in 46 Mpix at s <= 0.10 on 480 x 1280; 10x nominal on
   100 x 500); the threshold depends on image size through the number of blanks and n_exceed.
5. **Read noise >= 0.20 e- or the noise tail**: 2-3 e- events are lost, partly unavoidably (miscount occupancy
   0.6-2 % per pixel); `detection_bg` absorbs tail miscounts as dark charge (x60), so the threshold rises and
   3-e- efficiency drops from 0.82 to 0.51 (s0.15 l1e-4), while fakes stay ~nominal.
6. **Counting energies lose ~1 e- in 25-30 % of events** and 3-9 % of 6-10 e- events are split.
7. **Faint hot columns (< 0.05 e-/pix) are not flagged**; first-column artefact half unflagged; hot pixels and
   single-pixel excursions are reported as 2-6 e- events (need a mask / multi-image persistence veto).
8. fe_candidates' background expectation is biased (x0.4 to x>5) at lam <= 1e-4 and s <= 0.15.

## 8. Candidate improvements, ranked (expected gain from these measurements)

1. **Detection background from a multi-image (or whole-image) occupancy fit, with symmetric miscounts.**
   Replace `max(lam_occ, bg.lam)` by an occupancy estimate that does not see the 2-D-correction noise; estimate the
   miscount rate from the n = -1 pixels (P(n=+1) - P(n=-1)) instead of the Gaussian model, so the noise tail and s
   errors cancel; pool lam, s and the tail over the images of a run (lam 1e-5 = 6 dark e- per 0.6 Mpix image: +-40 %
   per image, +-6 % over 50 images); skip or heavily smooth the 2-D correction when lam x cell < ~10 e-.
   Expected gain: touching 2-e- pairs 0.63-0.74 -> ~1.0 at s <= 0.10 (overall 2 e- ~0.6 -> ~0.75), 3 e- ->
   ~0.97; with the noise tail 3 e- 0.51 -> ~0.8 (back to the Gaussian value); fe_candidates expectation unbiased.
   Cheap (a few lines in detection_bg / background_map; must be recalibrated so fakes stay nominal).
2. **Analytic false-alarm probability instead of blank simulations** (count_threshold and fe_candidates blanks).
   The null of a group / window charge is a convolution of per-pixel pmfs (as in proto_box.window_sf) or an exact
   sum over polyplet shapes; give it an explicit uncertainty on lam (upper limit when few singles are seen), and
   never accept a 1-electron group. Expected gain: removes the L2-geometry fakes (0.7-0.9 -> <= 0.08/Mpix), makes
   the fake rate independent of image size (thr scatter +-2.2 in T now), saves the 70 blank simulations per image (not timed separately), and reaches the
   rates 0.01-0.1/Mpix without simulating hundreds of Mpix.
3. **Likelihood-ratio few-electron finder with a diffusion kernel** (matched filter in electron counts: for each
   centre and sigma in {0.3, 0.7, 1.2} px, LR of "n_e electrons spread by the kernel + background" vs "background",
   over the counted pixels in a 5 x 5 window, maximised over n_e), instead of 8-connected groups. Prototype
   (`proto_box.py`, only a top-hat window count with the exact null, Bonferroni threshold, no shape term):
   at s <= 0.10, lam <= 1e-5: **2 e- 0.90-0.98** (A_GMX 0.56-0.62), 3 e- 1.00 (0.90-0.98), non-touching 2-e- pairs
   0.64-0.92 (A_GMX 0.36-0.38), with 0 fakes in 46 Mpix; L2 geometry 2 e- 0.76-0.77 (A_GMX 0.51-0.64), 0 fakes in
   20 Mpix. But the plain window sum is worse where the pixel arrangement matters: 2 e- in one pixel at lam 1e-4
   (0.14-0.21 vs A_GMX 0.40-0.64), 3 e- at lam 1e-3 (0.05 vs 0.65), 4 e- at 1e-2 (0.08 vs 0.38) - the kernel LR
   keeps A_GMX's per-pixel likelihood and adds the window grouping, so the expected gain is the better of the two
   in every cell: 2 e- -> ~0.95 at the L2 target (s 0.06-0.10, lam <= 1e-5), 3 e- -> ~1.0 everywhere s <= 0.15,
   and it collects the non-touching electrons (energy completeness 70 % -> ~90 %+, fewer splits). Most work.
4. **Masks / vetoes for detector artefacts**: hot-column threshold relative to lam (0.02 e-/pix columns are
   2000x the dark rate at 1e-5), always mask the first column(s), hot pixels by persistence over images, single
   pixel excursions by per-skip consistency. Gain: removes ~4 fake events/image in the `art` configuration and the
   0.2-0.3/Mpix hot-column fakes; necessary before any of the above matters on real L2 data.
5. **Bayesian evidence per candidate** (marginal likelihood over position, sigma and n_e with a depth / diffusion
   prior). Detection gain over 3 small (same information); main value is a posterior for n_e and sigma (charge and
   depth) and a calibrated per-candidate probability for the catalogue. Do after 3.

1-e- events: not findable per event at any s (the null is the same as the signal); only a rate excess over the
measured dark charge (requires 1 and a multi-image dark-charge measurement).

## Uncertainties / caveats

- 45-90 events per n_e per configuration: efficiency errors 0.02-0.08; fake rates at 0.05-0.1/Mpix are measured
  only for s = 0.15 and the L2 geometry (55 / 10 Mpix); below 0.022/Mpix only upper limits.
- 30 events per 0.6 Mpix image (~50 per Mpix) is far denser than real low-background data; their stray single
  electrons add ~1.4e-4 counted pixels per pixel (a raw n=+1 minus n=-1 count is biased by it; the isolated-pixel
  estimators of detection_bg / proto_box were not visibly affected: base vs blank agree).
- Diffusion sigma drawn uniformly in 0.2-1.2 px (not the physical depth distribution); events are not near the
  edges; the read-noise tail model (0.5 % at 3 sigma) and artefact amplitudes are guesses until the L2
  characterisation fixes them (all are parameters of sim.py / bench_agmx.py).
- The prototype threshold is Bonferroni (conservative, 0 fakes observed); a calibrated version would be somewhat
  more efficient. It estimates one lam per image and has no tail model.
