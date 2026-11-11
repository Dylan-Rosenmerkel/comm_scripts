# Validation of the A_GMX / Bayes few-electron switch (2026-11-11)

Code under test: `wip/lowdc/bayes/agmx_switch.py` (switch) and the `fewe` hook in `clustering_tests/adaptive.py`.
Frozen copies here: `agmx_switch_v0.py` (as implemented 2026-11-10), `agmx_switch_v1.py`, `agmx_switch_v2.py` (= the
live file now). All tables: `results/tables.md` (from `analyze.py --versions v2`). Raw per-group records:
`results/v2/<group>__{events,fakes,images}.csv` (tag `agmx` = A_GMX alone, `single`, `pooled`, `on`) and
`results/v1/` (9 groups, an intermediate version, used only for the per-image-budget comparison).

The work was cut short at the coordinator's request. **Not run: the halo hand-over test (`halo_test.py`), the
block-boundary test (`boundary_test.py`), the L2 ACM 103 real data (`real_l2_switch.py`) and a v0 benchmark.**
The scripts are written and ready (see "Next steps").

## Method

- `bench_switch.py`: the bench_agmx images (same configs and seeds), plus `base_s0.16_l0.0001` and
  `base_s0.17_l1e-05` (between the f_c cuts) and `sparse2_s0.15_l1e-05 / _l0.0001` (2 events per image instead of 30:
  realistic density). A_GMX's own output (fewe='off') is cached per image; the switch is applied to it.
  `check_hook.py` shows that this equals `adaptive_cluster(fewe='auto')` exactly (single and pooled).
- Groups as `bayes/bench_bayes.py`: base_x + blank_x pooled; the other configs alone.
- Modes: **single** = `apply()` with single-image estimates; **pooled** = `prepare_run(images of the group, full=True)`,
  then each image with the run's lam map, R_N, P(event) cut and noise model; **on** = forced Bayes.
- Matching `metrics.match`. Fakes are unmatched objects < 20 e-, not within 40 px of a big deposit, counted over all
  images of the group (blank images included).
- The finder modules (noise, finder, pipeline, background) were edited this morning by another agent. The benchmark
  ran on a snapshot of them taken 07:51 (`SCRATCH/switch_validation/finder_snapshot`, md5 there).
  A smoke test against the live modules (`SWV_LIVE=1`, prepare_run(full=True) + hook) passes.

## Bugs found and changes v0 -> v2 (agmx_switch.py)

| # | problem in v0 | evidence | fix (v2) |
|---|---|---|---|
| 1 | At lam >= LAM_2E only `N_mean >= 3` events were kept. The posterior mean of a true 3-e- event is 2.98-2.999. | Stand-alone records: N_mean < 3 for **92 %** (lam 1e-3), 86 % (s0.15 l1e-4) and 59 % (s0.06 l1e-4) of found 3-e- events, so v0 dropped them. | Keep events with P(N >= 3 \| event) >= 0.5, applied **before** the fake budget (the dropped 2-e- candidates no longer use up the budget). Result at lam 1e-3: 3 e- 0.91 (pooled) vs A_GMX 0.65. |
| 2 | Per-image fake budget F x 0.61 Mpix = 0.05 expected fakes per image. On images with many events the residual P(no event) of the true events uses it up, which forces P(event) >= 0.999. | s0.15 l1e-5: per-image cut 0.999 vs 0.12 for the stand-alone run budget; 2 e- 0.76-0.79 instead of 0.97. L2 geometry (0.05 Mpix): budget 0.004 per image. | `run_threshold()` + `prepare_run(..., full=True)` returns the run's P(event) cut, passed through `apply(threshold=)` / `adaptive_cluster(fewe_threshold=)`. |
| 3 | Single-image noise fit: the finder's excursion component keeps a prior weight of about 1/n_neg (4e-5 on a 100 x 500 image). A clean single-pixel 4-7 e- event gets log J1 8-10 and P(event) 0.03-0.9. | L2 geometry s0.06 (v1 pooled, per-image noise): 4 e- **0.92** vs A_GMX 1.00. All lost events have sigma 0.2-0.35, i.e. single pixels. | `prepare_run` pools the noise model (nm_pooled=True by default) and returns it. `apply(nm=)` / `adaptive_cluster(fewe_nm=)` use it. L2 s0.06 pooled: 0.99/1.00/1.00/1.00. |
| 4 | Bayes clusters reported y, x in the finder's convention (pixel i = [i, i+1)). A_GMX and panaSKImg put the centre of pixel i at i. | A_GMX centroid minus the true pixel index: -0.0003 px. Bayes: +0.5 px. | y - 0.5, x - 0.5. |
| 5 | Halo measured from the big clusters' **centroids**. A muon track is up to 120 px long, so events near a track were handed to Bayes, and A_GMX pieces of the deposit outside the 15 px were dropped. | By construction; not quantified (halo test not run). | Halo = distance to the nearest **pixel** of a big deposit (A_GMX `_pix` plus the finder's own > 20 e- candidate groups). The hook requests pixel lists from GMX when fewe is on and strips them afterwards. |
| 6 | f_c counted the pixels of big deposits, so a muon or Fe-55 could flip its 100 x 100 block to A_GMX. | By construction (v0 smoke test on busy: 1 agmx block). v2 on busy: 0.97 bayes / 0.03 both / 0 agmx blocks. | f_c is counted outside the big-deposit footprints padded by BIG_PAD = 2 px (`f_c_all` is still reported). |
| 7 | Inconsistent ownership. An A_GMX cluster was judged by its centroid's block, a Bayes event by its own centre plus the finder's halo flag (any pixel within 15 px). One event could be counted twice or lost at block and halo edges. | By construction; not measured (boundary test not run). | An A_GMX cluster within LINK = 1 px of a neighbourhood follows that neighbourhood's decision; otherwise its own centre decides. |
| 8 | Fake budget over all neighbourhoods and the whole image area, including 'agmx' blocks and halos. | By construction. | Budget over the neighbourhoods and the area of the Bayes region only. |
| 9 | `measure()` ran even in A_GMX crowded mode. | Wasted time only. | Crowded mode returns before the noise stage. |

Hook (`adaptive.py`): new optional arguments `fewe_threshold`, `fewe_nm`. GMX is asked for pixel lists when fewe
is not 'off' (dropped again if `return_pixels=False`). Signatures of `apply()`, `prepare_run()` and the existing
fewe arguments are unchanged (new optional arguments only). The default `prepare_run(images)` still returns
(lam_maps, R_N), now with the pooled noise model.

**fewe='off' is bit-identical** to adaptive.py before the hook (`check_hook.py`, re-run on the final code). Checked
on base s0.15, busy, s0.06 l1e-4, art, s0.25, a lam 0.5 image and an unmodelled s1.5 image, with and without
pixel lists. `fewe='auto'` equals `apply()` on A_GMX's output, and `return_pixels=False` gives the same clusters.
(The lam 0.5 test image did not trigger crowded mode, so the crowded path is covered only by code inspection.)

## 1. Benchmark (efficiency 2 / 3 / 4 / >= 5 e-, fakes count and per Mpix)

Selected groups (all 38: `results/tables.md` section 1). "Bayes alone" = the stand-alone records
`bayes/results/physics` at their group operating point.

| group (Mpix) | A_GMX alone | switch single | switch pooled | Bayes alone |
|---|---|---|---|---|
| base s0.06 l1e-5 (46) | 0.56/0.98/0.98/1.00 (0) | 0.90/0.98/1.00/1.00 (0) | **1.00**/1.00/1.00/1.00 (3, 0.07) | 1.00/1.00/1.00/1.00 (0) |
| base s0.10 l1e-5 (46) | 0.62/0.90/0.98/1.00 (0) | 0.78/1.00/1.00/1.00 (0) | **1.00**/1.00/1.00/1.00 (3, 0.07) | 1.00/1.00/1.00/1.00 (1) |
| base s0.15 l1e-5 (55) | 0.20/0.80/0.93/1.00 (3, 0.05) | 0.79/0.95/1.00/1.00 (0) | **0.97**/1.00/1.00/1.00 (2, 0.04) | 0.97/1.00/1.00/1.00 (2) |
| base s0.15 l1e-4 (55) | 0.12/0.82/0.94/1.00 (6, 0.11) | 0.47/0.95/1.00/1.00 (6, 0.11) | 0.72/0.99/1.00/1.00 (7, 0.13) | 0.72/0.99/1.00/1.00 (7) |
| base s0.15 l1e-3 (55) | 0.00/0.65/0.85/1.00 (4, 0.07) | 0.02/0.83/0.98/1.00 (1, 0.02) | 0.03/**0.91**/0.99/1.00 (2, 0.04) | 0.03/0.91/0.99/1.00 (2) |
| sparse2 s0.15 l1e-5 (25; 2 ev/img) | 0.07/0.50/1.00/1.00 (0) | 0.73/1.00/1.00/1.00 (0) | 1.00/1.00/1.00/1.00 (1, 0.04) | - |
| busy s0.15 l1e-4 (9.2; Fe-55 + muon) | 0.24/0.72/0.96/0.99 (1) | 0.51/0.95/1.00/1.00 (0) | 0.73/0.97/1.00/1.00 (0) | 0.73/0.97/1.00/1.00 (0) |
| art s0.10 l1e-5 (9.2) | 0.42/0.72/0.98/1.00 (65, **7.05**) | 0.77/0.93/0.95/0.97 (14, 1.52) | 0.79/0.93/0.95/0.96 (7, 0.76) | 0.79/0.95/0.95/0.97 (3, 0.33) |
| tail s0.15 l1e-4 (9.2) | 0.04/0.51/0.82/1.00 (1) | 0.16/0.87/0.93/1.00 (2) | 0.14/0.87/0.98/1.00 (2) | 0.14/0.87/0.98/1.00 (2) |
| L2 100x500 s0.06 l1e-5 (10) | 0.64/0.90/1.00/1.00 (9, **0.90**) | 0.72/0.91/**0.92**/0.98 (0) | **0.99**/1.00/1.00/1.00 (1, 0.10) | 0.99/1.00/1.00/1.00 (1) |
| L2 100x500 s0.10 l1e-5 (10) | 0.51/0.85/0.99/1.00 (7, 0.70) | 0.78/0.94/0.96/0.98 (0) | 0.96/1.00/1.00/1.00 (0) | 0.96/1.00/1.00/1.00 (0) |
| base s0.16 l1e-4 (9.2; 'both') | 0.07/0.75/0.90/0.99 (0) | 0.50/0.94/1.00/1.00 (0) | 0.73/1.00/1.00/1.00 (0) | - |
| base s0.17 l1e-5 (9.2; f_c 1.58e-2) | 0.17/0.61/0.85/1.00 (1) | 0.38/0.67/0.93/1.00 (0) | 0.45/0.67/0.91/1.00 (0) | - |
| base s0.15 l1e-2 (55; 'both') | 0.00/0.01/0.38/0.92 (5, 0.09) | 0.00/0.04/0.61/0.96 (12, 0.22) | 0.00/0.01/0.50/0.95 (7, 0.13) | - |
| s0.20 / s0.25 / tail s0.15 l1e-2 (all) | as A_GMX | **identical** to A_GMX (90/90 images) | - | - |

- **Pooled mode reproduces the stand-alone finder exactly** in every group that has a stand-alone record: same
  efficiencies, same fake counts, and the same run threshold (s0.15 l1e-5: 0.1245 in both). In the sparse regime it
  is better than A_GMX everywhere: 2 e- +0.2 to +0.8, 3 e- up to +0.26, art fakes 0.76 vs 7.05 per Mpix, L2
  fakes 0.0-0.1 vs 0.7-0.9 per Mpix.
- **Single-image mode** is better than A_GMX at 2-3 e- but conservative: the per-image budget and the per-image
  noise fit cost 2-e- efficiency (0.47-0.90 vs 0.72-1.00 pooled). It is **worse than A_GMX for single-pixel 4-10 e-
  events on small images** (L2: 4 e- 0.92-0.96 vs 0.99-1.00; bug 3 still applies in this mode). Its fakes: art 1.52
  per Mpix, hotcol s0.15 l1e-5 0.76 per Mpix (A_GMX 0.33).
- Per-image budget with run statistics (v1 `pooled_it`, 9 groups): s0.15 l1e-5 2 e- 0.76, l1e-4 0.21, L2 s0.06
  0.57. So most of the single-image loss comes from the per-image budget.
- Blank images (`tables.md` 1b), per Mpix, A_GMX / single / pooled: s0.15 l1e-5 0.08 / 0.00 / 0.05; s0.15 l1e-4
  0.08 / 0.11 / 0.16; s0.15 l1e-3 0.11 / 0.00 / 0.00; s0.06 and s0.10 l1e-5 0.00 / 0.00 / 0.05.
- Electron counts (single object found, sparse groups): exact count A_GMX 0.71-0.95 (low by >= 1 e-: 3-28 %), switch
  0.98-0.99 (low <= 1 %). Split events: A_GMX 5.5 %, switch 0.
- 1-e- events: A_GMX finds 1 % of them, the switch none (the Bayesian model has no N = 1 events; they belong to lam).
- Big deposits (busy, 60 Fe-55 / muons): found 60/60 with q/true = 1.0000 in every mode. The A_GMX big clusters pass
  through unchanged.
- Runtime per 480 x 1280 image (A_GMX 3.2-3.7 s): switch 0.5-1.4 s (s <= 0.15 Gaussian), 3.4-5.4 s with the noise
  tail, 7-9 s at f_c 1.4-1.6e-2 (lam 1e-2, s0.17). prepare_run adds 0.4-2.3 s per image (7-18 s near the cut).
  **prepare_run runs the full finder even when every block is A_GMX's**: 44 s per image for tail s0.15 l1e-2, which
  the rule hands to A_GMX completely, and several minutes per image at s0.20 (that is why the pooled run of the
  s0.20 / s0.25 groups was stopped). 100 x 500 images: 0.06-0.15 s (A_GMX 0.53 s).

## 2. Regime decisions

`tables.md` section 2: the decisions follow the rule in every group.
- s <= 0.15 Gaussian, lam <= 1e-3: f_c 1e-5 to 5.5e-3, 'bayes' blocks 0.78-1.00 (rest 'both').
- s0.16: f_c 9.0e-3, all 'both'. Noise tail at s0.15: f_c 1.1-1.2e-2, 'both'. lam 1e-2: f_c 1.42e-2, 76 % 'both' and
  24 % 'agmx'. s0.17: f_c 1.58e-2, 25 % 'both' and 75 % 'agmx'.
- s0.20 / 0.25, tail s0.20, hotcol s0.20, tail s0.15 l1e-2 (f_c 2.1e-2 to 0.2): not used, output **identical to
  A_GMX** in 90/90 images.
- The N >= 3 rule switched on exactly in the lam >= 1e-3 groups (lam estimate 5e-4 to 1e-3); never at 1e-4.

## 3. DM09 Image_5 through src/agmx_root.py (AGMX_FEWE=auto vs off, no g(n) curve, 12 CCDs)

`dm09.py`, `results/dm09_compare.csv`, `results/dm09_decisions.csv`.
- A_GMX was chosen everywhere: f_c = 1.0 (image 0, lam 1.0-1.2 e-/pix by A_GMX's estimate) and 0.13-0.16
  (images 1 and 2) in every block. 0 Bayes events. Crowded mode was not triggered.
- The cluster files are **bit-identical** (Energy, STD_XY, PosX, PosY) on all 12 CCDs. Cluster counts 916-1028 per
  CCD, unchanged. K-alpha is therefore unchanged by construction. My simple Gaussian fit found no clean K-alpha peak
  in these files (no line in 5.4-6.6 keV), so I quote no K-alpha value.
- Overhead of the switch: +0.2-1.7 s per CCD (the noise stage), on 8-16 s.
- The run with --curve (g(n) applied) was not done.

## Parameters in force (agmx_switch.py)

| parameter | value | status |
|---|---|---|
| FC_BAYES / FC_AGMX | 6e-3 / 1.5e-2 | unchanged. 'bayes' and 'both' blocks are treated the same (Bayes catalogue), so only FC_AGMX matters. See the note below. |
| LAM_2E | 3e-4, now with P(N >= 3 \| event) >= 0.5 | the cut value itself is untested between 1e-4 and 1e-3 |
| BIG_E / HALO | 20 e- / 15 px from the nearest deposit pixel | halo not tuned (test not run) |
| BIG_PAD | 2 px (new) | footprint padding for f_c |
| LINK | 1 px (new) | ownership link radius |
| BLOCK / MIN_BLOCK_PIX | 100 / 2000 | unchanged |
| F_FAKE | 0.08 per Mpix | unchanged; per run in pooled mode, per image in single mode |

Evidence on FC_AGMX, not acted on: at s0.17 (f_c 1.58e-2, just above the cut) forced Bayes gives
0.64/1.00/1.00/1.00 with 0 fakes, vs A_GMX 0.17/0.61/0.85/1.00 (1 fake) and the rule's 0.38/0.67/0.93/1.00
(15 images, single mode, 7 s per image). At lam 1e-2 (f_c 1.42e-2, below the cut) the switch helps 4-e- events
(0.50-0.61 vs 0.38) but raises fakes (0.13-0.22 vs 0.09 per Mpix) and costs 3x A_GMX's time. So the cut is in about
the right place. It may be raised to about 2e-2 for read-noise-dominated images, but lam-dominated images argue
against that. More images are needed before changing it.

**Recommended use:** run-pooled mode, i.e. `run = prepare_run(images_of_run, full=True)`, then for image k
`adaptive_cluster(..., fewe='auto', fewe_lam=run['lam_maps'][k], fewe_RN=run['R_N'], fewe_threshold=run['threshold'],
fewe_nm=run['nm'])`. Single-image mode (`AGMX_FEWE=auto` alone, which is what agmx_root does) is safe but
conservative, and it loses single-pixel events on small images.

## Untested / open

1. **Halo hand-over** (events 5-40 px from Fe-55 / muons) and **block boundaries** (events straddling blocks with
   different regimes): scripts `halo_test.py`, `boundary_test.py`, `synth.py` are ready, not run. Fixes 5 and 7 are
   verified only by construction and by the busy group (60/60 big deposits unchanged, no small-event loss vs
   stand-alone).
2. **L2 ACM 103 real data**: `real_l2_switch.py` ready (per-row median pedestal, 1.3 ADU/e-, EXT4 1.3 and 0.82,
   column 0 masked, pooled per NDCM pair), not run.
3. prepare_run cost when the run is out of regime (44 s to minutes per image). It should first measure f_c and skip
   the finder when no block can be Bayes's.
4. src/agmx_root.py (not mine) maps cls 'few-e' and regime 'bayes' to no code (agmx_cls 0, agmx_regime -1). It also
   calls the switch in single-image mode only. For production it needs the run-pooled call and codes for the new
   class and regime.
5. Crowded-mode path: covered by code inspection only (no test image triggered crowded mode).
6. The g(n)-corrected DM09 run (--curve) was not done. No v0 benchmark (v0's bugs are quantified from the
   stand-alone records and v1 instead).

## Next steps

1. Run `halo_test.py` and `boundary_test.py` (about 15 min with 2 workers), then tune HALO from the efficiency and
   fakes vs distance.
2. Run `real_l2_switch.py` (light, 100 x 500 images).
3. In prepare_run: skip the finder for runs whose f_c is above FC_AGMX everywhere.
4. agmx_root: add a run-pooled path (group the images of a run, prepare_run(full=True)) and CLS / REGIME codes for
   'few-e' / 'bayes'.
5. Revisit FC_AGMX with more s0.17-0.18 images (Bayes beat A_GMX at f_c 1.58e-2).

## Files

`common.py` (paths, configs, A_GMX cache, scoring), `bench_switch.py`, `check_hook.py`, `analyze.py`, `dm09.py`,
`synth.py`, `halo_test.py`, `boundary_test.py`, `real_l2_switch.py`, `agmx_switch_v0/v1/v2.py` (frozen copies).
The A_GMX cache and the DM09 outputs are in the scratchpad
(`/private/tmp/claude-501/-Users-dylanrosenmerkel/511be665-1540-4ae6-86c3-c4f8e8738785/scratchpad/switch_validation/`).
