# Handoff (2026-10-25, late): A_GMX phase 1 done, phase 2 and the physics-audit fixes in progress

> **Status: current.** Read this first in the next session, then `OPEN_ITEMS.md`. Earlier context:
> `CONTEXT_2026-10-24.md` (sections 8-10) and `CONTEXT_2026-10-16.md` (sections 1-3: ground rules).

Everything was stopped on request at the end of the session: no background job is running, the repository is clean,
all work is committed. Unfinished work (scripts, prototypes, partial results) is saved under
`~/Documents/comm_scripts_data/session_2026-10-23_clustering/clustering_2026-10-25/` (called **DATA** below);
`DATA/next_steps_2026-10-25/` holds the scratch prototypes (the session scratchpad in /private/tmp is temporary).

## 0. Ground rules (unchanged)
- Show proposed edits to comm_scripts and get approval before executing them. In this session the user approved:
  phase 1, phase 2 *in a scratchpad*, a physics audit, "fix any physics issues found", appending the A_GMX document.
- Do not modify panaSKImg / pysimdamicm (read-only use) or the colleague's `LSM_Module_Testing_Analysis` files.
- Fe-55 results: front side (sigma_xy < 1) unless asked. Commit only when asked; branch `clustering-methods-2026-10-25`
  (nothing pushed; main untouched).
- Python: `/opt/homebrew/Caskroom/miniforge/base/envs/damic/bin/python`. Tests: `python -m unittest discover -s tests`
  (55 tests, all pass at 12f97f2).

## 1. Repository state (branch `clustering-methods-2026-10-25`, HEAD 12f97f2)
Commits of 2026-10-25, newest first:
| commit | what |
|---|---|
| 12f97f2 | **phase 1**: compiled (numba) EM, pixel-likelihood few-electron finder, injection footprint check |
| 9200a20, a321499, d92a06a | A_GMX document re-run with the audited code (`*_v3` results), generated tables, layout |
| 0e25293 | g(n) measurement honours `--autoconfig-args`; incoherent combination order-independent |
| 37e4e53 .. 0297609 | repository audit parts A-E (bugs, pipeline, utils, clean-up, docs organisation) |
| 4ee5521 | A_GMX in run_WADERs (`--clusterer`), `bin/run_AGMX`, WADERs-format writer, user guide |

Phase 1 (12f97f2) in detail:
- `gmx.py`: `_em_pass` / `em_numba` (numba; identical results to 1e-10 keV, crowded CCD 32 s -> 15 s; `AGMX_NUMBA=0`
  for numpy). Few-electron finder `_counting_pix` (default `count_mode="pix"`, `count_fakes=0.08` per Mpix): statistic
  T = -sum ln P(n_i | Poisson(lam) x N(0, s), rounded) - ln N_shapes(m); threshold = max(ln(N_pix/F), calibrated on
  simulated blanks of the image's own background, fixed seed). Fitted-background blanks (30 per level): 0.03 / 0.13 /
  0.07 fakes per 1.75 Mpix at lam 0.001 / 0.01 / 0.05 (target 0.14). `count_mode="box"` = the old test.
- `inject_utils.py` + `wide_test.inject_eval` / `mhc_tracks.inject_eval`: an injected object is placed only where its
  whole footprint (+2 px) avoids real events (`INJECT_FOOTPRINT=0` = old centre-only placement; the saved results
  before 12f97f2 used the old placement).
- `adaptive_validate.factory`: methods WADERs and MFD+G added.
- **Not yet done for phase 1:** analysing its validation runs (section 2) and documenting it in the A_GMX document.

## 2. Interrupted phase-1 validation runs (code 12f97f2; outputs in DATA)
| run | output | state |
|---|---|---|
| wide test A_GMX, old placement (`INJECT_FOOTPRINT=0`, comparable with the other methods' saved runs) | `results_wide_adaptive_gmx_v4/` | 167/878, interrupted |
| few-electron validation (GMX, A_GMX) | `results_adaptive_gmx_v4/lowE/` | 6/6 done, not analysed |
| Fe-55 validation | `results_adaptive_gmx_v4/fe55/` | 16/16 done, not analysed |
| crowded CCD | `results_adaptive_gmx_v4/crowded/` | 29/29 clusters done; `crowded fit` not run |
| tracks with the NEW placement, all methods (WADERs, MFD+G, MHC, GMX, A_GMX) | `results_tracks_footprint_v4/tracks/` | 17/18, interrupted |

Resume (jobs skip existing outputs), from a snapshot of the code so that later edits do not leak into running workers
(`git worktree add --detach /tmp/wt HEAD`; run from `/tmp/wt/clustering_tests`):
```
INJECT_FOOTPRINT=0 WIDE_OUT=DATA/results_wide_adaptive_gmx_v4 NPROC=4 python wide_run.py adaptive+GMX
METHODS=WADERs,MFD+G,MHC,GMX,adaptive+GMX VALIDATE_OUT=DATA/results_tracks_footprint_v4 NPROC=4 python adaptive_validate.py tracks
METHODS=GMX,adaptive+GMX VALIDATE_OUT=DATA/results_adaptive_gmx_v4 python adaptive_validate.py crowded fit
```
Then point `make_agmx_perf.py` at the v4 folders (its `latest()` picks `_v3`; make it prefer `_v4`) and, for the track
tables, at `results_tracks_footprint_v4` for all methods (the old `results_tracks` used the old placement and too-faint
muons, see 4.4).

## 3. Phase 2 (scratch, not in the repository): exact pixel integrals and EM convergence
Prototype: `DATA/next_steps_2026-10-25/phase2/gmx.py` (repo gmx at 12f97f2 + switches):
- `P2_EXACT=1`: exact pixel integrals (erf) in `evaluate()` for clouds with |rho| < 0.5, with the consistent M-step
  (pixelised variance / mean inverted, tables `_pixvar_table`, `_pixmean_table`) **only for those clouds** (the
  improvement agent's prototype applied it to all clouds - fixed); the numba EM is bypassed in exact mode.
- `P2_TOL=x`: EM until the largest centre step < x px (cap n_iter x `P2_MULT`); `EM_LOG` records (iterations, converged).

Results so far:
- **Single X-rays** (`sim_single.py`, 80 per width, 1580 e-): energies unbiased in both models (median E/E 0.9999-1.0001,
  rms 0.05-0.14 %). Approximate model: sharp X-rays (sigma <= 0.3 px) split into several clouds (1-cloud 41 % at 0.2,
  21 % at 0.3) and their width is biased (0.289 at 0.3); exact: 1 cloud 100 %, width 0.304 at 0.30, 0.250 at 0.20 (the
  latter and the 0.14-px centroid rms are the one-pixel physical limit). Above 0.4 px identical.
- **Crowded CCD convergence** (`conv_crowded.py`, one runs_29img EXT4 image; `c_fixed/c_tol01/c_tol003.npz`): the
  whole-image EM never converges even after 300/600 iterations (the island EMs do, 98 %). Fixed 30 iterations vs
  converged (tol 0.01 px): only ~50 % of clusters > 1 keV reproduced within 0.5 px; instability concentrated in the
  charge sheet (ped >= 10: 42 % vs 66 % reproduced). Clean selection (sxy<0.7, blend<0.02, ped<10): 82 % reproduced,
  78 % energy within 0.5 %; tol 0.01 vs 0.003: 91 % / 90 %. 59.5 keV median 59.535 (fixed) -> 59.494 (converged), one
  image. Converging costs 3.4x on this CCD. Interpretation: in a charge sheet individual clusters are not identifiable
  (many near-equivalent cloud arrangements); only selections excluding the sheet are physically meaningful, and the
  crowded-CCD numbers in the document depend on the iteration count.
- **Not yet run** (scripts ready in `phase2/`, launcher `p2_launch.sh`): `p2_real.py` on real data (current vs exact,
  margin 100; Fe-55 K-alpha per cut, K-alpha widths = front/back, flags, injections) for every 12th Fe-55 and every 6th
  Am-241 image-CCD; `crowd29.py` all 29 crowded images fixed vs converged (then the constrained Lepy fit:
  `am241_29img_fit.py` with `CDIR`, `CGLOB="*.npz"`, `BLEND=0.02`); `sim_sharp.py base,exact_m100` (sharp-on-diffuse).
- Decision still to make after those: adopt exact integrals (+ island margin 100, flags recalibrated, K-alpha re-checked
  per data set; the improvement agent saw the flag-selected K-alpha move by up to +-44 eV on some images) and/or a
  convergence criterion (tol 0.01 px) for spectroscopy.

## 4. Physics audit of the clustering methods (agent report, partly verified)
Agent scripts: `DATA/next_steps_2026-10-25/physics_audit/simA-E.py`. Status per finding:

### 4.1 WRONG, verified here, prototype fixes in `next_steps_2026-10-25/phase2b/` (switches default on)
| id | finding (verified numbers) | prototype fix | sim result (off -> on) |
|---|---|---|---|
| A1 | `agent_methods.fit_row_fast` declares a row 'resolved' if abs(z) > 0.018, below the noise floor sqrt(pi/4N) of unresolved noise: Gaussian noise s=1.5 fitted as s~0.4 e-, the switch does not fire, ~8000 fake clusters/Mpix | threshold max(0.018, 4/sqrt(N)) (`P2B_A1`) | 7966 -> 1.0 fake/Mpix (s=1.5); 387 -> 0 (s=3) |
| I1 | read-noise estimate = MAD of pixels with abs(Q)<0.5 cannot exceed 0.371 | `AM.read_noise`: robust MAD of all active pixels when the first estimate is > 0.3 (`P2B_I1`); used by adaptive and simple_bg | estimate 0.37 -> 1.50 / 3.00 |
| C2 | counting mode for median s < 0.5, but rounding miscounts are ~10 % at s=0.4: fakes (old finder 50/Mpix at s=0.4; also the new pix finder 313/Mpix at s=0.45) | counting only if s < 0.35 (`P2B_C2`) | 313 -> 0/Mpix (s=0.45) |
| B1 | hot columns: bright ones hidden by the cluster mask (not flagged); flagged ones only have the mean subtracted (no variance): +2 e-/pix column -> 50-58 fake low-E clusters (15-183 eV), +0.5 -> 10 fakes | `gmx.hot_columns_as_dark`: clipped-mean excess without the cluster mask, added to the column's lam (mean, variance, Poisson tests follow) (`P2B_B1`) | 50 -> 0 fakes (+2 e-/pix); very hot column (+10) no longer reported as an 'artifact-column' cluster (listed in info['hot_columns']) |
Control (40 simulated Fe-55 X-rays): unchanged (E/E 0.9998, rms 0.0009, no extra clusters). Script: `audit_fix_check.py`.
**Not yet done:** the real-data test `p2b_real.py` (fixes on vs off; the phase-2 subset + all PD06/DM12 image-CCDs,
launcher `phase2b/launch.sh`) - A1/I1 change which real images count as 'unmodelled', so this test decides adoption.

### 4.2 QUESTIONABLE: fixes being written when the session was stopped (nothing implemented yet)
- **A2 lam miscount bias**: rounding counts noise > +0.5 e- as an electron; lam_fit - lam ~ p = Phi_c(0.5/s) (verified:
  lam 0.0595 at s=0.3 and 0.66 at s=0.4 for true 0.01). Energies biased by -N_pix * dlam (-0.1..-0.2 % on K-alpha at
  s 0.25-0.3). Planned fix in `fit_row_fast` (returns lam at `agent_methods.py` line ~250 in phase2b):
  lam <- max((lam_fit - p) / (1 - 2p), floor), p = Phi_c(0.5 / s); test on blanks lam in {0.001,0.01,0.1,1},
  s in {0.15,0.2,0.3}.
- **G1 muon label** (`mhc.classify_ext`): geometry only. A MIP deposits ~0.27 keV/um -> >= ~4 keV per 15-um pixel of
  projected path (more for steep tracks); planned rule: muon only if E/length >= 2.5 keV/px, else 'extended'.
  **Test-design physics bug found at the same time:** the injected muons in `wide_test.sim` / `tracks_test.sim`
  deposit 250 e-/px = 0.94 keV/px, ~4x below a MIP. Fix the simulation together with the rule (realistic energy:
  0.27 keV/um x 3-D path sqrt((15 L)^2 + 675^2) um with Landau-like fluctuations, width growing with depth), behind a
  switch so old results stay reproducible.
- **G2 saturation flag**: 4-9 MeV alphas saturate/bloom; add a flag (pixels at the image's saturation plateau).
- **D2 width bias**: the E-step floor eps = sqrt(V) makes fitted sigma 1-4 % low (3-11 % for 200-e- clusters at lam=1),
  so sigma cuts (front/back) depend on energy and dark charge; report sigma from a final pass with eps -> 0 (or sxy_mom).
- **D1 pixelisation** (+1/12 approximation; profile sums 1.38 / 0.68 at sigma 0.2) -> phase 2 exact integrals; also the
  cause of spurious 1-e- 'low-E' clusters next to 1.7-4.2 % of sharp X-rays and of the chi2 flag bias for sharp clusters.

### 4.3 QUESTIONABLE / undocumented: to document (or calibrate before changing)
- A3: the 2-D cell correction adds pedestal drift to lam (raises V); negative drift lost at the lam floor.
- C1: the detection fake rate depends on lam: 0 in 4 Mpix at lam <= 0.05, ~1 per 3-4 Mpix at lam 1-2 (16-27 e-).
- E1: footprint noise sigma_E^2 = sum r^2 V (Kalpha sigma 1.5: 2.2 e- at lam 0.01, 11 e- at lam 1, comparable to the
  Fano width 13.7 e-): trade-off vs threshold sums; Naylor-weighted photometry would reduce it.
- F1: the ridge ratio (rho > 0.5) merges equal Gaussians up to d = 3.32 sigma_tot although bimodal above 2 sigma_tot
  (~3.5 px back side, 1.4 px front side); only peakfrac reveals such merges.
- F2: island BIC: heuristic chi2 likelihood, margin 30 not derived, asymmetry (chi2_free uses the global model incl.
  external clouds, H_xr refits only the island) biases toward merging; round H_xr penalises elliptical X-rays (CTI,
  column binning). Fix only with calibration on injected tracks.
- H: chi2 flag biased for sharp clusters (D1); ped slightly < 0 on empty background (-0.1..-0.14 e- at lam 0.01), and
  computed from Q (offset+dark included) in the 'unmodelled' regime; blend < 0.02 removes deep (diffuse) events
  preferentially in crowded fields -> energy-dependent efficiency (fine for line positions, needs correction for rates).
- I2: 'unmodelled' hysteresis energies are raw sums (not dark-subtracted), a different convention from GMX clusters.
- J2/J3 (agmx_root): A_GMX Energy is dark-subtracted (WADERs includes N_pix x lam); A_GMX STD_X/Y, DX/DY, npix are
  footprint sizes, not above-threshold pixels: WADERs cuts on them do not transfer (fSTD is comparable).
- Sound (verified): Poisson x Gaussian pixel model and variance; matched filter; exact Poisson box check; charge
  conservation and unbiased energies (also at lam=1); single dark subtraction everywhere; units; satellites, width
  ratio, CTI growth.

### 4.4 Test-harness physics
- Injected muons ~4x too faint (above). Injected tracks crossing real events (fixed in phase 1, footprint check).

## 5. A_GMX improvement study (agent, 2026-10-25) - status
Scripts/results: `DATA/next_steps_2026-10-25/agmx_improve/` (harness.py, subsets.py, variants.py, summ.py, out/).
| idea | status |
|---|---|
| numba EM (2x, identical) | **done** (phase 1) |
| pixel-likelihood few-electron finder | **done** (phase 1) |
| exact pixel integrals + consistent M-step | phase 2, in progress (section 3) |
| Gaussian background + GMX on 'unmodelled' (PD06/DM12) images: Fe-55 0.864 -> 0.960, pairs 0.148 -> 0.972, but the 59.5 keV line ~40 % wider | deferred; needs a high-energy energy estimate (threshold pixels for E > ~20 keV); interacts with audit fixes A1/I1 |
| fewer EM iterations | rejected (results depend on n_iter; see convergence) |

## 6. Next steps, in order
1. Resume and analyse the phase-1 validation runs (section 2); note that phase 1 changed GMX results only for
   few-electron clusters (new finder) and the track metrics (new placement).
2. Physics fixes: finish A2, G1 (+ realistic muon simulation), G2, D2 in `phase2b`; run the real-data test
   `p2b_real.py` (on vs off); if no regression, port A1, I1, C2, B1 (+ the finished ones) into the repository with
   unit tests (simulated: Gaussian-noise image, s=0.45 blank, hot column, lam bias) - user already asked to "fix any
   physics issues found".
3. Phase 2: run `p2_real.py` (base/exact), `crowd29.py` (fixed/converged + Lepy fit), `sim_sharp.py`; decide on exact
   integrals and a convergence criterion; if adopted, recalibrate flags and the island margin, re-check K-alpha.
4. Re-run the A_GMX performance tests with the final code (`*_v5`), then **append to docs/aGMX/aGMX_clustering.tex**:
   phase-1 changes and results, phase-2 validation, the physics audit (section 4 here, with fixes and what remains),
   the convergence caveat for the crowded CCD, the corrected injection tests; rebuild the PDF and Overleaf zip
   (`make_agmx_perf.py`, `am241_lines_flags.py fit` with LINES_OUT, `summ_wide_flags.py` with FLAGS_NEW,
   `make_agmx_contamination_figs.py`, `make_overleaf_zip.py`).
5. Later: high-energy energies on noisy images, charge-sheet model instead of the ped cut, track model, accelerated EM.

## 7. Practical notes (things that bit this session)
- Long runs: launch from a git worktree snapshot; macOS multiprocessing uses spawn, so workers re-import the scripts
  (edits during a run leak into it); configuration that workers need must be set at module level (see wide_run.py).
- Waiters: wait on a file (`until [ -f done.txt ]`), never on `pgrep` of a pattern that matches the waiter itself.
- Machine: 14 cores; keep total parallel processes <= ~12 (validation scripts use NPROC).
- `WIDE_OUT`, `VALIDATE_OUT`, `LINES_OUT`, `FLAGS_NEW` accept absolute paths (results go straight to DATA).
- Killing processes may need explicit user approval.
