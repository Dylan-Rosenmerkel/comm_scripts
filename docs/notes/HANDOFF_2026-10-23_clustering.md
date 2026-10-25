# Handoff: cluster thresholds and alternative clustering methods (session ending 2026-10-23)

> **Status (2026-10-25): historical, superseded by `CONTEXT_2026-10-24.md`.** B was installed (acbac7d); the clustering methods are now in comm_scripts (`clustering_tests/`, A_GMX via `run_WADERs --clusterer` / `bin/run_AGMX`). Open items: `OPEN_ITEMS.md`.

**The next step is to install B after checking, then move onto a review of the agents methods and pipeline and code and tests on real data.**

Everything from this session is in `~/Documents/comm_scripts_data/session_2026-10-23_clustering/`.
The original scratchpad is temporary. Scripts in the subfolders still contain hard-coded paths to it:
`/private/tmp/claude-501/-Users-dylanrosenmerkel/955da34a-e15a-4efb-8b0d-d9cf5445ae16/scratchpad/...`.
Replace that prefix with this folder before re-running anything.

## Ground rules (from the user, Dylan)
- Show proposed edits and get approval before changing `comm_scripts`. Experiment in a scratch folder first.
- Commit only when Dylan asks. Do not modify pysimdamicm/panaSKImg.
- For Fe-55 results, report front-side clusters by default (sigma_xy < 1). The back side is the topic here, because it is where clustering fails.
- Explain results plainly and do not overstate them.

## State of comm_scripts (`~/Documents/Software/comm_scripts`, branch `main`)
- `main` contains the reorganisation plus the coherent g(n) pooling. The pooling is commit `9550705`, merged in `0123ec5`/`00c06c1`, and makes `--gain-combine coherent` the default.
- **Nothing from this clustering work is in the repo yet.** The working tree was clean at the end of the session.
- Earlier docs: `docs/gain_correction_method/` (LaTeX note plus an Overleaf zip) and `PROGRESS_2026-10-23.md` (from the reorganisation session).

## The problem found
- WADERs/panaSKImg ClusterFinder keeps a pixel if it is > 10·σ and keeps a cluster if one pixel is > 30·σ (8-connected).
- σ is the median per-row *pedestal-fit width*, not the read noise. On noisy or high-dark-current images it is 5-15× the true noise.

| images | read noise | σ used | pixel / seed thr. |
|---|---|---|---|
| DM09 img 1-2 | 0.14 e- | 0.14 | 1.4 / 4 e- |
| DM10 img 1-2 | 0.20 e- | 0.73 | 7.3 / 22 e- |
| DM09 img 0 | 0.19 e- | 0.78 | 7.6 / 23 e- |
| DM10 img 0 | 0.33-0.38 e- | 1.8-2.1 | 18-21 / 53-62 e- |

- Diffuse back-side Fe-55 clusters (sigma_xy ≈ 1.4 px, brightest pixel ≈ 120 e-) lose their faint edge. The result is K-alpha 5-16 % low.
- Injection-recovery (simulated clusters of known charge added to the real images) reproduces the real deficit exactly. This is a clustering effect, not a gain effect.
- Threshold too low: dark-current pixels (Poisson, 0.2-1.7 e-/pix, growing along the readout) join clusters and push energies high.
  - Above ~41 % "on" pixels they percolate (8-neighbour site percolation).
  - At λ = 1.7, even ≥ 3 e- pixels make up 24 % of the background.
- Data used: `~/Documents/LSM_Module_Testing_Analysis/Production_Modules_Data/2025-08-27-DM07_DM09_DM10_Low_Temp/`, Image 5, DM09 (ACM 109) and DM10 (ACM 106), images `_0` (noisy), `_1`, `_2`.
- The WADERs default run is in `gain_curve_study_frontside/WADERs_analysis_Image5_DM09_DM10/` inside that data folder.

## Options tested (real WADERs reruns, coherent gain correction)
Back-side and front-side K-alpha / 5.899 keV over 21 image-CCDs (`thr_runs/analysis2.json`, plot `thr_runs/threshold_runs.png`):

| setting | back spread | worst | front spread |
|---|---|---|---|
| WADERs default | 4.6 % | 15 % low | 0.38 % |
| fixed pixel ≥ 2 e- (`--cluster-threshold-e 1.5 9.5`) | 2.3 % | 6.8 % high (DM10) | 0.51 % |
| fixed pixel ≥ 3 e- (`2.5 9.5`) | 1.8 % | 5 % | 0.30 % |
| A: per dataset (2 e- for DM09 img 1-2, 3 e- elsewhere) | 1.6 % | 5 % | 0.27 % |
| **B: `--cluster-threshold-e auto`** | **0.9 %** | **3.3 %** | **0.12 %** |

Note: panaSKImg keeps pixels strictly above the threshold. Pixel values are electrons ± ~0.2, so "≥ k e-" means a threshold of k − 0.5.

## B: `--cluster-threshold-e auto` (implemented, tested, NOT installed)
- Code is in `cs3/` (a copy of comm_scripts `main` with the change). Diff vs repo: `src/autoconfig_panaSKImg.py` (+99/−9) and the new `tests/test_cluster_thresholds.py` (5 tests).
- All 32 tests pass when run from `cs3/` (`docs/am241_note` was copied in so its test imports).
- Rule, per CCD:
  - λ = median per-row dark charge over all rows. It is taken from autoconfig's `measure_dark_current`, which now also returns `lam_all_median` and `lam_all_p90`.
  - Pixel threshold = smallest k ≥ 2 with P(Poisson(λ) ≥ k) ≤ α = 0.10, written as k − 0.5 e-.
  - Seed > 9.5 e-.
  - Resulting k: 2 for λ ≤ 0.53, 3 up to 1.10, 4 up to 1.74, 5 up to 2.43, 6 up to 3.15 e-/pix.
  - Unresolved CCDs keep the reference thresholds.
- Why the median: autoconfig's per-row λ does not mask clusters. Its 90th percentile was 1.6-1.8× the median on DM10 (rows with tracks).
- Chosen values on the test data: DM09 img 1-2 → 2 e-; DM10 img 1-2 → 3 e-; DM09 img 0 → 3-4 e-; DM10 img 0 → 6 e-. On DM10 img 0 it recovers the back side (0.99-1.00) and the cluster count, where every fixed threshold failed.
- Usage: `run_WADERs list.txt --gain-correction --autoconfig-args "--cluster-threshold-e auto"`.
- **Caveat: α = 0.10 was tuned on these same images** (α = 0.05 picks 4 e- at 1.0 e-/pix, about −1.5 % on injected diffuse clusters).
- **Check before installing:**
  1. Run it on at least one other dataset (another date/module, e.g. DM24-26 2025-09-13 or the AMT Am-241 images). Look at back-side Fe-55 and injection-recovery.
  2. Show Dylan the diff.
  3. Install into the repo, add a README line and a `run_WADERs -h -v` note.
  4. Commit only if asked.

## C: exploratory agent's methods (implemented by the agent; NOT yet reviewed)
Files are in `explore_agent/`:
- `agent_methods.py` (methods), `tests.py`/`tests.json` (its unit tests vs literature), `compare.py`/`compare2.log`, `summarize.py`
- `results_table.md` (latest results), `results_inject.csv`, `summary_inject.png`, `summary_real.png` (the `_v1` files are an earlier round)

The agent was stopped before its final report. There is no README of its checks, and its last rerun may be incomplete.

Methods (from its literature search):
1. Robust dark-charge map λ(row, col).
2. Per-row Poisson-tail threshold.
3. Matched-filter / likelihood-ratio detection with a fixed-radius footprint (from SoFiA, wavdetect, PowellSnakes).
4. Poisson maximum-likelihood (Cash statistic) cluster fit with λ fixed.
5. Profile-weighted (Naylor) photometry.
6. Multi-emitter fit for overlaps (DAOSTORM/PyBDSF style).

Pipelines combine these, e.g. 1-3-4-6.

What its partial tables show (DM09 EXT1 img0, dark charge 0.95 e-/pix, 220 injected clusters per width):
- The matched-filter-based pipelines (3 MF disk, 3+4 MF+Cash, 3+5 MF+Naylor, 1-3-4-6) recover E/E_true = 1.000-1.002 at every sigma_xy (0.3-1.8 px). Threshold methods reach 0.97-0.99 at 1.8 px; WADERs default 0.91.
- Pairs of clusters 5 px apart: MF+Cash and the 1-3-4-6 pipeline resolve them in 94-97 % of cases, where all threshold methods merge them. At 3 px, all methods merge.
- Not yet checked: real-data results, the other images (high dark charge DM10 img 0), correctness of the code against the references, runtime.

The agent's literature list (sources) is in this session's transcript, summarised in the "Candidate table" of its first report.
- Key references: Mighell 1999 (LSQ bias on Poisson data); Cash 1979; Smith et al. 2010 (MLE localisation); Naylor 1998; SoFiA 2; Hopkins 2002 FDR; Freeman 2002 wavdetect; DAOSTORM.
- Percolation threshold 0.407 (8-neighbour square lattice).

## Other material in this folder
- `clustering/`: my method comparison and injection framework.
  - `cluster_methods.py`: WADERs replica, topo/fixed thresholds, aperture, least-squares Gaussian fit, `inject`.
  - `run_study.py`: `load()` of the electron images, real-data and injection studies.
  - `auto_and_gfit.py`: auto-threshold scan and Gaussian-fit variants.
  - `clustering_comparison.png`.
  - Findings on the Gaussian fits:
    - Free-background least squares is good at low dark charge and biased low at high dark charge.
    - Fixed-background least squares was +2.5-3.9 % high on DM09 injections, so the per-row λ estimate there is suspect.
    - The agent's Cash fit is the intended replacement.
- `thr_runs/`: WADERs reruns with `--cluster-threshold-e 2.5 9.5`, `1.5 9.5` and `auto`, plus `analyse.py`, `analysis.json`/`analysis2.json` and `threshold_runs.png`.
- `cs3/`: comm_scripts copy with B.
- `coherent_test/`: coherent vs incoherent g(n) comparison.
  - g(n), A(n), peak-shift and back-side plots; `coherent_vs_incoherent.json` (module curves used everywhere above); `dm07_curves.json`.
- `compact_test/`: compact-vs-diffuse self-consistency test. After the gain correction a +0.2-0.6 %/1000 e- slope remained, the same for all g(n) forms, consistent with cluster-threshold loss. Repeat it with B's thresholds.
- `comb_regions/`: comb plots before and after the correction (`comb_regions_DM09_EXT4.png`), and the range-extension prototype that led to the coherent pooling.
- `nonlin/`: nonlinearity curves Δ(n) = A(n)/g(0) − n.

## Open items
1. Install B after checking it on other data (see above).
2. Review the agent's methods, pipeline, code and tests on real data:
   - read `agent_methods.py` against the references
   - rerun its comparison on all images (DM09/DM10 EXT1/EXT4, images 0-2), including real front/back K-alpha and runtime
   - compare with B
3. Repeat the compact-vs-diffuse test with the best clustering, to separate gain from clustering effects.
4. Coherent pooling on AMT EXT4 moved the Am-241 lines down by 0.4-0.7 %. Above about 2000 e- the curve is extrapolation only, so the high-charge gain remains the main uncertainty for Am-241.
5. The incoherent `--gain-combine` mode is still image-order-dependent (coherent, the default, is not).

## Update 2026-10-24
- **B installed** in `comm_scripts` (uncommitted) as an option, NOT the default:
  - files: `src/autoconfig_panaSKImg.py`, `tests/test_cluster_thresholds.py`, a README line; all tests pass
  - usage: `--autoconfig-args "--cluster-threshold-e auto"`
- **Independent check of B** (DM24-26, 2025-09-13, 9 Image 5s, 36 CCD-images), in `check_B/`:
  - back-side K-alpha 0.963 → 0.997 of true; spread 3.7 → 2.0 %; worst 9.2 → 6.0 %; front spread 1.85 → 1.64 %
  - high-dark-charge CCDs (DM26 EXT3/4, ~2 e-/pix) are still ~2.5 % low on the back side
  - DM25 EXT1 is off by +5-7 % in both runs, front included (a gain problem, not clustering)
- **Agent's code reviewed against the references:**
  - matched filter: Σw(Q−b)/√(Σw²var)
  - Cash fit: C = 2Σ(m − n ln m), gradient and Poisson Fisher matrix; Huang read-noise option
  - Naylor weights; deviance z-score
  - all match, and its unit tests pass (`tests.json`)
  - built-in risk: Cash and Naylor assume a pure Gaussian cluster profile
- **Final comparison** in `explore_agent/results_final/` (rerun on 2026-10-24 with B's installed rule as baseline (c)); tables in `results_table_final.md`, plots in `summary_*_final.png`:
  - **Injection, all 8 image-CCDs × 4 widths, worst bias:** WADERs 26 %, fixed thresholds 2.3 %, B 1.9 %, MF disk+grow 0.45 %, MF+Cash 0.38 %. MF methods: efficiency 1.00 vs 0.91.
  - **Real data, 12 images, mean |back − front| K-alpha:** WADERs 254 eV, fixed 43 eV, B 63 eV, MF disk 14 eV, hybrid 12 eV
  - back-side resolution: 81-87 eV (MF) vs 107-121 eV (threshold methods)
  - clusters kept relative to WADERs: 0.97-0.98 (MF) vs 0.78-0.82 (fixed/B)
  - **Absolute K-alpha / true** (images 1-2, then img 0):

    | method | images 1-2 front | back | img 0 front | back |
    |---|---|---|---|---|
    | MF disk+grow | 0.9965 | 0.9972 | 0.9973 | 0.9998 |
    | MF+Cash, 1-3-4-6 | 0.993 | 0.993 | 0.992 | 0.994 |

    The Cash fit runs ~0.5 % low on both sides, probably the Gaussian-profile assumption. B with fixed thresholds: 0.998 / 1.002 on images 1-2, but varies more on img 0.
  - runtime: MF 0.6 s per image, Cash pipelines 2.3 s per image
- **Next:** decide whether to bring the matched-filter detection + footprint ("3 MF disk+grow", or the 1-3-4/3-6 hybrid) into `comm_scripts` as an alternative clustering step after panaSKImg; check its profile assumption on Am-241 and tracks; commit B if Dylan wants.

## Update 2026-10-24 (later)
- **B committed** on `main` as `acbac7d` (optional `--cluster-threshold-e auto`; not the default; not pushed).
- **MFD+G tested on Am-241** (AMTEst 10 Image 8s, EXT1/3/4) and documented in `mfdg_am/`:
  - `MFDG_report.pdf`: method, diagrams, origins, performance; figures `fig_*.png`; scripts `am_test.py`, `make_report.py`; results in `results/`
  - injected 17.75 / 59.5 keV clusters: MFD+G 1.00 everywhere (WADERs down to 0.78 on EXT4)
  - real 59.5 keV: 0.997-0.998
  - BUT it splits tracks and big events (55-75 % of WADERs clusters above 80 keV contain more than one MFD+G cluster) and gives 2-5x more clusters on dense CCDs; the fragments bury the weak Np L lines
  - needs a track / big-event stage first before it could replace ClusterFinder on Am-241 or track-rich data

## Update 2026-10-24 (MFD+G follow-up: tracks, big events, few-electron events) -> `mfdg_tracks/`
- `MFDG_followup.pdf`: diagrams, injection tests, real Am-241 data, low-energy test
- methods are in `methods2.py`:
  - `track_first`
  - `mfdg_merge_B`
  - `mfdg_merge_saddle`
  - `mfdg_hybrid` (`ext_big=14`; `f_big=0.2` gives v2)
  - `counting_energy`
- Best: MFD+G + hybrid merge. On EXT1, injected (v1 / v2):
  - muon 0.83 / 0.97
  - worm 0.87 / 0.73
  - 4 px pairs 0.97, 6 px pairs 1.00
  - WADERs, for comparison: 0.87 / 0.93 / 0.22 / 0.82
- Track-first (option 1) is worse: it merges close pairs and does badly on dense EXT3.
- Few-electron events:
  - MFD+G at 5 sigma has 0 fakes and finds >= 5 e- events at 0.001 e-/pix; at 3 sigma it finds ~3 e- with a few fakes per Mpix
  - WADERs and B need 10-20 e-
  - counting-mode energy gets the exact charge 0.85-0.96 of the time (vs 0.55)
- EXT4 of AMTEst is not usable (40-90 % of pixels > 20 e-); distrust the EXT4 numbers in `mfdg_am/MFDG_report.pdf`.
- Nothing has been put into comm_scripts yet.
