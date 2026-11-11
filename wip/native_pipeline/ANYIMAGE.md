# Native pipeline on any image: characterisation front end, `--clusterer auto`, robustness suite

Status 2026-11-11, WIP (wip/native_pipeline/). Nothing in src/, bin/, utils/, docs/, wip/lowdc/bayes/ or
clustering_tests/ was changed.

## Files

| file | what | origin |
|---|---|---|
| native_pipeline.py | the pipeline, extended (below) | copy of src/native_pipeline.py at commit d6859c2f (md5 in `.upstream_md5.txt`; upstream unchanged when this was written); port as `diff src/native_pipeline.py wip/native_pipeline/native_pipeline.py` |
| native_kernels.py, native_rootio.py, native_plots.py | unchanged copies (imported from here, not from src/) | src/ at d6859c2f |
| characterise.py | NEW: the characterisation front end (`characterise`, `characterise_many`, `decide_ccd`, `compress_skips`, `skip_samples`, gain resolution, chain plan) | new |
| run.py | NEW: robustness suite (`python run.py [CASE ...] [-j 2]`, `--report-only`, `--composite CASE MODULE N`) | new |
| composite/ | MakeComposite copy (compiled) used by `run.py --composite` | unchanged |

Changes in native_pipeline.py (vs d6859c2f): `clusterer='auto'`; `auto_tasks()` (front end for a list of images:
records, gain resolution, groups, run-pooled dark charge); `process_image(..., fewe, fewe_pool, mask_cols,
characterisation)`; `agmx_clusters(..., fewe, fewe_lam, fewe_RN, bg, sigma)` (Bayesian switch, pre-computed background,
padding of images with < 61 rows); clustersRec branches `finder`, `p_event`, `N_post`, `sigma_post`; agmx_cls code
12 = 'few-e', agmx_regime 5 = 'bayes'; TNamed `characterisation`; `fit_limits()` (fit parameters at a limit -> summary
`fit_at_limit`); first-column mask on the clustering region only; strips: A_GMX valid-region fallback; CLI `--clusterer
auto`, `--gain`, `--gain-file`, `--fewe`, `--skip-start`; OUT/characterisation/<stem>.json.

```
PY=/opt/homebrew/Caskroom/miniforge/base/envs/damic/bin/python
$PY native_pipeline.py images.txt -o OUT --clusterer auto [--curve RUN/gain_vs_charge] [-j 2]
$PY native_pipeline.py l2_images.txt -o OUT --clusterer auto --gain 1.3          # no comb: gain given, flagged
$PY native_pipeline.py l2_images.txt -o OUT --clusterer auto --gain EXT1=1.3,EXT2=1.3,EXT3=1.3,EXT4=0.82
$PY native_pipeline.py imgs.txt -o OUT --clusterer auto --gain-file gains.json   # {EXTn: g} | {acm: {EXTn: g}} | native_summary.json
$PY characterise.py IMAGES... -o OUT [--gain ...]                                # front end only (JSON per image)
$PY run.py -j 2                                                                  # the suite
```

## 1. Front end (characterise.py)

Per file and per CCD, measured before clustering. Every item is `{value, method, confidence, ...}`; every fallback
is appended to a `fallbacks` list of the file or the CCD (nothing is silent). The record goes to
OUT/characterisation/<stem>.json (with the per-CCD result and decision added after processing) and, per CCD, into
each ROOT file as TNamed `characterisation` (file items + that CCD + `decision` + fitted gain / noise / DC).

| item | how | notes |
|---|---|---|
| data form | header NDCM / NSAMP / NSKIPS / ANSAMP and the data shape: width = NCOL x Nskips (panaSKImg layout, skips of a pixel consecutive along the row) or a 3-D cube -> `raw`; width = NCOL -> `averaged` (NDCM = 1: single skip); `var_` / `std_` names -> `std` (refused) | no real skip-resolved file exists on this machine (1872 FITS headers scanned): tested on a synthetic file |
| raw compression | `compress_skips`: mean and std over skips skip_start.. (CompressSkipperProcess; saturated = mean at max and std 0) -> OUT/compressed/<stem>.fits (NDCM = skips used, NATRAW, NSTART, NUSED) + `_std.fits` | `skip_samples(raw_path, ext, rows, cols)` -> (n_pixels, n_skips) per-skip ADU for the excursion veto |
| geometry | autoconfig.geometry (PHYSCOL - NCOLPRE + PRESCAN; --ccd-cols; else the reference JSON's 854, confidence low + fallback), prescan, first overscan column, overscan width, NPBIN x NSBIN, number of CCDs (ACM multi-extension), shape class (binned / strip / full width / small window) | |
| group (run setting) | ACM + NROW, NCOL, NDCM, NSAMP, NPBIN, NSBIN, NCOLPRE, EXPOSURE + file-name pattern (date/time/seq removed) + folder | used for sibling gains and run pooling |
| channel | autoconfig.is_dead_channel; a 'dead' CCD with a sharp zero peak (noise < 0.3 gain) is `quiet` (cannot be told from ultra-low dark charge) -> kept (`--keep-dead`), fallback logged; otherwise skipped | L2 103: all CCDs 'dead' by autoconfig (before: the src pipeline silently wrote nothing) |
| first column | median of the first active column minus its neighbours (per-row medians removed), significant if > 5 x its error and > 0.2 e- | measured 0.6-2.0 ADU on L2 103 (2000/NDCM = 2 ADU at NDCM 1000), 2.4-3.7 ADU on DM09 (4 ADU at NDCM 500), 4.5-7 ADU on PD06 -> column masked in the CLUSTERING region (not in the pedestal / fit: keeps DM09 identical) |
| saturation | pixels at the extension's maximum (>= 3 -> flagged) | reported only (none found) |
| gain | comb measurable = autoconfig resolved and gain from the comb. Otherwise, in this order: --gain, --gain-file (per ACM), siblings (median comb gain of the same CCD in the same run setting), reference prior. Non-comb gains: confidence low (none for the reference prior), `flag: GAIN NOT MEASURED ON THIS CCD`, fallback logged; passed to autoconfig as prior (--gain EXTn=..) and as gain_override | siblings tested on constructed records only (no real run with a mixed comb) |
| noise, dark charge, counting | autoconfig: read noise [e-], lam01 (Poisson-like peaks) or the rough mean charge; counting = resolved and noise < 0.3 e- | |
| rows / columns | first rows vs all rows dark charge (> 1.5 flagged: glow / charge along the readout); autoconfig bad (hot) columns | reported (hot columns still clustered, as before) |
| after calibration (`decide_ccd`, before clustering) | A_GMX's own numbers on the image A_GMX will cluster: AM.read_noise, AM.background_map (spread, lam), crowded.core_fill; the same bg and sigma are handed to adaptive_cluster | A_GMX's decisions and the front end's cannot disagree |

## 2. Decision table (characteristic -> chain)

| characteristic (per CCD) | chain | pedestal | calibration | fewe |
|---|---|---|---|---|
| form `std` / unknown | refused (logged) | - | - | - |
| form `raw` | compressed first, then as below | | | |
| no CCD signal, zero peak not sharp | skipped (logged) | | | |
| 'dead' but sharp zero peak (`quiet`) | as counting | autoconfig set_pedestal | --gain / file / siblings / prior (flagged) | auto |
| no resolved peaks and mean charge > 10 e-/pix (flooded, e.g. AMtest EXT2, 20x20 binned) | **threshold** clusterer, sigma0 rule (no background map, no A_GMX) | autoconfig (Gaussian fit when unresolved) | prior (flagged) | off |
| read noise >= 1 e-, or background spread >= 3 x read noise and >= 0.5 e- (A_GMX's test) | **unmodelled** (A_GMX's WADERs-like hysteresis on the calibrated image) | autoconfig | comb or flagged | off |
| ... and > 30 % of the pixels above 4.5 e- (flood guard) | threshold instead (logged) | | | off |
| electron counting, core fill >= 0.10 | **A_GMX crowded mode** | autoconfig (overscan median; lowest-peak start for a charged overscan) | comb fit (+ g(n) curve if given and reliable for the CCD) | off (agmx_switch leaves crowded images to A_GMX) |
| electron counting, core fill < 0.10 | **A_GMX normal** | as above | as above | **auto**; run-pooled dark charge (agmx_switch.prepare_run on all images of the run setting, same CCD) when the setting has >= 2 images and the dark charge <= 1.5e-2 e-/pix |
| not counting, noise < 1 e-, not flooded | threshold clusterer (thresholds auto -> sigma0 rule) | autoconfig | comb or flagged | off |

Logged warnings / fallbacks: core fill within 0.02 of 0.10 (mode may flip between CCDs of a run), fit parameters at
a limit (`fit_at_limit`), A_GMX padded rows, strip valid-region fallback, pooled DC unavailable, A_GMX regime
different from the front end's (never happened: same numbers).

## 3. Robustness suite (run.py; outputs in scratchpad/anyimage/suite/, report suite.md / suite.json)

All 16 cases rc 0, no open problem flags. Times with 2 cases in parallel (one process each).

| case | images | chain per CCD | key numbers | time |
|---|---|---|---|---|
| dm09 (DM09 Image_5, 3 images, g(n)) | averaged 480x1280 | A_GMX normal (core fill 0.053-0.066), fewe auto -> not used (f_c above the cut everywhere) | **identical to --clusterer agmx: 12/12 files, same energies and positions; K-alpha 5.8883 / 5.8826 / 5.8977 / 5.8968 keV in both** | 127 s |
| dm28 / dm27 / dm07 (Image_8 2025-09-23, first 2 images, module g(n) curves) | averaged 500x3500, overscan from PHYSCOL | **A_GMX crowded** (core fill 0.130-0.155 on every CCD) | K-alpha (front, 2 images) DM28 5.8917 / 5.8956 / 5.8958 / 5.8960; DM27 5.8889 / 5.8455 (EXT2, known bad comb) / 5.8897 / 5.8918; DM07 5.8776 / 5.8933 / 5.8834 / 5.8859 keV. Same images in the four-scenario root_agmx_gaincorr: DM28 5.8781 / 5.8817 / 5.8819 / 5.8825 etc.: the difference (+13 to +14 eV) is the native epsilon(T) = 3.7486 vs 3.74 eV/e- (x 1.002289 gives 5.8916 / 5.8952 / 5.8954 / 5.8960 for DM28, within 0.6 eV; DM27, DM07 within 0.2-0.9 eV except DM27 EXT2, 3.4 eV). Composite of all 30 images (g(n) + A_GMX): DM28 5.8857 / 5.8872 / 5.8899 / 5.8876 | 413-524 s |
| dm28, crowded off (check) | | A_GMX normal | K-alpha within 1 eV of crowded mode (5.8920 / 5.8964 / 5.8963 / 5.8952), front K-alpha N 2731 vs 2932 (crowded keeps ~8 % more) | 182 s |
| MakeComposite on dm28 auto output (`run.py --composite dm28 DM28 8`) | 8 files with the new branches + TNamed | | rc 0, energy_summary.csv front 5.8917 / 5.89556 / 5.89575 / 5.89599 keV (= the suite's fit), back 5.8926 / 5.8971 / 5.8834 / 5.9022 | 51 s |
| pd06_A / B / C (full readouts, images 8 / 14 / 20, EXT2-4, setting g(n)) | averaged 400x1500 | counting: A_GMX normal or crowded (core fill 0.094-0.120, i.e. at the cut) - **not 'unmodelled'**: background spread 0.16-0.18 e- = read noise; the earlier A_GMX + g(n) files of these images are also regime sparse / crowded | clusters image 8: 1003 / 1562 / 1189 (earlier files 1003 / 1549 / 1184); K-alpha single image 5.8967 / 5.8866 / 5.9064 (A), 5.8879 / 5.8906 / 5.8904 (B), 5.8950 / 5.8723 / 5.9114 (C); composites (6 images) 5.8844 / 5.8699 / 5.8755 (A) | 39-51 s |
| amtest (runs_29img image 21) | averaged 500x3500 | EXT1 A_GMX normal (fill 0.004), **EXT2 flooded -> threshold** (382 e-/pix, gain = reference prior, flagged), **EXT3 crowded (0.274), EXT4 crowded (0.665)** | 1360 / 6421 / 9692 / 17983 clusters | 242 s |
| l2_103 (100x500 NDCM 1000 pair + 3000 pair, 4x6294 and 12x2000 strips, --gain 1.3) | averaged | channels 'quiet' -> kept; counting (noise 0.036-0.106 e- at g = 1.3); A_GMX normal, fewe used, run-pooled for the pairs | 0 clusters (as wip/lowdc/bayes RESULTS section 7: no significant event at NDCM >= 500); pooled dark charge ~7e-6 e-/pix (prior-dominated: no dark electron seen); the comb fit's DC sits at its limits on most CCDs (flagged); column 0 masked (except EXT4) | 17 s |
| l2_103_ext4_g082 | EXT4, gain 0.82 | as above | noise 0.101 e-; 0 clusters (1 and 3 A_GMX groups replaced, 0 Bayesian events above the operating point) | 4 s |
| l2_103_nogain | no gain given | as above | gain 1.26 = reference prior, confidence none, flagged | 2 s |
| bayes_inject (two L2 NDCM 1000 images + 8 events per CCD, N = 3-10 e-, sigma 0.3-0.7 px, g = 1.3) | synthetic on real noise | A_GMX normal, fewe used, pooled | **58 of 64 events found as Bayesian clusters (finder 2, cls 'few-e'), all within 2 px, N_post - N_true = 0.00 +- 0.02, p_event 1.0**; branches written and read back | 8 s |
| binned (real 10x10 and 20x20, synthetic 2x2 from DM09) | averaged, binned | 10x10: EXT1/2/4 counting -> A_GMX (comb 1.33-1.37), EXT3 noise 5.7 e- -> unmodelled (gain prior, flagged); 20x20: flooded (170-210 e-/pix per bin, noise 13-19 e-) -> threshold (1-5 clusters); 2x2: A_GMX (60 rows -> padded, logged) | | 24 s |
| raw_skips (synthetic: 2 CCDs, 120 x 400 crop of DM09, 20 skips, 0.8 ADU per-skip noise, one 400 ADU excursion in one skip) | raw | compressed -> A_GMX normal | compressed - truth: mean -0.0003 ADU, std 0.1792 ADU (expected 0.8/sqrt(20) = 0.1789); skip_samples returns (1, 20) and finds the excursion (skip 3, +400.5 ADU) | 3 s |
| single_skip (NDCM = 1) | averaged | noise 3.5-3.7 e- -> unmodelled (gain prior, flagged; fit mu at its limit, flagged) | 42-90 clusters | 10 s |
| std_image (var_img) | std | refused, logged | | 0.5 s |

## 4. Failures found and fixes

| failure | where | fix |
|---|---|---|
| L2 ultra-low-DC images: every CCD 'no CCD signal' (autoconfig.is_dead_channel) -> the src pipeline wrote nothing and reported success | src process_image | `quiet` channel (sharp zero peak) kept with --keep-dead, logged |
| ultra-low DC / no comb: the gain silently = the reference JSON prior (L2 103, 20x20 binned, single skip, AMtest EXT2) | src process_image | gain resolution with source / confidence / flag; --gain, --gain-file, siblings |
| A_GMX crash on images with < 61 rows (12x2000 strip, 60-row binned image): `gmx.detection_bg` np.convolve 'same' over a 61-row window returns 61 values | clustering_tests/gmx.py (not edited) | agmx_clusters pads with invalid rows (logged); **should be fixed in gmx.py** (np.convolve(..., 'same') length = max(H, 61)) |
| 4-row strips: A_GMX's valid region (rows id_row_start + 3 .. H - 2) is empty -> no clustering, A_GMX said 'unmodelled' from nan noise | agmx_valid | rows id_row_start.. used, logged |
| AMtest EXT2 (flooded, 382 e-/pix): the 'unmodelled' hysteresis at 4.5 e- made one CCD-sized cluster; the process was killed (SIGKILL, memory) | adaptive_cluster unmodelled path | flooded rule (no comb and > 10 e-/pix -> threshold clusterer, no background map) + flood guard (> 30 % of pixels above 4.5 e-) |
| first-column artefact clustered | all L2 / ACM data (also DM09, PD06) | column masked from the clustering region only (DM09 stays identical to --clusterer agmx) |
| dark-current fit at its limits on ultra-low DC (lam = 1.3 ADU or 1e-4) reported as a result | gain_noise_dc | `fit_at_limit` in the summary and the characterisation |
| raw skip files refused ("only skip-averaged images are supported") | read_image | compressed first (CompressSkipperProcess), skip_samples |
| var_ (std) images would be processed as charge images | | refused |

## 5. Remaining limits / open issues

1. Fe-55 Image_8 CCDs (500 x 3500, ~5000 clusters per CCD) have core fill 0.13-0.155, so the 0.10 rule (also A_GMX's
   own crowded='auto') turns crowded mode on; crowded.py's docstring quotes 0.015-0.02 for Fe-55 production CCDs (DM09
   measures 0.05-0.07). K-alpha moves < 1 eV, crowded keeps ~8 % more front clusters. Whether crowded mode is wanted
   for dense Fe-55 images is a decision for the A_GMX owners (CROWD_FILL).
2. PD06 is NOT 'unmodelled' with the current background map (spread = read noise); its core fill sits at the crowded
   cut (0.094-0.120), so the mode flips between CCDs of one setting (warning logged). A hysteresis band or a run-level
   decision would make it stable.
3. Dark charge at ultra-low DC: the comb fit cannot measure it (at limits), AM.background_map has a ~1e-3 floor, the
   pooled Bayesian background is prior-dominated without dark electrons. The record gives all three; none is a
   measurement below ~1e-5 on 2 images.
4. Gains for L2 103 are unconfirmed (1.3 / 0.82 given by argument, flagged): no comb, no LED / high-DC image.
5. Skip-resolved raw data: only a synthetic file (no real one on this machine); id_skip_start = 0 as panaSKImg's
   reference (first-skip effects of real data not handled; `--skip-start`). Compressed copies are written into OUT.
6. Binned images: A_GMX / thresholds are not adapted to the bin size (diffusion per bin, charge per bin); the 20x20
   installation images are flooded and get only the sigma0 threshold rule.
7. Saturation and hot columns are reported, not masked.
8. PD06 setting C windowed images (17-19) give diffuse clusters (median sigma_xy 1.5 px vs 0.27 on setting A): no
   front-side K-alpha; the suite uses the full readout (image 20).
9. gmx.detection_bg (< 61 rows) and agmx_valid (strips) are worked around here, not fixed at the source.
10. Bayesian events in real data: none found (L2 noise test has none); the branch path is verified on injected
    events only. Excursion veto (per-skip) is another agent's work: `characterise.skip_samples` is the interface.
11. The sibling-gain path is tested on constructed records only.
