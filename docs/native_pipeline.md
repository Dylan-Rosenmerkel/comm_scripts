# Native skipper-CCD pipeline (no panaSKImg / pysimdamicm)

`native_pipeline.py` turns raw skip-averaged images (`.fz`, `.fits`, `.fits.bz2`) into cluster files in the
panaSKImg format (`panaSKImg_clustersRec_<image>_ext<n>.root`, trees `clustersRec`, `info`, `process_config`),
which MakeComposite and the comm_scripts analysis code read unchanged. It replaces the cluster-producing part of
`bin/run_WADERs` (autoconfig + gain_dc_split + panaSKImg runs + g(n) + agmx_root).

Status: the default engine of `bin/run_superWADERs` (= `bin/run_WADERs --engine native`; `bin/run_WADERs` keeps
panaSKImg as its default, and both commands take `--engine native|waders`). Validated
against panaSKImg on the 3 DM09 Image_5 images (native_pipeline_validation.md), and on the board-103 skip-noise images
with an overscan (2026-11-10: gain and noise equal to panaSKImg's on 80 CCD-images, median difference 0.00 %; up to
2.8 % on unresolved 100-skip CCDs, where panaSKImg used a Gaussian-fit pedestal).

Raw DAQ files (`src/raw_data.py`; native engine only; readers copied from ccd_opt_ml into `src/raw_levels/`,
`utils/sync_from_ccd_opt_ml.py [--check]`):

- **level-0 traces** (`trace_*`, `psd_*` `.RJ45ADCDATA<ch>`): decode_trace (raw samples, window flags, inline
  per-skip sums; check: inline sums = sums of the flagged raw samples), then `L0/<trace>/overview.png`
  (ccd_opt_ml's plot_trace: raw cycles, per-skip window means, folded cycle, noise spectrum; for psd_ traces without
  windows: the samples and their spectrum), `viewer.html` (view_trace; `--l0-no-html` to skip) and `summary.json`.
  Channel files that hold only the keyword words (a channel not recorded) are listed as such. Not images: they do
  not enter the image pipeline.
- **level-1 images** (`*_1x1_L1_*.RJ45ADCDATA<ch>` + `.meta`): read row by row (memory map, decode_l1's layout and
  checks), one FITS image per N of `--l1-skips` (auto: 1, 2, 5, 10, ... and all; the mean of skips K..K+N-1,
  `--l1-skip-first K`, default 0 as an image taken with N skips) with the DAQ header from the `.meta` (NDCM = N,
  NROW = complete rows, L1SRC / L1SKIP0 / L1NSKIP / L1ALLSK) and one extension per channel (a missing channel: zeros,
  dropped later as "no CCD signal"). They join the run's images (every CCD is processed, also unresolved ones, as
  for a skip study) and afterwards `L1/noise_vs_skips.csv/.png` (noise, gain per CCD against N) and
  `L1/compare/<image>/` (compare_dcfit_vs_skips: the DC fit, baseline and sigma of every N side by side) are made.
  Rows cut by the end of the file are left out; pixels failing a layout check get their row's median.
  Check (2026-11-10): synthetic level-1 file (decode_l1.write_l1) from a 25 x 250 crop of a DM09 Image 5, 4
  channels, 1000 skips, 4 ADU per-skip noise of zero mean per pixel: N = 1000 reproduces the image to 2e-4 ADU
  (float32), N = 10 / 100 leave 1.262 / 0.381 ADU (expected 1.265 / 0.379); end to end through run_superWADERs
  (with an L0 trace and a psd trace) in 9 s. The real level-1 files on this machine (installation check
  2026-10-06) are cut after 80 pixels (no complete row) or binned (20 x 20): both reported, no image made.

In run_WADERs (`run_native`): with --gain-correction the fits run first without cluster files (each image's gain
for the gain periods), g(n) is measured from the raw images as with panaSKImg (`measure_gain_curves`), then one
process per image writes root/, root_gaincorr*/, root_agmx*/ from the same fits (`process_image(outputs=...)`).
The files are in comm_scripts/src (native_pipeline.py, native_kernels.py, native_rootio.py); the validation script
is utils/validate_native.py.

## Usage

```
PY=/opt/homebrew/Caskroom/miniforge/base/envs/damic/bin/python
# WADERs-like clusters, thresholds in electrons from the dark charge (default), g(n) from a run_WADERs folder
$PY native_pipeline.py img1.fz img2.fz img3.fz -o OUT -j 2 --curve RUN/gain_vs_charge
# A_GMX clusters
$PY native_pipeline.py list.txt -o OUT -j 2 --clusterer agmx --curve RUN/gain_vs_charge/module_109_gain_vs_charge.json
# both (OUT/root/ and OUT/root_agmx/)
$PY native_pipeline.py list.txt -o OUT --clusterer both
# panaSKImg's own threshold rule (10 / 30 x the pedestal sigma)
$PY native_pipeline.py list.txt -o OUT --thresholds sigma0
# fixed thresholds in electrons (pixel, seed)
$PY native_pipeline.py list.txt -o OUT --thresholds 2.5,9.5
# reproduce a run_WADERs run (its JSON settings) - validation mode
$PY native_pipeline.py list.txt -o OUT --panaskimg-json RUN/json --curve RUN/gain_vs_charge
```

Options: `-j N` images in parallel (separate processes; keep N <= 2 on the laptop, A_GMX uses ~1 core per
process), `--ext 1 2` CCD subset, `--anchor comb|fit` (g(n) anchor, as gain_vs_charge), `--ev-per-e` (energy per
electron of the written energies), `--crowded auto|on|off` (A_GMX crowded mode), `--save-fits` (calibrated image
[e-] + a cluster mask, `<image>_EXT<n>_native.fits`), `--autoconfig-args "..."` (geometry / gain prior options of
autoconfig_panaSKImg, e.g. `--ccd-cols`, `--gain`). The output folder also gets `root_files.txt` (input list for
MakeComposite) and `native_summary.json` (per CCD: gain, noise, dark current, fits, thresholds, cluster counts,
timings).

Library: `native_pipeline.process_image(path, out_dir, curve=None, clusterer='threshold', thresholds='auto', ...)`
returns the same summary dict.

`--curve` takes a `gain_vs_charge measure` JSON or a run_WADERs `gain_vs_charge/` folder (module curve of the
image's ACM / gain period, `agmx_root.curve_file`). Measuring g(n) itself (gain_vs_charge.py measure) is not part
of this pipeline; it already works on raw images without panaSKImg.

## What it does, per image (file read once, all CCDs in one process)

1. **measure** (reused: `autoconfig_panaSKImg.analyse_extension`): comb gain, read noise, per-row dark charge,
   bad columns, rows to fit, resolved or not; the fit settings autoconfig --per-ext would write
   (`fit_settings`, `set_dark_current_fit`) for the two fits of gain_dc_split ("all peaks" and "Poisson-like peaks").
2. **pedestal**: per-row median of the active region (of the overscan if the image has one and the reference JSON
   asks for it) after masking pixels above median + 1.5 x MAD (MAD unscaled) - panaSKImg's PedestalSubtractionProcess
   method "median". The per-row std of the kept pixels is the pedestal sigma.
3. **gain, noise, dark current**: panaSKImg's FitDarkCurrentProcess model (Gaussian peaks of width sigma, spacing
   gain, Poisson weights of mean lambda; chi2 with Neyman errors, empty bins dropped, model at the bin centres, same
   window, binning, limits and masks), fitted with scipy's bounded least squares. Both fits of gain_dc_split are made
   and `gain_dc_split.choose_gain` (reused) decides whether the many-peak gain is held; a held gain is fixed in a
   final fit. ~10 ms per fit.
4. **calibration**: e- = (pixel - pedestal) / gain.
5. **g(n)** (optional): ADU -> electrons with `gain_vs_charge.electrons_with_anchor` (reused), energies with
   epsilon(T) (`pair_energy.pair_energy_eV`, 3.7486 eV at 130 K); without a curve 3.74 eV (panaSKImg's constant).
6. **clusters**:
   - `threshold`: the WADERs ClusterFinder (zero outside the active region, pixel > T_pix, 8-connected
     `ndimage.label`, keep groups with a pixel >= T_seed), with panaSKImg's cluster and pixel order.
   - `agmx`: `clustering_tests/adaptive.adaptive_cluster(engine='gmx')` (reused) on the calibrated or g(n) image,
     active pixels as agmx_root (rows from id_row_start + 3, 2-pixel border), agmx_* branches and the
     `agmx_config` TNamed as agmx_root writes them.
7. **cluster quantities** (`native_kernels.py`, numba, all clusters at once): every clustersRec branch of panaSKImg with
   the formulas of pysimdamicm's `Cluster` (positions, widths, PCA, Qmax, projections, Gaussian profile fits,
   track end points, seed quantities, closest cluster, pixel lists ...), summed in numpy's pairwise order so the
   values agree to the last bit (this matters: `closest_cls_*` skips neighbours with an exactly equal PosX/PosY,
   and the sign of a ~0 covariance flips the PCA axes).
8. **output** (`native_rootio.py`): PyROOT imported only here; numpy arrays copied into std::vector branches by small C++
   helpers (one tree entry per image, as panaSKImg).

Physics choice: the default thresholds. panaSKImg thresholds at 10 / 30 x sigma0, sigma0 = mean per-row std of the
masked pedestal pixels. That width is the 0e-/1e- mixture, not the read noise: 0.13 e- at a dark charge of
0.17 e-/pix but 0.75-0.79 e- at ~0.5 e-/pix (DM09 image 0), so the pixel threshold jumps from 1.3 e- to 7.5 e- and
the seed from 4 e- to 23 e- with the dark charge. In a skipper image the pixels around a cluster hold dark-current
electrons, so the native default (`--thresholds auto`) uses the rule of autoconfig `--cluster-threshold-e auto` /
ClusterFinder `spr_threshold_auto`: pixel >= k e- (threshold k - 0.5) with k the smallest integer >= 2 for which
P(Poisson(lambda) >= k) <= 10 %, lambda = median per-row dark charge; seed >= max(9.5, k + 0.5) e-. With resolved
peaks the read noise is < 0.3 e-, so k - 0.5 >= 1.5 e- is also >= 5 sigma of read noise; CCDs without resolved
peaks fall back to the sigma0 rule.

## Files

| file | content |
|---|---|
| native_pipeline.py | driver, pedestal, dark-current fit, gain choice, clustering, cluster table, CLI |
| native_kernels.py | numba kernels: moments / widths / PCA / profile fits / track end points / closest cluster |
| native_rootio.py | ROOT writer (lazy PyROOT import) |
| utils/validate_native.py | comparison with a run_WADERs (panaSKImg) run; writes validation.json / validation_tables.md |
| native_pipeline_validation.md | results on the DM09 Image_5 run |
| composite/ | copy of MakeComposite_Fe55_Clusters_Extensions.cc used for the compatibility test (compiled here) |

Reused from comm_scripts (imported, not copied): `src/autoconfig_panaSKImg.py` (measurements, fit settings,
auto thresholds), `src/gain_dc_split.py` (choose_gain), `src/gain_vs_charge.py` (g(n)), `src/agmx_root.py`
(curve_file only), `src/naming.py`, `src/pair_energy.py`, `clustering_tests/adaptive.py` (A_GMX).

## What differs from panaSKImg, and why

- **Gaussian profile fits (fSTD_X/Y, fPosX/Y)**: same likelihood (ROOT "L" on the weighted projection, InitGaus
  start and its sigma limit [0, 10 rms], mean limited to [minX, maxX] and fixed when equal), but minimised to
  convergence with a Newton solver from 4 starting widths. Minuit stops at its EDM tolerance, which leaves sigma
  off by ~1e-3 relative on flat likelihoods and up to a few 0.1 px on 2-column and single-column clusters (where
  the likelihood is degenerate and Minuit stops anywhere in [0, 0.3]); on the CCD checked (1205 differing fits)
  the native value has the lower or (2 cases, within float32 rounding) equal likelihood. See native_pipeline_validation.md for the effect on the front/back (sigma_xy < 1) split.
- **Dark-current fit**: scipy least squares instead of Minuit on the same chi2: the same minimum within a small
  fraction of the error (native_pipeline_validation.md). A held gain is fixed exactly (panaSKImg: +-0.001 % window); the info tree
  then gets the held value's error (many-peak fit, scaled by sqrt(chi2/ndf)) instead of Minuit's ~0 error.
- **Calibration gain**: panaSKImg's CalibrationProcess divides by the fitted gain rounded to 4 decimals; the native
  default uses the fitted gain (the rounding is applied only in the --panaskimg-json mimic mode).
- **Cluster shapes with g(n)**: panaSKImg computes PosX, widths, PCA, Qmax, fSTD ... from the energies BEFORE the
  g(n) correction and replaces only pixels_E / Energy afterwards. The native default computes them from the
  written (corrected) energies, so Qmax and pixels_E are consistent (mimic mode: as panaSKImg). has_seed and
  seed_nsig_* use the uncorrected energies in both (they belong to the threshold, which is applied on the image
  calibrated with the fitted gain).
- **A_GMX energies**: panaSKImg's agmx run (and agmx_root) converts the g(n) electrons to energy with 3.74 eV;
  the WADERs+g(n) files use epsilon(T) = 3.7486 eV. The native pipeline uses epsilon(T) for both clusterers
  whenever g(n) is applied (+0.23 % on the A_GMX energies relative to panaSKImg's root_agmx files).
- **Thresholds**: default `auto` (above); `--thresholds sigma0` gives panaSKImg's rule.
- **Pedestal methods** (`pedestal()`): all of PedestalSubtractionProcess's row methods with the settings
  autoconfig chooses per CCD (`set_pedestal`): median / mean (sigma = std or MAD), and the Gaussian fit per row
  (`gaus_fit`: default / freedman / sturge / manual bins, mu in the 10-90 % quantiles, sigma in 0.5-1.5 std;
  scipy least squares), with the lowest-peak start for a charged overscan (`is_highDC`: the first local maximum of
  the histogram). Tail masks (cut_right_tail / cut_left_tail, n_sigma_to_mask) and the fallback to the active
  region for < 3 overscan columns as panaSKImg. Checked per row against panaSKImg's info tree (2026-11-10):
  gaus_fit (freedman, 8 unresolved 100-skip CCDs, board 103): |diff| median 1e-5 ADU, max 1.6e-4 ADU (Minuit's
  tolerance); lowest peak (L2 surface run, EXT2 of 2 images): 996 of 1000 rows within 1e-4 ADU, the 4 others
  rows with a large track in the overscan (sigma 6-221 ADU). For the lowest-peak case the peak search and the fit
  are done with ROOT itself (TH1::Interpolate / TF1::Derivative / Fit 'Q EM B'): with ~150 overscan pixels in 150
  bins the derivative is 0 between equal bins up to floating-point residue (whose sign decides; a Python replay
  agrees on 79 % of the rows) and the chi2 has several minima (which one panaSKImg reports is where Minuit stops).
- **--tune** (`tune_ccd`): autoconfig's tune_pedestal + refine_peaks with native pedestals and fits instead of
  panaSKImg runs: the same candidates (chosen / median / Gaussian fit or lowest peak), score (fit_quality total
  relative error + 1 per parameter at a limit), wide scan and window rules (autoconfig.peaks_in_histogram,
  apply_peak_window, gain_dc_split.sync_from_tuned). L2 surface run (2 images, 6 CCDs): same candidate chosen on
  every CCD, scores equal to 3 decimals, same windows, same final fits; 9 s instead of 58 s.
- **Not implemented**: polynomial calibrations, skip (non-averaged) images, ClusterFinder options
  other than the defaults used in run_WADERs (method 1 k-radius, split_image_in_rows, masks, moskita),
  BuildClusterMask (only an optional simple dilated mask with --save-fits) and the `*_waders.fits.bz2` images.
- **Plots** (`--plots`, always on in run_WADERs; `native_plots.py`): panaSKImg's --save-plots PDFs with the same
  names: `_FitDarkCurrentProcess_DCfit_EXTn.pdf` (same numbers as panaSKImg's on DM09: chi2/ndf 521.9 / 87, gain
  1.3409, sigma 0.1846 ADU; the dotted continuation beyond the fit window covers every plotted peak, not only the
  next one) and `_PedestalSubtractionProcess_Baseline/Sigma_EXTn.pdf` (per row). matplotlib, ~0.1 s per plot.

### Branches

clustersRec: every branch of the panaSKImg file (checked: no branch only in one of the two files), with the same
types (vector<float>, vector<int>, vector<vector<float>>, vector<vector<int>>), units (keV, 1-based pixels, ADU for
*_PS / *_AVG) and meaning; A_GMX files add the agmx_* branches. MakeComposite reads pixels_x/y, pixels_E, PosX/Y,
wSTD_XY, fSTD_X/Y, Energy, Qmax, Npix (+ agmx_cls) and from `info` ext<n>_gainext / sigext / dcext (+ _err).

info: all branches of the panaSKImg file (RUNID, EXT<n>_pedestal_mu/sigma_row(_mean)_ADU, ext<n>_dcext /
gainext / mu0ext / sigext (+ _err; ADU, as panaSKImg), idccd, nfile, readout_start / end, total_exposure,
readout_time_row).

process_config: deliberately reduced. Written: id, e2eV (3.74, the eV/e- of UNCORRECTED energies, as panaSKImg;
gain_vs_charge.apply and agmx_root rely on it), native_eV_per_e (the eV/e- actually used for Energy),
FitDarkCurrentProcess_chi2 (likelihood chi2/ndf, as panaSKImg), _chi2_neyman, _ndf, _dc_norm/_mu0/_sigma/_lambda/
_gain, _included_pixels, _total_pixels, _n_bins_used, _binning_size_used, _x_min, _x_max, _n_peaks,
ClusterFinder_pixel_threshold_e, ClusterFinder_seed_threshold_e, native_version. Omitted: the ~90 other panaSKImg
process settings (ADC2e, ADC2eV, BuildClusterMask_*, CalibrationProcess_*, ClusterFinder_* settings,
CompressSkipsProcess_*, FitDarkCurrentProcess_* limits and guesses, GainNonlinearityProcess_*,
PedestalSubtractionProcess_*, img_exp_time, waders_version, waders_commit): nothing in comm_scripts or MakeComposite
reads them; the full native configuration is in the TNamed `native_config` (JSON).

## Speed

DM09 Image_5 (3 images x 4 CCDs, 480 x 1280), measured (native_pipeline_validation.md section 5): threshold clusterer 5.6 s end to
end with -j 2 (7.1 s with -j 1); in a running process ~0.3 s per CCD (pedestal 25 ms, fits 10-40 ms, clustering
13 ms, cluster quantities 0.15-0.3 s, ROOT writing 50 ms) plus 0.5-0.6 s per image for autoconfig's measurements;
~1 s per process for imports and the numba cache (first run after a cache wipe: +5 s compilation), +0.8 s for the
ROOT import at the first file. A_GMX: 59.8 s with -j 2, i.e. adaptive_cluster's ~6.6 s per CCD. panaSKImg: 7.1 s
per CCD for the threshold run and 13.6 s for the A_GMX run (one process per CCD), plus gain_dc_split's fit-only
runs before.

## Open issues

- fSTD from Minuit cannot be reproduced exactly (see above); the native value is the converged one.
- Pedestal methods and --tune: see above (all of panaSKImg's row methods; checked per row against panaSKImg).
- The clustering region is the whole active region, as panaSKImg (bad columns and the first rows are only masked
  in the dark-current fit). Hot columns therefore still give clusters, as before.
- Unresolved CCDs (no electron counts) keep the sigma0 threshold rule in `auto` mode.
- A_GMX speed is that of clustering_tests/adaptive.py (~6.5 s per 480 x 1280 CCD, about 1.8 s of it the
  background map); not changed here.
- Only tested on DM09 Image_5 (no overscan, windowed readout). Images with an overscan / prescan use autoconfig's
  geometry and the overscan pedestal path, which has not been validated against panaSKImg yet.
