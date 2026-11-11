# comm_scripts

Scripts to analyse skipper-CCD images with panaSKImg (WADERs) without writing JSONs by hand.

**User guide (every part of the repository and how to run it):** `docs/usage/comm_scripts_usage.pdf`
(LaTeX source `docs/usage/comm_scripts_usage.tex`, Overleaf upload `docs/usage/comm_scripts_usage_overleaf.zip`).

## Layout

```
comm_scripts/
  README.md
  bin/     commands to run          run_WADERs (and the alias runWADERs), run_superWADERs (run_WADERs with the native
                                    engine as default, plus the commissioning tools), run_AGMX (A_GMX clusters for a
                                    finished run)
  src/     the pipeline modules     autoconfig_panaSKImg.py, gain_dc_split.py,
                                    gain_vs_charge.py, fit_quality.py, agmx_root.py (A_GMX -> panaSKImg cluster files);
                                    native engine and commissioning tools of run_superWADERs (see that section)
  utils/   standalone helpers       compare_dcfit_vs_skips.py, split_clustersRec_by_ext.py,
                                    skip_noise_L1.py, compare_runs.py
  clustering_tests/  research code   alternative clustering methods (A_GMX, GMX, MHC, MFD+G, ...) and the
                                    scripts that tested them; A_GMX is used by run_WADERs --clusterer / run_AGMX
  tests/   unit tests               python -m unittest discover -s tests -v
  docs/    guide, notes, reports    README.md (index: what to read for what), usage/ (the user guide, LaTeX + PDF),
                                    PIPELINE_GUIDE.md (the physics and code of each step), aGMX/ (A_GMX clustering),
                                    clustering_methods_report.pdf, gain_correction_method/, am241_note/,
                                    report_scripts/ (+ the two gain-correction PDFs they make),
                                    notes/ (hand-over notes, OPEN_ITEMS.md), patches/, data/
```

Keep `bin/`, `src/` and `clustering_tests/` together: `run_WADERs` and `run_AGMX` load the modules from `../src`, the
`src/` scripts import each other from their own folder, and `src/agmx_root.py` imports A_GMX from `../clustering_tests`. `utils/` scripts are independent, except
`skip_noise_L1.py`, which uses the gain finder in `src/`.

New to the code? Start with the user guide (`docs/usage/comm_scripts_usage.pdf`: what everything does and how to run
it), then [docs/PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md): the data flow, the physics behind each step, and which
function does what. Everything still open: `docs/notes/OPEN_ITEMS.md`.

## Quick start

```bash
/path/to/comm_scripts/bin/run_WADERs image.fz          # one image
/path/to/comm_scripts/bin/run_WADERs my_images.txt     # a list, one image path per line
/path/to/comm_scripts/bin/run_superWADERs my_images.txt   # the same without panaSKImg (native engine)
```

`run_superWADERs` is `run_WADERs` without panaSKImg (its own "native" engine), plus the commissioning tools: raw
L0 / L1 files, commissioning packet reports, dark current over a series, defect maps and serial-register images.
See [run_superWADERs](#run_superwaders-native-engine-and-commissioning-tools) below.

Results go to `WADERs_analysis_<image or list name>/` in the current folder (`-o` to choose another):

| what | where |
|---|---|
| panaSKImg plots (PDF) and processed images | top folder |
| gain, noise, dark current per CCD, with fit errors | `summary.txt`, `fit_quality.csv` |
| cluster ROOT files, one per CCD | `root/` |
| full paths of those files (e.g. for `MakeComposite_Fe55_Clusters_Extensions.cc`) | `root/root_files.txt` |
| dark-current fits as ROOT | `root/dc_fits/` |
| the JSONs that were run, one per CCD | `json/<image>/final/` |
| logs | `logs/` |
| with `--composite`: MakeComposite plots, one folder per image type and module | `composite/Fe_55_Analysis_Plots/<module>/Image_<N>_<module>_list/` |
| with `--composite`: the file lists given to MakeComposite | `composite/lists/` |

`run_WADERs -h` lists the flags; `run_WADERs -h -v` prints the full guide. Useful options:
`-j N` (parallel panaSKImg runs, default 4), `--overwrite`, `--no-clusters` (noise, gain and
dark current only; much faster), `--tune` (check each CCD's pedestal method and fit window
with panaSKImg, by smallest total fit error; ~1 min more per image),
`--autoconfig-args "--cluster-threshold-e auto"` (optional, not the default: cluster thresholds in
electrons chosen per CCD from its dark charge - the smallest k with P(Poisson(lambda) >= k) <= 10 %,
pixel > k - 0.5 e-, seed > 9.5 e-; on Fe-55 back-side (diffuse) clusters it removed most of the 4-9 %
low bias of the default thresholds, see docs/notes/HANDOFF_2026-10-23_clustering.md),
`--autoconfig-args "--cluster-threshold-e 2 5"` (cluster thresholds in electrons; faster on
images with high dark current), `--all-ext` (also run flooded/unresolved CCDs),
`--gain-correction` (measure gain vs charge per pixel and write gain-corrected cluster files;
see `src/gain_vs_charge.py` and the reports in `docs/`), `--composite` (then run
`MakeComposite_Fe55_Clusters_Extensions.cc` with ROOT on the cluster files, gain-corrected ones if
`--gain-correction` is on, once per image type and module; module
names from `--module 109=DM02` or a `..._DMa_DMb_DMc_...` data folder; macro from
`--composite-macro` or `$MAKECOMPOSITE_MACRO`).

Gain correction options (put the input first: `run_WADERs images.txt --gain-correction both`):

| option | what it does | output |
|---|---|---|
| `--gain-correction` or `--gain-correction pooled` | one curve per module (ACM) from all its images | `root_gaincorr/` |
| `--gain-correction per-image` | each image corrected with its own curve; a CCD whose own comb is not usable (e.g. a noisy image) gets the module curve, noted in `summary.txt` | `root_gaincorr_per_image/` |
| `--gain-correction both` | both, and a comparison per image and CCD (curve, cluster energies, front-side Fe-55 K-alpha) | both folders, `gain_vs_charge/per_image_vs_pooled.csv`; with `--composite` also `composite_per_image/` |
| `--gain-model interpolated` (default) / `broken` / `linear` | form of g(n): monotone interpolation through the measured points, two straight lines with a fitted knot, or one straight line | |
| `--gain-combine incoherent` (default) / `coherent` | how several images of a module are combined for the module curve | |
| `--gain-memory off` (default) / `auto` | remove the horizontal readout memory (a pixel keeps ~3e-4 of the pixel read before it, fitted per CCD) before measuring g(n); the comb stays visible to higher charge | `memory` in the curve JSON |
| `--gain-anchor comb` (default) / `fit` | low-charge gain from the comb, or each image's fitted gain | |

Curves and plots are in `gain_vs_charge/` (`module_<ACM>_...` and `image_<image>_...`): the
local gain `*_gain_vs_charge.png` and the nonlinearity curve `*_nonlinearity.png` (measured charge
minus peak number against measured charge, `A(n)/g(0) - n`, reference gain g(0) of the curve; its
quadratic fit is also in the JSON and in `summary.txt`). Several
images of a module are combined incoherently by default (`--gain-combine incoherent`: comb powers
added, independent of the image order). `--gain-combine coherent` converts each image to electrons
with its own curve, aligns the images' teeth and pools all pixels into one comb, which reaches higher
charge; it is not the default because on the Am-241 lines (AMtest EXT4, 2026-10-26) it split the two
gain periods by 0.6 % (on Fe-55 it was as good or slightly better).
On the 2025-08-27 Image 5s the per-image and pooled curves gave the same front-side K-alpha to
within a few eV whenever an image's own comb was usable.

## run_superWADERs: native engine and commissioning tools

`run_superWADERs` = `run_WADERs --engine native`: comm_scripts' own pedestal, fits, clustering and cluster files in
panaSKImg's format (`src/native_pipeline.py`), one process per image, with the same options, output folders and
`summary.txt`. `run_WADERs` still runs panaSKImg by default; `--engine native|waders` switches either command. It
needs the `damic` conda environment (found automatically, as for `run_WADERs`).

```bash
run_superWADERs INPUT [-o PARENT] [-j N] [--overwrite] [options]    # INPUT: one file, or a .txt list (one path per line)
run_superWADERs -h                                                  # every flag;  -h -v: the full guide
```

Results: `PARENT/WADERs_analysis_<input name>/`; read `summary.txt` first. The full user guide, with what every
number and plot means and what data are still needed to validate each tool, is
[docs/superWADERs_user_guide.md](docs/superWADERs_user_guide.md).

### Inputs

| file | what it is | what happens to it |
|---|---|---|
| `avg_*.fits`, `.fits.bz2`, `.fz` | averaged images from the DAQ (any size, binning, number of skips) | image pipeline: pedestal, fits, clusters |
| `trace_*.RJ45ADCDATA<ch>` (+ `.meta`) | level-0 trace: raw video samples of one channel | decoded, checked, plotted (`L0/`) |
| `psd_*.RJ45ADCDATA<ch>` (+ `.meta`) | level-0 noise trace (no skip windows) | samples and noise spectra |
| `*_L1_*.RJ45ADCDATA0..3` + `.meta` | level-1 image: every skip of every pixel (binned or not) | FITS images with N skips each, then the pipeline (`L1/`) |

Files in a list may come in any order: the type is read from the file, the four channel files of an L1 image are
grouped by name. Give averaged (L2) images as their `avg_*.fits`.

**Module names** (`--module-list FILE`): a correlation list, one line per ACM (or per CCD when one board carries
CCDs of different modules); comma, tab or space separated, `#` comments and a header line are ignored:

```
# ACM, module[, EXT<n> or CCD name]
101, DM24
102, DM25
103, PA03, EXT2
```

Without an entry the name comes from `--module ACM=NAME`, then the data folder name (`..._DMa_DMb_DMc_...`), else
`ACM<id>`.

### What to run

| task | command | output (in the run folder) |
|---|---|---|
| ordinary analysis without panaSKImg | `run_superWADERs images.txt [--gain-correction] [--clusterer both] [--composite]` | as run_WADERs |
| L0 traces / PSD traces | `run_superWADERs traces.txt [--l0-no-html]` | `L0/<trace>/`: `overview.png`, `viewer.html`, `summary.json` |
| L1 images at several skip counts | `run_superWADERs l1.txt --l1-skips 1,10,100,1000 [--l1-skip-first K]` | `L1/`: one FITS per N (all through the pipeline), `noise_vs_skips.csv/.png`, `compare/`, `<image>_diagnostics/` |
| commissioning packet | `run_superWADERs packet.txt --packet --module-list modules.csv` | `packet/index.html` + one page per module |
| dark current over a series | `run_superWADERs series.txt --dc-series [--dc-limit X] [--no-clusters]` | `dc_series/`: `dc_report.pdf`, `dc_series.csv`, one page per module |
| defect maps | `run_superWADERs images.txt --defect-map [--defect-sigma 5] [--no-clusters]` | `defects/<module>_ACM<id>_<size>/`: `defects.csv`, `defect_map.html`, composites, the group's Image_4 / Image_2 outputs |
| serial-register images | `run_superWADERs sr.txt --sr [--sr-threshold 4] [--sr-min-length 3] [--no-clusters]` | `sr/`: `sr_streaks.csv`, `sr_summary.csv`, PNGs, one page per module |

The commissioning flags can be combined in one run (e.g. `--dc-series --defect-map --sr`), and every report folder
has an `index.html` listing its pages. Add `--no-clusters` when you need no cluster files: it is much faster.

What each tool does:

- **Native engine**: the same steps as run_WADERs (autoconfig, gain from the many-peak fit where better, dark current
  from the 0e-/1e- peaks, every panaSKImg pedestal method, `--tune`), and panaSKImg's `--save-plots` PDFs. Cluster
  thresholds: `--thresholds auto` (default; in electrons from each CCD's dark charge), `sigma0` (WADERs' 10 / 30
  sigma) or `PIX,SEED`.
- **L1** (`--l1-skips`, `--l1-skip-first K`): one image per number of skips N (mean of skips K..K+N-1) with the DAQ
  header from the `.meta`; noise and gain per CCD against N with the 1/sqrt(N) line; the DC-fit, baseline and sigma
  plots of every N side by side; diagnostics per channel (start-of-pixel transient and the suggested
  `--l1-skip-first`, single-skip noise, noise vs N within a pixel, spectrum over the skips, drift during a pixel).
- **Packet** (`--packet`): a list of a commissioning packet's files (traces, PSD traces, 1-skip and 500-skip images
  of any size and binning, L1 files; any order). One report per module with the tabs it has data for: Overview,
  Trace, PSD (sigma_ADU, sigma_CDS), per image a numbers tab (gain, noise, dark current, chi2, the pixel-charge
  histogram with the dark-current fit when the electron peaks are resolved, profiles) and an image viewer. No
  clustering.
- **Dark current** (`--dc-series`): every image of the list is one point in time (DATEINI). Rate = DC /
  (exposure + readout/2) in e-/pix/day, and from the growth of the dark charge along the rows (slope; the intercept
  is charge made during the readout). Settling fit (plateau + exponential) with 5 or more images; flags "above
  limit" (`--dc-limit`) and "still falling". Pages: Rates, Fits (each image's histogram with its dark-current fit),
  Per row, Data (table of every number), Method.
- **Defect map** (`--defect-map`): images of one ACM with the same size and binning form a group (any type or skip
  count). Median composite per CCD; bright / dark columns and hot / cold pixel spots (persistent local excess,
  `--defect-sigma`); the group's `Image_4` / `Image_2` scripts (`src/lsm_defects/`) run when the images have an
  overscan. `defects.csv` is the defect log.
- **Serial-register images** (`--sr`): streaks in the serial register (readout artifacts): runs of >=
  `--sr-min-length` pixels above `--sr-threshold` x noise along one read, confined to it. Streak list, counts per
  1000 reads, lengths, positions along the register, dark charge per read.

### The interactive pages

Every page is one self-contained HTML file: it works offline in Chrome, Safari or Firefox (double-click it), and it
can be sent to others as it is.

- **One page per module.** The row below the tabs lists its CCDs: tick the ones to show, **all CCDs in the module**
  for all of them, **only** next to a CCD for that one alone (remembered in the browser). Traces are shown one below
  the other; PSD spectra, profiles and series plots are overlaid with one colour per CCD; numbers are side by side
  (a value that is the same for every CCD is shown once, in grey); fit histograms one per CCD.
- **Images, like ds9 frames**: one CCD at a time; the slider, the arrow buttons or the left / right arrow keys switch
  CCD and keep zoom, position, scale mode and the row / column cut, so the same spot can be compared; **blink**
  switches every second. Wheel or + / - zoom, drag to move, click a pixel for its row and column cut, shift + drag a
  box for the region's statistics, scale (auto = around the background, like zscale), colormap, asinh, PNG. Defect
  and streak pages mark what was found (tick boxes per type); click a line of the list to jump to it.
- **Plots**: drag a box to zoom, shift + drag to move, wheel zooms x (alt + wheel: y), double-click or **reset**,
  **log y / lin y**, **PNG**; hover to read values.
- **Flags**: a red number is outside the usual range (noise > 0.3 e-, peaks not resolved, defects found, DC above the
  limit, ...); a red dot marks the tab and the CCD it belongs to.
- **Comments**: a box at the bottom of every tab, saved in the browser as you type; **Save comments** writes them to a
  `.json` file (keep it next to the page or send it), **Load comments** reads one back.
- **Export PDF**: every tab on its own page (the ticked CCDs) with the comments; choose "Save as PDF".
- **Log**: at the bottom of each tab, how its numbers were made (files, header settings, checks, fit results).
- **? Help** (top right) explains all of this on the page itself.

### Where the code is

| file | what |
|---|---|
| `bin/run_superWADERs` | the command (starts `bin/run_WADERs` with the native engine as default) |
| `src/native_pipeline.py`, `native_kernels.py`, `native_rootio.py`, `native_plots.py` | native engine: pedestal, fits, clustering, cluster files, plots |
| `src/raw_data.py`, `src/raw_levels/` | L0 / L1 readers, L1 -> FITS, diagnostics (`raw_levels/`: copies from ccd_opt_ml, `utils/sync_from_ccd_opt_ml.py`) |
| `src/packet_report.py`, `src/packet_report_template.html` | packet reports and the page used by every report |
| `src/dc_series.py` | `--dc-series` |
| `src/defects.py`, `src/lsm_defects/` | `--defect-map` (the group's scripts: copies, `utils/sync_from_lsm_analysis.py`) |
| `src/sr_images.py` | `--sr` |
| `docs/superWADERs_user_guide.md`, `docs/native_pipeline.md`, `docs/superWADERs_commissioning_next_steps.md` | user guide, native-engine validation, plan and open items |

**Not yet validated on real data** (see the user guide, section 5, for the data needed): a full commissioning packet
(5-minute 1x1 1-skip and 10x10 500-skip images), serial-register images (tested only on synthetic data), the defect
thresholds against known defects, and dark-current limits.

## Requirements

A python environment with numpy, astropy, PyROOT and pysimdamicm (panaSKImg). `run_WADERs` can
be started from any python: if that one lacks these, it finds the conda/mamba environment that
has panaSKImg and re-runs itself there with `conda run` (nothing stays activated). Choose the
environment with `--env NAME` or `$WADERS_ENV`, and panaSKImg with `--panaskimg` or `$PANASKIMG`.

To call it by name from anywhere, add the `bin/` folder to your PATH (in `~/.bashrc` or
`~/.zshrc`): `export PATH="/path/to/comm_scripts/bin:$PATH"`.

## What it does

For each image, `gain_dc_split.py --per-ext` makes one JSON per CCD:

1. `autoconfig_panaSKImg.py` measures every CCD from the raw pixels (geometry, gain, noise,
   dark current, bad columns, pedestal method) and fits only the electron peaks that follow
   a Poisson distribution (usually 0 e- and 1 e-), which keeps the dark current unbiased.
2. The gain is also measured from a fit of all peaks; where that is more precise (errors scaled
   by sqrt(chi2/ndf)) and agrees with the 2-peak gain, it is held in the final JSON.
3. Flooded / unresolved CCDs are flagged and not run (unless `--all-ext`).

Then panaSKImg runs once per CCD (`--ext N`), in parallel, and the results are collected.

## The individual scripts

| script | purpose |
|---|---|
| `bin/run_WADERs` | everything above, in one command |
| `bin/run_AGMX` | A_GMX cluster files (WADERs format) for a finished `run_WADERs` folder (see "A_GMX clustering") |
| `src/autoconfig_panaSKImg.py` | one image -> panaSKImg JSON (or one per CCD with `--per-ext`); the reference JSON is built in (`--reference` to use your own) |
| `src/gain_dc_split.py` | gain from a many-peak fit, dark current from the 0e-/1e- fit |
| `src/fit_quality.py` | chi2/ndf, scaled errors and dark-current bias of the fits in a panaSKImg or `run_WADERs` output folder |
| `src/gain_vs_charge.py` | gain vs electrons per pixel g(n) from the electron comb (`measure`: images -> curve, fit, plot; `--model interpolated`/`broken`/`linear`) and the cluster-energy correction (`apply`: cluster ROOT files -> corrected copies) |
| `src/agmx_root.py` | A_GMX clusters -> panaSKImg cluster files (used by `run_AGMX` and `run_WADERs --clusterer`) |
| `src/naming.py` | image name and ACM id of an image (shared by the scripts) |
| `utils/split_clustersRec_by_ext.py` | split a cluster file with all CCDs (`_ext1234.root`) into one file per CCD |
| `utils/compare_runs.py` | compare two `run_WADERs` output folders (summary, fits, JSONs, every cluster energy, composite tables; files present in only one run count as differences); use it to check that a code change leaves the results unchanged |
| `utils/compare_dcfit_vs_skips.py` | tile the DC-fit PDFs of several runs (e.g. different skip numbers) into one PDF per CCD |
| `utils/skip_noise_L1.py` | decode a level-1 image (every skip kept, `*.RJ45ADCDATA<ch>`) and plot noise vs number of skips, the noise spectrum over skips and the start-of-pixel transient; writes `<name>_noise_vs_skips.png/.csv` and `<name>_avg.fits` to `-o` (`<name>` = file name + `_ch<k>`) (default: current folder); `--composites N` also writes N images, each the average of a consecutive 1/N of the skips, to `<name>_composites_N<N>/`. Needs `src/` next to `utils/` (gain finder) |

Each script prints its options with `-h`. The `src/` modules can also be run on their own,
e.g. `python src/autoconfig_panaSKImg.py image.fz -o auto.json`.

`docs/gain_correction_report.pdf` and `docs/gain_correction_Am241.pdf` show what
`--gain-correction` does and how it changes the Fe-55 and Am-241 peaks on the production data.

Note: the wider DC-fit plot (fit 2 peaks, show 5) uses an `x_max_plot` option that is only in
patched pysimdamicm versions; without it the scripts simply leave it out.

## A_GMX clustering (optional, `--clusterer` / `run_AGMX`)

`run_WADERs` clusters with panaSKImg's ClusterFinder (WADERs: pixel > 10 sigma, seed > 30 sigma), or with thresholds in
electrons (`--autoconfig-args "--cluster-threshold-e auto"`). The clustering study of 2026-10-23..25 produced A_GMX
(adaptive switch + Gaussian-mixture deblending, `clustering_tests/`), which runs on the calibrated images panaSKImg
writes (`*_EXT<n>_waders.fits.bz2`) and writes **cluster files in the WADERs format** (`src/agmx_root.py`: the same
trees and branches, computed by pysimdamicm's own `Cluster` class, plus `agmx_*` branches):

```bash
run_WADERs images.txt --gain-correction --clusterer agmx --composite   # A_GMX files -> root_agmx*/, composite on them
run_WADERs images.txt --gain-correction --clusterer both --composite   # also composite_agmx/ next to composite/
run_AGMX WADERs_analysis_<run>                                        # for a run that is already finished
```

| folder | clusters of |
|---|---|
| `root_agmx/` | the images as calibrated by panaSKImg |
| `root_agmx_gaincorr/` (`_per_image/`) | the images converted with the g(n) curves of `gain_vs_charge/` (same `--gain-anchor`) |

Extra branches: `agmx_cls` (1 x-ray, 2 x-ray diffuse, 3 low-E, 4 muon, 5 worm, 6 alpha, 7 big deposit, 8/9 row/column
artefact, 10 extended), `agmx_blend`, `agmx_ped`, `agmx_peakfrac`, `agmx_chi2`, `agmx_sig`, `agmx_regime`; the file also
has a `TNamed agmx_config`. Recommended selection for X-ray lines: `agmx_blend < 0.02 && agmx_ped < 10` plus the usual
sigma_xy cut; `-999` in a flag branch means "not computed" (e.g. `agmx_chi2` of clusters that were not fitted by the
mixture). The default (`--clusterer waders`) leaves the pipeline unchanged (checked with `utils/compare_runs.py`).
Format check against real panaSKImg files: `clustering_tests/agmx_root_check.py` (identical except numerically
degenerate cases: fits of 1-3 bin profiles, PCA eigenvector signs, same-row/column closest-cluster ties).

Against WADERs (878 image-CCDs): unbiased Fe-55 energies at every cluster width, 2.7x better energy resolution, close
X-ray pairs resolved, few-electron events found without fakes; on a crowded CCD more significant lines with the right
energy scale. Method, equations, figures and all results: `docs/aGMX/aGMX_clustering.tex` (Overleaf:
`docs/aGMX/aGMX_clustering_overleaf.zip`) and `docs/clustering_methods_report.pdf`; code map: `clustering_tests/README.md`.
From Python: `adaptive.adaptive_cluster(Q_electrons, valid_mask, engine="gmx")` (a list of dicts: `E` [keV], `y`, `x`,
`sxy`, `cls`, `blend`, `ped`, `peakfrac`, `chi2`, `sig`, `edge`).

## Tests

```bash
python -m unittest discover -s tests -v      # from the comm_scripts folder, with the panaSKImg python
```

Unit tests cover the input handling of `run_WADERs`, the g(n) models of `gain_vs_charge.py`, the Am-241 fit model, the
automatic cluster thresholds, the clustering methods of `clustering_tests/` and the A_GMX file writer (simulated images);
they need no data and run in seconds. For changes to the pipeline itself, compare the outputs of `run_WADERs` on the
same images before and after with `utils/compare_runs.py OLD NEW` (the pipeline is deterministic; see
docs/PIPELINE_GUIDE.md, section 6).

## Data

Data are not kept in this folder (`.gitignore` excludes raw level-1 files). The level-1 noise study data
(`img_200x1500x4000_..._L1_101_..._15.RJ45ADCDATA0` and its `skip_noise_L1.py` outputs) are in
`~/Documents/comm_scripts_data/L1_noise/`.
