# comm_scripts

Scripts to analyse skipper-CCD images with panaSKImg (WADERs) without writing JSONs by hand.

**User guide (every part of the repository and how to run it):** `docs/usage/comm_scripts_usage.pdf`
(LaTeX source `docs/usage/comm_scripts_usage.tex`, Overleaf upload `docs/usage/comm_scripts_usage_overleaf.zip`).

## Layout

```
comm_scripts/
  README.md
  bin/     commands to run          run_WADERs (and the alias runWADERs), run_AGMX (A_GMX clusters for a finished run)
  src/     the pipeline modules     autoconfig_panaSKImg.py, gain_dc_split.py,
                                    gain_vs_charge.py, fit_quality.py, agmx_root.py (A_GMX -> panaSKImg cluster files)
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
```

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
