# comm_scripts

Scripts to analyse skipper-CCD images with panaSKImg (WADERs) without writing JSONs by hand.

## Layout

```
comm_scripts/
  README.md
  bin/     commands to run          run_WADERs (and the alias runWADERs)
  src/     the pipeline modules     autoconfig_panaSKImg.py, gain_dc_split.py,
                                    gain_vs_charge.py, fit_quality.py
  utils/   standalone helpers       compare_dcfit_vs_skips.py, split_clustersRec_by_ext.py,
                                    skip_noise_L1.py
  docs/    reports and notes        gain_correction_report.pdf, gain_correction_Am241.pdf,
                                    CONTEXT_2026-10-16.md (hand-over notes: what was done, open items),
                                    patches/ (local pysimdamicm change), data/, report_scripts/
```

Keep `bin/` and `src/` together: `run_WADERs` loads the modules from `../src`, and the `src/`
scripts import each other from their own folder. `utils/` scripts are independent, except
`skip_noise_L1.py`, which uses the gain finder in `src/`.

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
| `--gain-anchor comb` (default) / `fit` | low-charge gain from the comb, or each image's fitted gain | |

Curves and plots are in `gain_vs_charge/` (`module_<ACM>_...` and `image_<image>_...`): the
local gain `*_gain_vs_charge.png` and the nonlinearity curve `*_nonlinearity.png` (measured charge
minus peak number against measured charge, `A(n)/g(0) - n`, reference gain g(0) of the curve; its
quadratic fit is also in the JSON and in `summary.txt`). Several
images of a module are combined by adding each image's comb power, so their combs need not line up.
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
| `src/autoconfig_panaSKImg.py` | one image -> panaSKImg JSON (or one per CCD with `--per-ext`); the reference JSON is built in (`--reference` to use your own) |
| `src/gain_dc_split.py` | gain from a many-peak fit, dark current from the 0e-/1e- fit |
| `src/fit_quality.py` | chi2/ndf, scaled errors and dark-current bias of the fits in a panaSKImg output folder |
| `src/gain_vs_charge.py` | gain vs electrons per pixel g(n) from the electron comb (`measure`: images -> curve, fit, plot; `--model interpolated`/`broken`/`linear`) and the cluster-energy correction (`apply`: cluster ROOT files -> corrected copies) |
| `utils/split_clustersRec_by_ext.py` | split a cluster file with all CCDs (`_ext1234.root`) into one file per CCD |
| `utils/compare_dcfit_vs_skips.py` | tile the DC-fit PDFs of several runs (e.g. different skip numbers) into one PDF per CCD |
| `utils/skip_noise_L1.py` | decode a level-1 image (every skip kept, `*.RJ45ADCDATA<ch>`) and plot noise vs number of skips, the noise spectrum over skips and the start-of-pixel transient; writes `<name>_noise_vs_skips.png/.csv` and `<name>_avg.fits` to `-o` (default: current folder); `--composites N` also writes N images, each the average of a consecutive 1/N of the skips, to `<name>_composites_N<N>/`. Needs `src/` next to `utils/` (gain finder) |

Each script prints its options with `-h`. The `src/` modules can also be run on their own,
e.g. `python src/autoconfig_panaSKImg.py image.fz -o auto.json`.

`docs/gain_correction_report.pdf` and `docs/gain_correction_Am241.pdf` show what
`--gain-correction` does and how it changes the Fe-55 and Am-241 peaks on the production data.

Note: the wider DC-fit plot (fit 2 peaks, show 5) uses an `x_max_plot` option that is only in
patched pysimdamicm versions; without it the scripts simply leave it out.
