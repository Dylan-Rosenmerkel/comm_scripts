# comm_scripts pipeline guide

For a physics student or collaborator new to this code: what the pipeline does, the physics behind each step,
which file and function does it, and where to start reading. For running it, the user guide
(`docs/usage/comm_scripts_usage.pdf`), `run_WADERs -h` (options) and `run_WADERs -h -v` (full guide) are the reference;
the [README](../README.md) has the quick start.

## 1. What it does

`run_WADERs` takes raw skipper-CCD images and produces, per CCD, the gain, noise and dark current with fit errors,
the reconstructed particle clusters (ROOT files), and optionally gain-corrected cluster energies and the Fe-55 /
Am-241 composite-cluster analysis. It does this with **panaSKImg** (the WADERs package, `pysimdamicm`), which needs a
JSON configuration per image. Writing those JSONs by hand is the hard part; comm_scripts measures the right settings
from the pixels themselves.

## 2. Data flow

```
 raw image (.fz / .fits)                              one image = 4 CCDs ("extensions" EXT1..EXT4)
     |
     |  step 1   src/gain_dc_split.py  ->  src/autoconfig_panaSKImg.py
     |           per CCD: geometry, pedestal method, bad columns, noise, gain, dark current
     |           two quick panaSKImg fits per CCD: 0e-/1e- peaks only (dark current) and all peaks (gain)
     v
 json/<image>/final/<image>_EXTn.json                 one panaSKImg configuration per CCD
     |
     |  step 2   panaSKImg --ext n  (in parallel)     pedestal, calibration, dark-current fit, clusters
     v
 root/  cluster files   root/dc_fits/  dark-current fits   *.pdf plots
     |
     |  step 3b  src/gain_vs_charge.py  (--gain-correction)
     |           measure g(n) from the electron comb, rewrite cluster energies
     v
 root_gaincorr/  (and root_gaincorr_per_image/)
     |
     |  step 3d  src/agmx_root.py  (--clusterer agmx|both)
     |           A_GMX clusters on the calibrated images *_EXTn_waders.fits.bz2 (g(n)-converted with the gain
     |           correction), written as panaSKImg cluster files
     v
 root_agmx/  (and root_agmx_gaincorr*/)
     |
     |  step 3c  MakeComposite_Fe55_Clusters_Extensions.cc  (--composite, ROOT macro)
     v
 composite/Fe_55_Analysis_Plots/<module>/...          composite clusters, Fe-55 / Am-241 spectra, CTI
     |
     |  step 4   src/fit_quality.py + gain_dc_split.held_quality
     v
 summary.txt, fit_quality.csv                         gain, noise, dark current per CCD with errors
```

Each step is one function in [`bin/run_WADERs`](../bin/run_WADERs) (`step1_configure`, `step2_run_panaskimg`,
`step3_collect_root_files`, `step3b_gain_correction`, `step3d_agmx`, `step3c_composite`, `step4_summary`); `main()`
calls them in order (3d before 3c, so the composite can use the A_GMX files) and is the best place to start reading.

## 3. The physics, step by step

### Skipper CCDs, ADU and gain
A skipper CCD reads each pixel's charge many times ("skips") and averages, so the read noise falls as 1/sqrt(skips)
to a fraction of an electron. The amplifier output is in ADU; the **gain** (ADU per electron) converts it to
electrons. With sub-electron noise the pixel-value histogram shows separate peaks at 0, 1, 2, ... electrons: the
**gain is their spacing**, the **noise** their width.

### Pedestal and overscan
Each row has a baseline (pedestal) that drifts slightly; it is measured in the **overscan** (columns read after the
real pixels, which hold no charge) and subtracted per row. If the overscan itself holds charge (high dark current) a
plain median lands on the 1 e- peak instead of 0 e-; autoconfig then switches to a lowest-peak finder. `--tune`
tries each pedestal method with panaSKImg and keeps the one with the smallest total fit error.

### Dark current from the 0e-/1e- peaks (why two fits)
Thermal electrons arrive at random, so the number per pixel is **Poisson** with mean lambda (the dark current per
pixel): P(0) = exp(-lambda), hence lambda = -ln P(0). Peaks at 2 e- and above also hold non-Poisson charge (particle
clusters, dark current varying across the CCD), so fitting many peaks biases lambda upward. Fitting only the 0e-/1e-
peaks gives an unbiased dark current but a less precise gain (less lever arm). `gain_dc_split.py` therefore runs both
fits and keeps the many-peak gain where it is more precise and agrees with the 2-peak gain within 3 sigma.

### Fit quality
ROOT's fit errors assume the model is right. When chi2/ndf > 1 they are too small, so `fit_quality.py` scales them by
sqrt(chi2/ndf), and compares the fitted dark current with the independent 0e-/1e- count estimate ("DC bias").
`summary.txt` shows both.

### Flooded and dead CCDs
A CCD with so much charge that the peaks are not resolved ("flooded/unresolved") cannot be calibrated; it is skipped
unless `--all-ext`. A channel with no CCD attached looks exactly like the overscan and is dropped.

### Clusters
panaSKImg groups neighbouring pixels above threshold into **clusters** (one particle interaction). A cluster's energy
is the sum of its pixels' electrons times 3.74 eV per electron-hole pair. Its width (sigma_xy, from Gaussian fits of
its row and column projections) tells how far the charge diffused, i.e. how deep in the silicon the interaction was:
`MakeComposite` uses it to split "front" and "back" events and measure charge-transfer inefficiency (CTI) from
composite clusters.

**A_GMX** (`--clusterer agmx|both`, or `bin/run_AGMX` on a finished run) is an alternative clusterer from the
2026-10-23..25 study (`clustering_tests/adaptive.py` + `gmx.py`; algorithm in `docs/aGMX/`). Instead of thresholds it
models the image (dark charge removed) as a sum of 2-D Gaussian charge clouds fitted by EM and shares each pixel's
charge between overlapping clouds, so it keeps faint cluster edges (unbiased energies at every width), separates close
X-rays and finds few-electron events; in a background it cannot model it falls back to hysteresis like WADERs.
`src/agmx_root.py` turns its clusters into pysimdamicm `Cluster` objects, so the files have exactly the WADERs format
plus `agmx_*` quality branches (blend, local pedestal, ...). Recommended X-ray selection: `agmx_blend < 0.02` and
`agmx_ped < 10`, plus the usual sigma_xy cut.

### Gain versus charge, g(n) (`--gain-correction`)
The gain falls slowly with the charge in a pixel (about 3 % lower at 1000 e-), so keV clusters converted with the
0e-/1e- gain come out a few percent low. The pixel-value comb continues to high charge wherever X rays deposit charge;
its **local spacing at n electrons is the local gain g(n)**. `gain_vs_charge.py measure` finds it with a periodogram
(the comb spacing that maximises |mean of exp(2 pi i x / g)|) in windows of charge, using only "clean" rows where the
readout was not just disturbed by a large charge. `apply` converts every pixel back with the inverse of
ADU(n) = integral of g from 0 to n. Images of one module are combined (pooled); by default incoherently (comb powers added;
independent of the image order), or coherently (`--gain-combine coherent`: each image converted with its own curve
first, so the combs line up and all pixels can be pooled; reaches higher charge); `per-image` gives each image its own curve. If the gain changed during a run, use one curve per gain period (see
[am241_note](am241_note/README.md), `correct_gain_per_period.py`).

### Fe-55 and Am-241 lines (what the composite step and the notes look at)
Fe-55 gives Mn K-alpha 5.899 keV and K-beta 6.490 keV. Am-241 gives Np L x rays at 11.9-22.4 keV, gamma rays at
26.34 and 59.54 keV, a Compton edge of the 59.54 keV gamma ray at 11.25 keV and a backscatter peak above 48.3 keV.

## 4. Files

| file | what it is |
|---|---|
| `bin/run_WADERs` | the whole pipeline; one function per step, `main()` at the bottom |
| `bin/run_AGMX` | step 3d alone, on a finished run |
| `src/autoconfig_panaSKImg.py` | raw image -> panaSKImg JSON: everything measured from the pixels (largest file; its module docstring lists what is measured) |
| `src/gain_dc_split.py` | gain from the many-peak fit, dark current from the 0e-/1e- fit, the final JSON per CCD |
| `src/fit_quality.py` | reads panaSKImg's dark-current fits: chi2/ndf, scaled errors, dark-current bias |
| `src/gain_vs_charge.py` | g(n): `measure` (images -> curve) and `apply` (cluster files -> corrected copies) |
| `src/agmx_root.py` | A_GMX clusters -> panaSKImg cluster files (step 3d) |
| `src/naming.py` | image name / ACM id of an image (standard library only) |
| `clustering_tests/` | the clustering methods (A_GMX is used by step 3d) and the scripts that tested them (README there) |
| `utils/` | standalone helpers (split cluster files per CCD, tile DC-fit PDFs, noise vs skips from level-1 data) |
| `docs/` | notes and reports with the scripts that made them (`report_scripts/`, `gain_correction_method/`, `am241_note/`) |
| `tests/` | unit tests (`python -m unittest discover -s tests -v`) |

## 5. Where to start reading

1. `bin/run_WADERs`: the module docstring, then `main()` and the `step*` functions.
2. `src/gain_dc_split.py`: short; shows how a configuration is chosen per CCD.
3. `src/autoconfig_panaSKImg.py`: the module docstring first (what is measured and why), then the function named
   in it for the quantity you care about.
4. `src/gain_vs_charge.py`: the module docstring explains the method and the three g(n) forms.

## 6. Changing the code safely

The pipeline is deterministic: running the same input twice gives byte-identical cluster files and summaries. So any
refactor can be checked by running `run_WADERs` on the same images before and after (e.g. with
`--composite --gain-correction both`) and comparing `summary.txt`, `fit_quality.csv`, the JSONs in `json/*/final/`
and the cluster energies in every `root*/` folder - `utils/compare_runs.py OLD NEW` does exactly this. Add a unit test in `tests/` for any new pure function. Never
change how normal images are fitted without saying so; new behaviour goes behind a flag.

## 7. Related documents

- `docs/usage/comm_scripts_usage.pdf`: the user guide (every part of the repository and how to run it).
- `docs/aGMX/`: the A_GMX clustering algorithm, its performance against WADERs, the quality flags.
- `docs/gain_correction_method/`: the method note for g(n) (LaTeX + figures + Overleaf zip).
- `docs/am241_note/` and `docs/AMtest_Am241_10-30keV_note_overleaf.zip`: the Am-241 10-30 keV analysis
  (gain correction per period, cluster-width cut, constrained fit with the Lepy et al. 2008 line list).
- `docs/gain_correction_report.pdf`, `docs/gain_correction_Am241.pdf`: earlier reports.
- `docs/README.md`: index of the documentation; `docs/notes/`: hand-over notes and `OPEN_ITEMS.md`.

Data are kept outside the repository; the level-1 noise study data are in `~/Documents/comm_scripts_data/L1_noise/`.
