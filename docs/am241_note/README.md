# Am-241 lines in the AMtest data (10-30 keV note)

The scripts behind the note *"Am-241 lines in the AMtest skipper-CCD data: gain-vs-charge correction, cluster-width
cut and a constrained fit of the 10-30 keV region"*. The LaTeX source is in `note/` (compiled with pdfLaTeX or
tectonic; the Overleaf upload is `../AMtest_Am241_10-30keV_note_overleaf.zip`).

## What the analysis does

1. **Gain correction per gain period.** The gain-vs-charge curve g(n) is measured from the images and used to convert
   pixel charges; because the gain stepped down by 1-5 % between images 40 and 41, one curve per CCD and period is used.
2. **Cluster-width cut** sigma_xy < 0.7 pixels: the low-energy x rays are absorbed close to the surface and make
   compact clusters, while the continuum under them is mostly wider. Chosen with a fit-based figure of merit on a grid
   of lower and upper limits, checked on independent halves of the images.
3. **Constrained fit of 10-30 keV**: every Np L line at its tabulated energy with fixed intensity ratios within its
   group (Lepy, Plagnard, Ferreux, Appl. Radiat. Isot. 66 (2008) 715, Table 3), one energy scale k for all lines, a
   smooth background and the Compton edge of the 59.54 keV gamma ray; binned Poisson likelihood.
4. **The two extra lines** at 22.1 and 24.9 keV: attributed to silver K fluorescence (to be confirmed); the
   alternatives (enhanced Np Lg4, Am self-fluorescence, conversion electrons, tin) are tested and disfavoured.

## Running it

Data: the two run_WADERs outputs of the same images, `<run-dir>/no_gaincorr/WADERs_analysis_<run-name>/` (no
correction) and `<run-dir>/gaincorr/WADERs_analysis_<run-name>/` (`--gain-correction`). Defaults:
`--run-dir ~/Documents/JH1_Testing/AMTEst/runs_29img` (or `$AM241_RUN_DIR`), `--run-name AMtest_29`. Intermediate files
go to `<run-dir>/analysis/` (`--work-dir`), the note's figures and tables to `note/` (`--note-dir`).
Use the python with ROOT (the conda environment with panaSKImg). In order:

| step | script | what it does | approx. time |
|---|---|---|---|
| 0 | `run_WADERs list.txt -o no_gaincorr --composite` and `... -o gaincorr --composite --gain-correction` | the pipeline | ~35 min each for 29 images |
| 1 | `correct_gain_per_period.py --list list.txt --periods 21-40 41-50` | g(n) per gain period -> `root_gaincorr_epochs/` | ~10 min |
| 2 | `extract_clusters.py` | cluster files -> one table per CCD (`clusters2_ext<N>.npz`) | ~1 min |
| 3 | `fit_all_peaks.py [--no-grid]` | all fits with / without the cut; the band-cut grid | ~40 min (grid) |
| 4 | `peak_table.py` | mean, sigma, FWHM, counts of every peak (`peak_table.csv`) | ~2 min |
| 5 | `identification_tests.py` | the extra-line hypotheses -> `note/tables/` | ~10 min |
| 6 | `make_note_figures.py` | every figure and table of the note -> `note/` | ~5 min |
| - | `gold_check.py` | gold L lines (smoke-detector foil check) | seconds |
| - | `make_report.py` | the companion 9-page PDF report | ~2 min |

Every script prints its options with `-h`.

## Files

| file | contents |
|---|---|
| `common.py` | shared settings: data folders, the cut value, `load_clusters()`, plot style |
| `amfit.py` | the constrained model (`Model`), the line list (`NP`, from Lepy et al. Table 3), `fit_595()` |
| `correct_gain_per_period.py` | step 1 |
| `extract_clusters.py` | step 2 |
| `fit_all_peaks.py` | step 3 (was `fits29.py`) |
| `peak_table.py` | step 4 |
| `identification_tests.py` | step 5 (`fit_with()`: the model plus extra lines) |
| `make_note_figures.py` | step 6 (was `figs_doc.py`) |
| `gold_check.py`, `make_report.py` | the gold check and the report |
| `note/` | `main.tex`, `figures/`, `tables/`, and `numbers.json` / `hypotheses.json` (numbers quoted in the text) |

The fit model is tested on synthetic spectra in `../../tests/test_am241_model.py`.
