# comm_scripts additions to WADERs (panaSKImg)

Three optional additions from the comm_scripts module-testing tools. With the default configuration
nothing changes: every option is off unless set in the JSON.

| What | Where | JSON |
|---|---|---|
| Gain vs charge g(n): cluster energies corrected for the charge dependence of the gain | `GainNonlinearityProcess` (processes/gain_nonlinearity.py) + ClusterFinder | `GainNonlinearityProcess` block, in `sequence` after `CalibrationProcess` |
| Thresholds from the dark charge | ClusterFinder (processes/comm_clustering.py) | `ClusterFinder.spr_threshold_auto` |
| A_GMX model-based clustering | ClusterFinder (processes/comm_clustering.py, pysimdamicm/comm/) | `ClusterFinder.algorithm: "agmx"` and `ClusterFinder.agmx` |

Example: `pysimdamicm/json/panaSKImg_config_comm_example.json`.
The comm_scripts modules used are in `pysimdamicm/comm/` (copied unchanged from comm_scripts; imports made
package-relative).

## 1. Gain vs charge, g(n)

The gain is the spacing of the electron comb, and it falls slowly with the charge in the pixel (about 3 % at
1000 e- on the L2 / DM CCDs). One calibration gain from the 0-1 e- peaks therefore puts X-ray and track
energies a few % low.

**Measure** (from the raw images of one module, before panaSKImg):

    gain_vs_charge measure IMAGES... -o gain_vs_charge/ [--model interpolated|broken|linear]
                                     [--combine incoherent|coherent] [--memory off|auto] [--autoconfig-args "..."]

Outputs per run name:

- `NAME_gain_vs_charge.json`: the g(n) points and model per extension (the `model_file` below)
- `NAME_gain_vs_charge.png`: g(n), absolute and relative to g(0)
- `NAME_nonlinearity.png`: measured charge minus peak number against measured charge
- `NAME_adu_of_n.png`: A(n) = integral of g, the curve the correction inverts, and A(n) - g(0) n

**Apply** inside panaSKImg:

    "sequence": "...;CalibrationProcess;GainNonlinearityProcess;ClusterFinder;BuildClusterMask",
    "GainNonlinearityProcess": {"model_file": "gain_vs_charge/NAME_gain_vs_charge.json",
                                "anchor": "comb", "temperature": null}

- With `algorithm: "waders"`, the clusters are found on the calibrated image exactly as before (same pixels).
- Their `pixels_E` and `Energy` are then converted pixel by pixel:
  `ADU = pixels_E / e2eV x gain`, `n = A^-1(ADU)`, `E = n x epsilon(T)`.
- This is the correction of comm_scripts' `gain_vs_charge apply` on cluster ROOT files, which is also
  available as `gain_vs_charge apply MODEL.json CLUSTERS.root... -o OUTDIR`.
- `anchor`: `comb` uses the comb's g(n) including its low-charge gain; `fit` keeps the calibration gain at
  low charge.
- `temperature` sets epsilon(T) (default `$COMM_TEMPERATURE_K` or 130 K, 3.7486 eV).
- With `algorithm: "agmx"` the calibrated image is converted with g(n) *before* clustering (as comm_scripts).
- **With a polynomial CalibrationProcess** (`gain: [g, c2, c3]`, from get_calibration_constant), the ADU is
  rebuilt with that polynomial (`gain_nonlinearity.calibration_adu`). g(n) then replaces the polynomial for the
  energies; it is never applied on top of it. See comm_scripts docs/WADERs_gn_vs_polynomial_calibration.md for
  the comparison of the two measurements.

## 2. Thresholds from the dark charge (`spr_threshold_auto`)

    "ClusterFinder": {"spr_threshold_auto": {"alpha": 0.10, "seed_e": 9.5}, ...}      (or true)

How the thresholds are chosen, per amplifier:

1. The dark charge λ is the median over rows of -ln P(0). The pixels of the calibrated active region are
   rounded to electrons, and the zero peak is the lowest well-populated value.
2. k is the smallest k ≥ 2 with P(Poisson(λ) ≥ k) ≤ alpha.
3. The thresholds become `[1.0, k - 0.5, max(seed_e, k + 0.5)]` in electrons.

On a CCD whose peaks are not resolved (noise ≥ 0.3 e-), the configured `spr_threshold` is kept. This is the
rule of comm_scripts' `autoconfig_panaSKImg --cluster-threshold-e auto`. Higher dark charge raises the
threshold, so that background pixels do not link clusters together.

## 3. A_GMX (`algorithm: "agmx"`)

    "ClusterFinder": {"algorithm": "agmx",
                      "agmx": {"crowded": "auto", "crowded_shapes": "on", "fe_candidates": true,
                               "temperature": null, "first_row": 2}, ...}

How it works:

- The image is described as background plus Gaussian charge clouds; in crowded mode, line components are
  added for straight tracks.
- Every pixel's charge is shared between the clusters in proportion to the model, so touching deposits are
  not merged and no tails are cut.
- The clusters are ordinary pysimdamicm `Cluster` objects in the same `clustersRec` tree. They carry the
  extra branches:
  - `agmx_cls`: class (codes in the `agmx_config` entry of the file);
  - `agmx_blend`: fraction of the cluster's model shared with others;
  - `agmx_ped`, `agmx_peakfrac`, `agmx_chi2`, `agmx_sig`;
  - `agmx_regime`.
- The `agmx_config` entry (TNamed, JSON) records the settings and the crowded-mode decision per amplifier.
- Recommended clean selection: `agmx_blend < 0.02 && agmx_ped < 10`.

Options (the run_WADERs flags `--agmx-crowded`, `--agmx-crowded-shapes`, `--temperature`, and
`$AGMX_FE_CANDIDATES`):

| key | values | meaning |
|---|---|---|
| `crowded` | auto, on, off | crowded mode (straight-track finder, background safety net, streaks, linking); auto: when the core fill ≥ 0.10 |
| `crowded_shapes` | on, off | in crowded mode, muons as line components of the mixture and a topology split |
| `fe_candidates` | true, false | few-electron candidate catalogue next to the output (counting CCDs only) |
| `temperature` | K | epsilon(T) for the A_GMX energies |
| `first_row` | int | first row of the image used (input.image.id_row_start); A_GMX starts 3 rows later |

A_GMX needs numba, scipy and scikit-image.

## Running it: run_WADERs

`bin/run_WADERs` goes from raw images to cluster files in one command:
configuration per CCD → g(n) → panaSKImg per CCD → MakeComposite → summary.
The options of this note are its flags, and it writes them into the panaSKImg JSONs it runs.
From a WADERs checkout it uses that checkout (no install needed); it finds a conda environment
with ROOT if the current python lacks it.

    run_WADERs images.txt [-o PARENT] [-j N] [--gain-correction [pooled|per-image]] [--gain-anchor comb|fit]
               [--gain-model interpolated|broken|linear|smooth] [--gain-combine incoherent|coherent] [--gain-memory off|auto]
               [--gain-periods auto|off]
               [--clusterer waders|agmx|both] [--agmx-crowded auto|on|off] [--agmx-crowded-shapes on|off]
               [--auto-threshold] [--temperature K] [--composite] [--module ACM=NAME] ...
    run_WADERs -h -v      the full guide

`images.txt` lists one image per line.

**Gain periods.** One g(n) curve per module assumes that all its images share the gain. A change of the readout
settings between campaigns breaks this (AMtest 106: all CCDs 2.5-3.5 % lower from 2026-05-21). Pooling both periods
puts two electron combs on top of each other, and every line of the corrected energies comes out doubled (59.5 keV:
two peaks 1.5 keV apart). With `--gain-periods auto` (default) run_WADERs:

1. puts the images of a module in time order (date and time in the image name);
2. takes each image's gain from its 0e-/1e- fit (step 1), relative to the module median and averaged over the
   CCDs;
3. splits the module where this gain steps by more than 1 % and more than 5 times the image-to-image scatter
   (`gain_vs_charge.gain_periods`). A single image with a poor fit is set aside and does not start a period.

Each period gets its own curve. A CCD whose period curve is not usable gets the curve of the whole module.
summary.txt lists the periods, their dates and their gain. Checks:

- AMtest 106 (29 images): split at 2026-05-21 00:46, the first image of the new settings (19 + 10 images).
- DM09 (3 images): not split.
- Simulated series: a 2.5 % step is found at the right image (±1) in 83 % of cases; a series without a step is split
  in 7 % (with 10 % of images given a poor fit).

What each flag sets:

| flag | in the final JSONs (json/<image>/final/) |
|---|---|
| `--gain-correction` | `GainNonlinearityProcess` after CalibrationProcess, `model_file` = gain_vs_charge/module_<ACM>_gain_vs_charge.json (per-image: image_<stem>_...) |
| `--gain-anchor` | `GainNonlinearityProcess.anchor` |
| `--gain-periods auto` (default) | a module whose images span a gain step gets one curve per period: `model_file` = gain_vs_charge/module_<ACM>_period<k>_gain_vs_charge.json |
| `--clusterer agmx` | `ClusterFinder.algorithm = "agmx"`, `ClusterFinder.agmx = {crowded, crowded_shapes, fe_candidates, temperature, first_row}` |
| `--clusterer both` | final/ as waders, final_agmx/ as agmx: two panaSKImg runs per CCD → root/ and root_agmx/ |
| `--auto-threshold` | `ClusterFinder.spr_threshold_auto = true` |

Example (module DM09, Image 5, three Fe-55 images; 2 min 15 s with -j 4):

    cd ~/Documents/comm_scripts_data/waders_integration_example
    ~/Documents/Software/comm_scripts/WADERs/bin/run_WADERs DM09_Image5.txt -j 4 \
        --gain-correction --clusterer both --composite

    WADERs_analysis_DM09_Image5/
      summary.txt                       gain/noise/DC per CCD, g(n) per CCD, clustering settings, MakeComposite
      gain_vs_charge/module_109_*       g(n) model (.json) and plots: _gain_vs_charge, _nonlinearity, _adu_of_n
      json/<image>/final/               the JSONs run (with GainNonlinearityProcess, algorithm waders)
      json/<image>/final_agmx/          the A_GMX JSONs (algorithm agmx)
      root/                             WADERs clusters, energies corrected with g(n)
      root_agmx/                        A_GMX clusters of the g(n)-converted image
      composite/, composite_agmx/       MakeComposite (Fe-55 lines, front/back) on root/ and root_agmx/

Compared with comm_scripts' run_WADERs on the same images (2026-10-26, root_gaincorr/ and
root_agmx_gaincorr/):

- WADERs clusters: identical clusters (same pixels on every CCD compared). The energies are higher by
  3.7486 / 3.74 (epsilon(T), introduced after that run).
- On EXT2 the g(n) curve itself changed: the single-electron anchor added since then (DM09 EXT2 gain rises
  at low charge).
- A_GMX: 926 of 931 clusters identical, energies equal to 4e-5.

## Validation (L2 surface run, image 9, EXT1, 2026-11-02)

- **g(n), in-pipeline vs `gain_vs_charge apply` on the uncorrected cluster file:** the same 256 clusters
  with identical pixels; energies equal to 2e-5. The difference is the calibration gain: 1.3377 as applied
  by CalibrationProcess, against the unrounded 1.33768 that `apply` reads from the info tree.
- **A_GMX with g(n), in-pipeline vs comm_scripts `agmx_root.process`:**
  - identical (6832 clusters, same pixels and energies) when given the same image (float32, as stored in
    `*_waders.fits`) and the same gain;
  - in normal use (float64 image, applied gain), 96 % of the clusters are identical and the median energy
    ratio is 1.00002. Crowded fits react to the last digits of the input.
- **`spr_threshold_auto` vs `autoconfig_panaSKImg --cluster-threshold-e auto`:** same dark charge
  (2.40 e-/pix) and thresholds (pixels ≥ 5 e-, seed > 9.5 e-).
