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
