# Validation of the native pipeline against panaSKImg (DM09 Image_5)

Reference: the run_WADERs run `scratchpad/e2e/WADERs_analysis_DM09_Image5` (3 images of module 109 / DM09, taken
2025-08-27/28, 4 CCDs each, 480 x 1280, no overscan): `root/` = WADERs ClusterFinder + g(n) inside panaSKImg
(GainNonlinearityProcess), `root_agmx/` = A_GMX inside panaSKImg + g(n), `json/<image>/final*/` the exact settings,
`gain_vs_charge/module_109_gain_vs_charge.json` the g(n) curve (anchor comb). panaSKImg = the WADERS_JHU copy.

Produced with `validate.py --run-dir RUN --images DM09_Image5.txt -o OUT -j 2 --time-panaskimg` (all numbers
below are from that run: OUT/validation.json, OUT/validation_tables.md, 2026-11-10) plus MakeComposite on the
native output. Cluster-by-cluster matching = identical set of (x, y) pixels. Branch comparisons are exact
(float32 as stored in both files).

| configuration | what | compared with |
|---|---|---|
| mimic | panaSKImg's JSON settings (pedestal, fit window/limits/masks, 10/30 sigma0, gain rounded to 4 decimals for the calibration, cluster shapes from the energies before g(n)); NATIVE dark-current fit | root/ |
| mimic_gain | as mimic with panaSKImg's own fitted gain | root/ |
| native | default: autoconfig settings, gain_dc_split gain choice with native fits, thresholds `auto`, g(n), epsilon(T) | root/ |
| native_sigma0 | native, but panaSKImg's 10/30 sigma0 thresholds | root/ |
| agmx_mimic | A_GMX with panaSKImg's settings, gain and 3.74 eV/e- | root_agmx/ |
| agmx | default A_GMX: native settings, g(n), epsilon(T) = 3.7486 eV/e- | root_agmx/ |

## 1. Pedestal, gain, noise, dark current

- **Pedestal per row**: all 480 rows of all 12 CCDs identical (mu_row and sigma_row, max |difference| 0 ADU), also
  the per-CCD means in the info tree (e.g. EXT1 of image 1: -36.74744199582848 / 0.17811227404078867 ADU in both).
- **Dark-current fit** (native scipy fit vs panaSKImg's Minuit fit, same chi2; `native` and `mimic` give the same
  numbers because the native gain_dc_split logic made the same gain choice on every CCD):

| quantity (info tree, ADU) | images 1, 2: max abs diff (max diff / panaSKImg error) | image 0 (gain held): max abs diff (diff / error) |
|---|---|---|
| gain ext<n>_gainext | 5.7e-7 (0.001) | 1.4e-5 (0.78) |
| noise ext<n>_sigext | 8.2e-8 (0.000) | 1.5e-6 (0.002) |
| dark current ext<n>_dcext | 2.6e-7 (0.000) | 6.0e-6 (0.002) |
| zero peak ext<n>_mu0ext | 7.0e-8 (0.000) | 5.4e-6 (0.006) |

  Image 0: gain_dc_split holds the many-peak gain within +-0.001 % (+-1.3e-5 ADU/e-); panaSKImg's Minuit ends
  somewhere inside that window, the native fit fixes the gain at its centre, so the difference is the window
  itself (the "0.78 sigma" is in units of Minuit's ~2e-5 error of a gain held by limits). The native info tree
  stores the held gain's real error (many-peak fit, scaled): 0.0042-0.0048 ADU/e-, as in gain_dc_split's summary.
  Gains also equal gain_dc_split's summary (1.3386, 1.1978, 1.3304, 1.2528 held on image 0; fitted on 1, 2).
- Branch sets: clustersRec and info have exactly the same branches and types as panaSKImg's files;
  process_config is reduced (81 panaSKImg settings omitted, 7 added, see README).

## 2. Clusters with panaSKImg's settings (mimic, mimic_gain, agmx_mimic)

| | clusters native / panaSKImg | matched (same pixels) | same order | branches not bit-identical | Fe-window clusters changing front/back (sigma_xy < 1) |
|---|---|---|---|---|---|
| mimic (WADERs) | 10931 / 10931 | 10931 | yes | fPosX, fPosY, fSTD_X, fSTD_Y only | 0 of 6104 |
| mimic_gain | 10931 / 10931 | 10931 | yes | fPosX, fPosY, fSTD_X, fSTD_Y only | 0 of 6104 |
| agmx_mimic | 11737 / 11737 | 11737 | yes | fPosX, fPosY, fSTD_X, fSTD_Y only | 1 of 6865 |

Every other clustersRec branch (82 compared: energies, pixel lists and energies, Qmax, positions, widths, PCA,
projections, track end points, seed quantities, closest_cls_*, cluster_id, agmx_*, ...) is identical to the last
bit on every cluster. The native fit gives panaSKImg's gain to 4 decimals, so mimic and mimic_gain are identical.

The Gaussian profile fits (fSTD_X/Y): |native - panaSKImg| median 2e-4 px, 99th percentile 0.17 px (WADERs) /
0.32 px (A_GMX), maximum 8.7 px (one multi-peak track). Likelihood check on EXT1 of image 1 (mimic): of the
1205 profile fits whose fSTD differs by > 1e-4 px, the native result has the lower Poisson likelihood in all but 2,
and those 2 are equal within 3e-9 (float32 storage of the parameters). The one cluster where an early single-start
version found a worse local minimum (75 px track, ROOT 5.850 px) now gives 5.853 px. Minuit stops at its EDM
tolerance; the large differences are clusters one or two columns wide, where the likelihood in sigma is flat or
degenerate (sigma -> 0). Front/back classification of Fe-55-window clusters: unchanged except 1 A_GMX cluster.

**K-alpha** (front side, the 3 images pooled per CCD, MakeComposite's fit): identical to panaSKImg for mimic
(5.8823, 5.8757, 5.8958, 5.8973 keV) and agmx_mimic (5.8761, 5.8686, 5.8844, 5.8839 keV).

## 3. Corrected thresholds (native default) and the other native choices

Thresholds per CCD: images 1, 2: panaSKImg sigma0 = 0.131-0.147 e- -> pixel > 1.3-1.5 e-, seed >= 3.9-4.4 e-;
native (median per-row dark charge 0.22-0.28 e-/pix): pixel > 1.5 e-, seed >= 9.5 e-. Image 0 (dark charge
1.06-1.33 e-/pix median per row, 0.48-0.54 e-/pix in the fitted rows): sigma0 = 0.75-0.79 e- -> panaSKImg pixel
> 7.5-7.9 e-, seed >= 22-24 e-; native pixel > 2.5 e- (EXT1, EXT3) / 3.5 e- (EXT2, EXT4), seed >= 9.5 e-.

| image | panaSKImg (root/): clusters, of them < 1 keV, K-alpha front [keV], sigma, N | native: clusters, < 1 keV, K-alpha, sigma, N | native A_GMX K-alpha (panaSKImg A_GMX) |
|---|---|---|---|
| 0 (high dark charge) | 3680, 39, 5.8736 +- 0.0020, 0.067, 1213 | 3496, 77, 5.8929 +- 0.0018, 0.061, 1151 | 5.8931 (5.8798) |
| 1 | 3610, 386, 5.8972 +- 0.0018, 0.056, 1013 | 3267, 39, 5.8969 +- 0.0018, 0.056, 1013 | 5.8924 (5.8793) |
| 2 | 3641, 382, 5.8954 +- 0.0017, 0.056, 1039 | 3311, 47, 5.8952 +- 0.0018, 0.056, 1039 | 5.8908 (5.8785) |

Per CCD (3 images pooled), K-alpha front [keV]: panaSKImg 5.8823 / 5.8757 / 5.8958 / 5.8973, native
5.8907 / 5.8879 / 5.9020 / 5.9002 (+8.4 / +12.2 / +6.2 / +2.8 eV; width 0.064/0.063/0.057/0.056 ->
0.058/0.054/0.058/0.056 keV).

Physics: sigma0 is the width of the masked 0e-/1e- pedestal mixture, so it follows the dark charge, not the read
noise (0.14-0.25 e-). On image 0 panaSKImg's pixel threshold of ~7.5 e- drops the low-charge edge pixels of every
X-ray cluster: its K-alpha is 19 eV low and 6 eV wider than on images 1 and 2 of the same CCDs. With the electron
thresholds image 0 gives 5.8929 keV, in line with images 1 / 2 (5.8969 / 5.8952), i.e. the image-to-image spread of
the K-alpha goes from 24 eV to 4 eV. On images 1 and 2 the pixel thresholds are nearly the same (1.33 vs 1.5 e-),
the front K-alpha clusters are the same (N equal, mu within 0.3 eV). The visible difference there is the seed: 30 sigma0 =
4 e- accepts ~385 sub-keV groups per image, the 9.5 e- seed keeps ~40 of them (cluster count -10 %); the slightly
higher pixel threshold (1.5 vs 1.33 e-) also changes the pixel set of part of the larger clusters (EXT1 of image 1:
602 of 924 panaSKImg clusters keep identical pixels), without moving the K-alpha. The isolated effect of the thresholds is confirmed by native_sigma0: native fits
and g(n) with panaSKImg's thresholds reproduce panaSKImg's clusters exactly (10931/10931 identical pixel sets) and
its K-alpha exactly.

Other differences in the native default (by design, see README): shapes and Qmax from the g(n)-corrected energies
(panaSKImg: before g(n); this changes Qmax, PosX, widths, PCA in the last digits of matched clusters), calibration
with the unrounded gain, and for A_GMX epsilon(T) = 3.7486 eV instead of 3.74 eV: the A_GMX clusters are identical
to panaSKImg's (11737/11737, same pixels and charges) and every energy is x 1.002289 (+12-14 eV on K-alpha), which
puts A_GMX on the same energy scale as the WADERs+g(n) files (native WADERs 5.895-5.897 keV vs native A_GMX
5.891-5.893 keV on images 1-2; panaSKImg's 3.74 eV made them differ by 18 eV).

## 4. MakeComposite on the native output

The macro was copied to `composite/` and compiled there (ACLiC), then run as run_composite.py does
(`root -l -q -b 'MakeComposite_Fe55_Clusters_Extensions.cc+("list",5,"DM09",OUT)'`) on the 12 native files:
it ran without errors (rc 0) and wrote all plots and energy_summary.csv. Its front K-alpha
(5.89065 / 5.88794 / 5.90199 / 5.90016 keV) equals the simple fit in validate.py to the last printed digit.

energy_summary.csv, K-alpha (location 0 = front, 1 = back) [keV]:

| CCD | panaSKImg front | native front | panaSKImg back | native back | panaSKImg A_GMX front | native A_GMX front |
|---|---|---|---|---|---|---|
| EXT1 | 5.8823 | 5.8907 | 5.9 +- 0 (fit failed) | 5.8624 | 5.8761 | 5.8878 |
| EXT2 | 5.8757 | 5.8879 | 5.7565 | 5.8491 | 5.8686 | 5.8817 |
| EXT3 | 5.8958 | 5.9020 | 5.9060 | 5.8871 | 5.8844 | 5.8982 |
| EXT4 | 5.8973 | 5.9002 | 5.7934 | 5.8864 | 5.8839 | 5.8974 |

Back side (diffuse clusters, most sensitive to the pixel threshold): the panaSKImg values scatter by 150 eV (one
failed fit); with the electron thresholds they are 5.849-5.887 keV. The front K-beta fit of the macro failed for
native EXT1 (1.13 +- 1.48 keV, the macro's unvalidated front K-beta fit); all other K-beta fits converged.

## 5. Wall time (this Mac, 14 cores, machine shared)

| run | time |
|---|---|
| native, threshold clusterer, 3 images, command line `-j 2` | 5.6 s (warm numba cache); 10.4 s on the first run after a cache wipe (compilation) |
| native, threshold clusterer, 3 images, `-j 1` | 7.1 s; per image in-process 1.7-3.0 s: read 0.04 s, autoconfig measurements 0.5-0.6 s, per CCD 0.25-0.35 s (pedestal 0.025, fits 0.01-0.04, clustering 0.013, cluster quantities 0.15-0.29, ROOT write 0.05 s; +0.8 s ROOT import on the first CCD) |
| native, A_GMX, 3 images, `-j 2` | 59.8 s (A_GMX itself ~6.6 s per CCD) |
| panaSKImg, one CCD, threshold + g(n), --save-plots | 7.1 s (6.5 s in a repeat) |
| panaSKImg, one CCD, A_GMX + g(n), --save-plots | 13.6 s |

panaSKImg needs one process per CCD and, before that, gain_dc_split's two autoconfig runs and 2-3 panaSKImg fit-only
runs per CCD (the whole run_WADERs with both clusterers took 146 s). Per CCD the native threshold pipeline is
~25x faster than one panaSKImg run; A_GMX takes ~6.6 s per CCD natively vs 13.6 s for a panaSKImg A_GMX run (not broken down further), and
stays dominated by adaptive_cluster (profile: background_map 2.5 s - mostly
8900 scipy minimize_scalar calls in truncated_poisson_mle -, count_threshold 2.4 s, EM 1.9 s), which is not changed
here.

## Not measured / open

- Images with an overscan, prescan, binning, or a charged overscan (is_highDC): not validated.
- No validation of the native pipeline on other modules or image types (only DM09 Image_5).
- fSTD values cannot be reproduced bit by bit (Minuit's stopping point); quantified above.
