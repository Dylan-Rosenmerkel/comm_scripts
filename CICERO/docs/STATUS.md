# CICERO: what has been checked, on which data, and what is still open (2026-11-24)

Ready for standard electron-counting images (Fe-55, muons, Am-241, unbinned or moderately binned) and for the
commissioning tools on L0 / L1 / averaged images. Few-electron, crowded, heavily binned and skip-resolved data run,
but parts of those chains are checked only on synthetic or injected data. The details are below; the plans behind
the open items are in [CICERO_plan_binned_crowded.md](CICERO_plan_binned_crowded.md).

## Checked

| part | data | result |
|---|---|---|
| engine vs panaSKImg (pedestal, gain / noise / DC fits, threshold clusters, ROOT format) | DM09 Image_5; board-103 skip-noise images (80 CCD-images); L2 surface run | same gain, noise and DC (median difference 0.00 %); same clusters with `--thresholds sigma0` |
| `--clusterer auto` = A_GMX | DM09 Image_5, 3 images, g(n) | identical to `--clusterer agmx` and to `run_WADERs --engine native --clusterer agmx`: 12/12 files, same clusters |
| Fe-55 energy scale | DM07 / DM27 / DM28 Image_8 (crowded mode) | K-alpha within 0.2-0.9 eV of A_GMX + g(n) once the eV per electron is matched (3.7486 vs 3.74) |
| energies vs truth | injections into every image type (physics audit 2026-11-14) | unbiased to <= 0.1 % for 1-60 keV at every depth, unbinned and binned; 'unmodelled' chain +0.7-1.3 % at 0.3-6 keV |
| g(n) (`--gain-correction`) | DM09, DM07/27/28, PD06 settings A/B/C; a fresh clone of only CICERO (2026-11-24) | curves per module and per image; corrected cluster files written |
| chain choice on other data | PD06 A/B/C, AMtest (Am-241), L2 ACM 103, real 10x10 and 20x20 binned images, single-skip, std images | every case runs; flooded CCDs -> threshold, noisy -> 'unmodelled', std images refused |
| binned images | DM09 binned in software (truth benchmark) | gain right in every shape; K-alpha within 7 eV up to 10x10; few-electron and X-ray efficiency at or near the oracle ceiling for every binning except 100x100 |
| Bayesian few-electron finder | events injected into real L2 noise; 90-image halo and boundary tests; real L2 ACM 103 | 58 of 64 injected events found (N exact); 0.9 % lost / 0.1-0.3 % double-counted at block boundaries; 0 events in the real L2 data (as expected) |
| crowded mode | L2 ACM 101 with injected tracks; Fe-55 Image_8 | muons' foreign charge 0.28 -> 0.085; Fe-55 K-alpha moves < 1 eV, ~8 % more front clusters kept |
| L0 traces and PSD traces | 2026-09-16 PA03 / DR03 | decoder checks agree; plots, trace viewer and the trace / PSD pages render; since 2026-11-24 every L0 input gets the commissioning trace / PSD pages (`packet/`), also without `--packet` |
| L1 -> FITS | synthetic unbinned file (round trip); real binned 20x20 file (2026-10-06) | exact round trip, noise 1/sqrt(N); 80/80 rows, levels match the DAQ's own image |
| commissioning tools | packet report (68 PA03 / DR03 files); DC series (DM24-26, 6 images over 19 h); defect maps (L2 surface run, DM26 Image_5) | pages build, numbers consistent; defect thresholds not yet tuned |
| campaigns | synthetic 26-module campaign | only new inputs processed; 9 injected faults flagged, no healthy CCD flagged |
| row overscan (2026-11-24) | DM07/27/28 Image_4 (1600 rows of a 1536-row CCD), Image_6 / Image_7 | rows read after the last CCD row (readout dark charge only) are no longer counted as active area |

## Open

1. **Crowded cut:** core fill 0.10 decides crowded mode per CCD; PD06 sits at it, so CCDs of one setting can differ.
   A run-level decision with a band is planned.
2. **Muons in crowded images** are often cut at crossings (56 % keep >= 90 % of their charge in one cluster).
3. **Heavily binned, ultra-low background:** 100x100 at 1e-5 e-/pix is below the ceiling; the flooded rule on the real
   20x20 images gives ~340 clusters per CCD (not understood yet). Real binned images with an X-ray source are needed.
4. **Dark current below ~1e-5 e-/pix** cannot be measured from a couple of images (per-image fit floor ~1e-3); pooled
   over a run it is prior-dominated without dark electrons.
5. **CCDs without an electron comb** (e.g. L2 ACM 103): the gain must be given (`--gain`, `--gain-file`); it is
   flagged.
6. **Skip-resolved raw data:** tested on a synthetic file only; no per-skip veto of readout spikes yet.
7. **Asymmetric read-noise tails** (L2 EXT4): the empirical noise model exists but is off by default.
8. **High-noise images (single skip):** no matched-filter detection of diffuse low-energy deposits.
9. **Saturation and hot columns** are reported, not masked.
10. **Defect-map thresholds and the serial-register streak finder** are not tuned (no SR images yet).
11. **Bayesian events in real data:** none seen yet; the path is verified on injected events.
12. **Sibling gains** (a CCD's gain from its neighbours) are tested on constructed records only.
13. **Row overscan** is left out but not yet used as a measurement of the readout dark charge.

## Data that would close items

Binned images of CCDs whose unbinned images exist, ideally with a source (3); crowded surface runs with known content
(1, 2); real skip-resolved L1 files with known events, e.g. an LED or source run (6); SR images (10); a dark-current
series with the group's limits; LED or high-DC images for gains of comb-less CCDs (5).
