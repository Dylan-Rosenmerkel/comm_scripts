# superWADERs clustering chain: plan for binned images, crowded images and the remaining gaps (2026-11-13)

Status: plan, nothing below is implemented. Priority order: **0 → 1 (binned) → 2 (crowded) → 3 (small items) → 4
(larger gaps)**. Each step ends with a validation on real data and, where truth is needed, on injections into real
images; nothing becomes the default before that.

Code: `superWADERs/` (engine, clustering, commissioning). Validation harness: `wip/superWADERs_validation/` (runs
every data type on this machine through `run_superWADERs` and compares with earlier results), A_GMX benchmarks in
`wip/lowdc/` and `wip/agmx_crowded/`.

## 0. First: the validation now running

`wip/superWADERs_validation/run_validation.py` (started 2026-11-13):
- the 16 suite cases (DM09, Fe-55 Image_8 DM07/27/28, PD06 A/B/C, AMtest, L2 ACM 103 and its strips, injected
  few-electron events, binned, raw skips, single skip, std image) through the program itself, against the earlier
  results of the wip engine;
- DM09: `--clusterer auto` = `--clusterer agmx` = `run_WADERs --engine native --clusterer agmx` (clusters identical);
- the Bayesian switch's three untested parts: halo hand-over, block boundaries, real L2 data;
- the efficiency patch E1-E3 on four real cases (time, clusters).

Whatever it finds is fixed before the steps below.

**Results (2026-11-13), and what was changed:**
- All 16 cases run; the chain chosen per CCD is the same as with the wip engine (PD06 flips on 1-2 CCDs at the 0.10
  cut, item 2.1). DM09: clusters identical in all 12 files for auto, agmx and `run_WADERs --engine native --clusterer
  agmx`, also after the changes below.
- **Fixed:** the summary crashed on CCDs whose fit has no error (flooded / unresolved, gain from the prior): AMtest and
  binned runs ended with an error after all processing (`fit_quality.quality`, `gain_dc_split.held_quality`, in
  `superWADERs/` and `src/`).
- **Fixed:** `gmx.detection_bg` on images with < 61 rows (np.convolve 'same' returns 61 values): window sums from the
  full convolution; identical for >= 61 rows (checked), no crash on strips. In `superWADERs/` and `clustering_tests/`.
- **Halo test** (switch, 90 images, Fe-55 / muons): efficiency of 2-10 e- events 10-20 px from a deposit 0.86-0.89
  with the 15 px halo, 0.91-0.96 with 3-10 px; no fakes and big deposits unchanged with every halo. **HALO 15 -> 5 px.**
- **Boundary test** (90 images, checkerboard of read noise): the current switch loses 26 of 2811 events A_GMX finds
  (0.9 %) and gains 207; double counting 0.1-0.3 % (A_GMX alone ~2 %); fakes 7 vs 9. Passed.
- **Real L2 ACM 103 through the switch:** A_GMX alone 109 small clusters (91 on EXT4, median 4 e- in 6 px: noise);
  the switch replaces 55, finds 0 Bayesian events (as the stand-alone study), 0.1-0.4 s per image vs 0.6 s. The rest
  on EXT4 is the asymmetric-noise item (3).
- **Efficiency patch E1-E3 applied:** same cluster counts and total energies in every file; DM09 and single skip
  identical; on crowded Fe-55 / AMtest CCDs a few cluster energies move by up to 9e-4 (relative); 10-45 % faster.
- **Open, first item of 1:** the flooded rule (2026-11-12) gives AMtest EXT2 (382 e-/pix) 0 clusters (6421 before)
  but the 20x20 binned CCDs ~340 each (1-5 before): to be understood with the binned truth benchmark (1.4, 1.6).
  The 10x10 CCD with 5.8 e- noise gives 1841 'unmodelled' clusters on 180 x 630 bins: probably noise.

## 1. Binned images (priority)

### What binning does to the data

On-chip binning NPBIN (rows) x NSBIN (columns) sums the charge of b_y x b_x pixels before the readout. Per binned
pixel ("bin"):
- **dark charge** lambda_bin ~ lambda_pix x b_x x b_y (+ charge picked up during the longer readout per bin);
- **read noise** once per bin (not summed): binning raises the signal over the noise;
- **diffusion** sigma_bin = sigma_pix / b along each axis: an event of sigma <= 1 px is one bin, or 2 on a boundary;
  with b_x != b_y the shapes are anisotropic;
- **events per bin**: the occupancy rises with the bin area, so a binned image is crowded much sooner.

Our data: 10x10 (PA03/DR03: counting on 3 CCDs, comb 1.33-1.37; one CCD at 5.7 e- noise) and 20x20 (flooded:
170-210 e-/bin, noise 13-19 e-), plus binned L1 (20x20).

### What goes wrong today

Every size in the chain is in pixels and assumes b = 1:
- A_GMX: cloud widths (the pixel-integrated Gaussian, minimum widths), merge distances, class rules (muon length, worm,
  alpha, X-ray 'diffuse' by sigma), crowded-mode trigger and track finder (min_len 40 px, max_gap, rms_max), halo
  radii;
- the Bayesian finder: the sigma grid 0.1-1.3 px and the diffusion prior (DAMIC law, sigma_max 1.01 px);
- the threshold clusterer: fine for energies, but its size cuts are in pixels;
- the front end: the 'flooded' rule (> 10 e-/pix) is right for the data but the gain then comes from the prior.

### Plan

1.1 **Bin geometry through the chain.** `characterise` already reads NPBIN / NSBIN; pass (b_y, b_x) to
`adaptive_cluster`, the finder and the threshold clusterer, and express every size in micrometres (15 um pixels):
lengths and widths scale by 1/b per axis. Anisotropic binning: separate y / x widths in the pixel-integrated model.

1.2 **Event shapes per bin.** The diffusion prior in bins (sigma / b per axis); the pixel-integrated Gaussian already
integrates over the bin, so a point-like event is a single bin with the right likelihood. The Bayesian finder's sigma
grid scaled the same way (many events become single-bin: the 'point deposit' component and its prior need checking).

1.3 **Classification in physical units.** Track length, width and dE/dx per micrometre instead of per pixel; X-ray vs
diffuse X-ray only where sigma_bin is resolvable (b <= 2-3), otherwise 'unresolved'.

1.4 **Flooded binned images (20x20).** Electron counting is impossible; the dark charge per bin is a near-Gaussian
continuum of variance lambda_bin + read noise^2. Cluster above the flooding level at k x that width (done for flooded
CCDs on 2026-11-12, check on binned data); the energy threshold is then ~ 5 sqrt(lambda_bin) e- (~70 e-, 0.26 keV at
200 e-/bin). Report the dark current per pixel (lambda_bin / b_x b_y) and the column defects per binned column.

1.5 **Gain for binned images without a comb:** use the gain of the same CCD from unbinned images of the same module
(the 'sibling' gain, extended across binning settings: the gain does not depend on binning). Flag it as such.

1.6 **Validation.**
- *Synthetic with truth:* bin real unbinned images in software (DM09 Image_5, Fe-55 Image_8, an L2 ACM 101 track
  image) by 2, 5, 10, 20 and 2x1: sum the electron images, add the read noise once per bin, the dark charge per bin.
  Truth = the unbinned clusters (energies, positions / b). Metrics: K-alpha position and resolution, cluster counts,
  matched efficiency and fakes, energy of matched clusters, track lengths.
- *Real:* PA03 / DR03 10x10 and 20x20, binned L1; the dark current per pixel and the column defects against the same
  CCD's unbinned images.
- Targets: K-alpha within 2 eV of the unbinned value up to b = 5; no change at b = 1 (identical clusters).

## 2. Crowded images: tracks and overlaps (muons not yet resolved well)

### Where crowded mode is (validation in `crowded_shapes.py`, L2 ACM101 with injections)

| measure | A_GMX before crowded mode | now |
|---|---|---|
| muons: median foreign charge | 0.28 | 0.085 |
| muons: 'correct' | 12.5 % | 43.5 % |
| muons: >= 90 % of the charge in one cluster | 62 % | **56 %** (worse: tracks are cut at crossings) |
| electron tracks whole | 66 % | 71 % |
| touching X-ray pairs resolved | 72 % | 76 % |
| Fe-55 clean K-alpha events | | -3 % |

So the line model takes foreign charge off the muons, but a muon crossing others is still often broken into pieces.

### Ideas, from the cheapest

2.1 **Run-level crowded decision.** One decision per run setting with a band around the 0.10 cut (PD06 flips between
CCDs today); recorded in the summary. Small.

2.2 **Peeling.** Fit the most certain long straight tracks first (longest, highest occupancy), subtract their model,
and look for tracks again in the residual: each pass sees a less crowded image. Today all hints are found in one pass
on the full image.

2.3 **Tracking through crossings.** At a junction, join the branches that continue each other: straightness, the same
width (sigma grows with depth: along a muon it changes linearly, so the two halves of one track must agree), and the
same charge per unit length (dE/dx, Landau-distributed but constant on average). Today the topology split uses angle
only (within 40 deg of opposite). Assign the branches at each junction by a small matching problem (minimum cost).

2.4 **A physical muon model.** A muon crosses the full thickness: its width goes from ~0 at the front surface to the
back-side diffusion width at the other end, its length and its width growth are tied by the track angle, and its
charge per length is fixed by the angle. Fitting this one-parameter-family (endpoints, direction, dE/dx) instead of a
free width per segment constrains overlapping tracks strongly and gives depth / direction for free.

2.5 **Curved tracks (electrons, deltas).** A path model instead of straight lines: a minimum-cost path through the
charge (skeleton graph) with smoothness, fitted as a chain of short line segments; handles worms and delta rays
leaving a muon (today often split off or merged).

2.6 **Learned hints (optional, later).** A segmentation network trained on injections into real crowded images (the
injection set of wip/agmx_crowded can produce unlimited labelled data) proposes track masks; the physical model stays
the arbiter of charge and energy. Only if 2.2-2.5 plateau.

2.7 **Benchmark first.** Extend the injection benchmark (wip/agmx_crowded) with: per object type (muon, electron,
X-ray, alpha) and per degree of overlap (number of crossings, foreign charge under the track); split / merge rates;
charge completeness; K-alpha of clean Fe-55 events (must stay within 1 eV). Targets: muons >= 90 % in one cluster from
56 % to >= 80 %, foreign charge <= 0.08, no loss on X-rays.

Order: 2.7 → 2.1 → 2.2 → 2.3 → 2.4, measuring each on the benchmark; 2.5 / 2.6 after.

## 3. Small items (each about a day, after 1)

- **Asymmetric noise:** switch to the empirical (measured) noise model automatically when the noise tails are
  asymmetric (L2 EXT4); the option exists, off by default.
- **gmx.detection_bg on images with < 61 rows:** fix at the source (worked around by padding today).
- **prepare_run outside its regime:** skip the Bayesian finder when no block can be Bayesian (minutes per image on
  noisy runs today).
- **Very low dark current:** the dark-current fit at its limits (lambda 1e-4 / 1.3, seen on L2 ACM 103): report the
  run-pooled value as the measurement and the fit as 'not measurable'.
- **Efficiency patch E1-E3:** apply if stage 3 of the validation shows only the expected 1e-9-level differences.

## 4. Larger gaps (after 1 and 2, with the data at hand)

- **Skip-resolved data (L1): per-skip veto.** A real 2-10 e- single-pixel event is in every skip; a readout spike in
  a few. The Bayesian finder's main fakes are these; the interface (`characterise.skip_samples`) exists and real L1
  files are on this machine.
- **High-noise images (single skip, sigma >= 1 e-):** matched-filter detection (smooth with the expected event shape,
  then threshold) for diffuse low-energy deposits that a per-pixel threshold misses.
- **Very low dark current:** pool the dark charge over all images of the run setting by default; a per-CCD gain store
  (from LED / high-dark-current images) for CCDs without a comb.
- **Serial-register images:** need real examples.

## Data needed from the group

- Binned images of CCDs whose unbinned images we have (same module, same cooldown), ideally with an X-ray source:
  the direct test of 1.
- Crowded images with known content (surface runs with a source), for 2 beyond the injections.
- Real serial-register images; L1 files with known events (e.g. an LED or source run) for the per-skip veto.
