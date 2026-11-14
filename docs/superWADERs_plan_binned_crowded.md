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

### 1A. The real use: ultra-low-background runs (2026-11-13, from the group)

Binned images will mostly come from ultra-low-background studies: **dark current 1e-6 to 1e-4 e-/pix**, few events,
**many images** per setting, **any binning** (no typical one: 10x10, 1x10, 1x100, 100x1, 100x100, ...), and **image
stitching in both senses**: stacking the images of one setting, and joining consecutive images of a continuous readout.
The Fe-55 benchmark below (0.5 e-/pix, ~1000 events per CCD) is the opposite regime; it stays as the robustness test of
step 1.0 (high occupancy), which these runs will rarely need (10x10 at 1e-4 e-/pix is 0.01 e- per bin).

So the binned work is mainly the **Bayesian few-electron regime with bins and pooling**:

1A.1 **Benchmark with truth for this regime** (`wip/binned/bench_lowbg.py`): simulated unbinned frames at 1e-6, 1e-5,
1e-4 e-/pix and read noise 0.12 e- (L2-like), sparse events (few-electron events 2-10 e- with the DAMIC diffusion law,
X-rays, muon tracks), persistent hot pixels and a hot column, binned on chip (charge summed, read noise once per bin)
at 1x1, 10x10, 1x10, 10x1, 1x100, 100x1, 100x100, 20 images per setting written as consecutive frames of a continuous
readout (events crossing an image boundary are split, with their truth). Measured: efficiency per event type and
electron number, fakes per image and per Mpix, dark charge (pooled) against the truth, energies; events split at image
boundaries. First run on the current code (baseline), then after every step.

**Baseline (2026-11-13, current code, `--gain 1.3`; 20 images x 2 CCDs per setting; efficiency = truth found):**

| binning | 2 e- | 3 e- | 4-5 e- | 6-10 e- | X-rays | muons found / whole | fakes | DC fit / true |
|---|---|---|---|---|---|---|---|---|
| 1x1 (1e-6 / 1e-5 / 1e-4) | 0.78 / 0.75 / 0.08 | 0.92 / 1.00 / 0.86 | 0.93-0.96 | 0.99-1.00 | 0.95-0.99 | 0.95-1.0 / 0.72-0.87 | 0-1 | 77x / 7.7x / 1.1x |
| 1x10 | 0.56 / 0.26 / 0.00 | 0.55-0.69 | 0.92-0.98 | 0.98 | 0.96-0.98 | 0.78-0.97 / 0.67-0.83 | 0 | 21x / 2.4x / 0.9x |
| 10x1 | 0.43 / 0.15 / 0.00 | 0.47-0.69 | 0.80-0.88 | 0.85-0.88 | 0.83-0.85 | 0.78-0.86 / 0.41-0.58 | 0 | 22x / 2.3x / 1.0x |
| 10x10 | 0.00 | 0.00-0.11 | 0.12-0.16 | 0.20-0.26 | 0.81-0.85 | 0.72-0.94 / 0.36-0.65 | 0-1 | 18x / 2.2x / 0.9x |
| 1x100 | 0.00-0.15 | 0.07-0.29 | 0.31-0.42 | 0.57-0.65 | 0.76-0.83 | 0.71-0.78 / 0.59-0.67 | 0-2 | |
| 100x1 | 0.00-0.15 | 0.00-0.42 | 0.07-0.41 | 0.06-0.33 | 0.59-0.71 | 0.55-0.71 / 0.39-0.57 | 0-2 | |
| 100x100 | 0 | 0 | 0 | 0 | 0.48-0.52 | 0.48-0.58 / 0.43-0.50 | 0 | 54x / 5.7x / 1.3x |

- Unbinned works (the 2 e- loss at 1e-4 is the known physics limit: pairs of dark electrons). **Binned, the few-electron
  efficiency collapses** (10x10: 0-26 %; 100x100: 0) although an event's charge sits in one bin over 0.12 e- noise:
  the finder and A_GMX expect events spread over pixels (diffusion prior 0.1-1.3 px), not single bins.
- X-rays 50-85 % binned (95-99 % unbinned): the size rules; 100x100 (6 x 20 bins) goes to 'unmodelled'.
- **Fakes ~0 everywhere:** the losses are conservatism / a wrong event model, not noise.
- **Events split at an image boundary: 0 % found as such** (stitching, 1A.4).
- **Dark charge per image: up to 77x too high at 1e-6** (the fit's lower limit, few dark electrons per image):
  pooling over the images (1A.3) is needed.

**Goal (2026-11-13, the group): A_GMX (+ the Bayesian finder) optimal for every image type, binned or not; threshold /
seed clustering only as a fallback once every other way is exhausted.** A threshold run (seed 1.5 e-, pixel 0.5 e-)
is kept as the bar to beat.

**Oracle ceiling** (`wip/binned/ceiling_lowbg.py`, results in `wip/binned/results/ceiling.md`; F = 0.08 fakes / Mpix;
noiseless counts, everything true): at 1e-6 and 1e-5 e-/pix essentially **100 % of 2-10 e- events are detectable at
every binning** (only 100x100 at 1e-5 for 2 e-: 0.17). A_GMX today: 2 e- 0.75-0.78 unbinned, 0-0.56 binned; the
seed-1.5 threshold 0.22-0.81. At 1e-4 the limit is physical for 2 e- (0.07-0.54 binned) and at 100x100 for 3-5 e-
(dark-charge patterns): there only pooling and the event rate help. Target for 1A.2: ~100 % at <= 1e-5 for all
binnings at <= 0.1 fakes / Mpix.

**Step 1A.2, first part (2026-11-13): A_GMX's Bayesian stage bin-aware, and the program's pooling fixed.**
1. finder: the event model per axis (diffusion width / bin size; centre grid floored at 0.02 bin), linking radius per
   axis (adjacent bins always linked), halo in unbinned pixels; adaptive_cluster(binning=), set by the engine from
   NPBIN / NSBIN.
2. agmx_switch: the fake budget F per Mpix of UNBINNED area (was per bin: 100x too strict at 10x10); the "only >= 3 e-
   above LAM_2E" rule scaled to the same chance rate per unbinned Mpix (LAM_2E x sqrt(b_y b_x)).
3. engine: A_GMX's edge margins in unbinned pixels (ceil((first_row + 3) / b_y), ceil(2 / b_y) rows, ceil(2 / b_x)
   columns).
4. engine: the FULL run-pooled mode of the switch (the run's P(event) cut and noise model too, as VALIDATION.md
   recommends); before, each image's own cut and noise fit (0.997 vs 0.21 at 10x10) dropped single-pixel / single-bin
   events. This also improves unbinned images.
Unbinned: DM09 identical (12/12, auto = agmx = run_WADERs); injected events on L2 58 -> 59 found.

| (2 / 3 / 4-5 / 6-10 e-) | ceiling | A_GMX before | A_GMX now | seed 1.5 |
|---|---|---|---|---|
| 1x1 1e-6 | 1.00 all | 0.78 / 0.92 / 0.93 / 0.99 | **1.00 / 1.00 / 0.95 / 0.99** | 0.22 / 0.42 / 0.74 / 0.92 |
| 1x1 1e-4 | 0.80 / 1.00 | 0.08 / 0.86 / 0.96 / 0.99 | **0.68 / 0.91 / 0.98 / 0.99** | 0.16 / 0.50 / 0.69 / 0.96 |
| 1x10 1e-5 | 1.00 all | 0.26 / 0.61 / 0.92 / 0.98 | **0.95 / 0.96 / 0.94 / 0.98** | 0.37 / 0.74 / 0.94 / 0.99 |
| 10x1 1e-5 | 1.00 all | 0.15 / 0.50 / 0.80 / 0.85 | **1.00 / 0.94 / 0.94 / 0.96** | 0.52 / 0.67 / 0.91 / 0.98 |
| 10x10 1e-6 | 1.00 all | 0.00 / 0.11 / 0.16 / 0.26 | **0.93 / 0.89 / 1.00 / 0.96** | 0.81 / 1.00 / 1.00 / 0.99 |
| 10x10 1e-5 | 1.00 all | 0.00 / 0.00 / 0.15 / 0.20 | **0.84 / 0.93 / 0.92 / 0.88** | 0.77 / 0.93 / 0.98 / 0.98 |
| 1x100 1e-5 | 1.00 all | 0.12 / 0.29 / 0.31 / 0.65 | 0.72 / 0.79 / 0.75 / 0.83 | 0.34 / 0.75 / 0.91 / 0.96 |
| 100x1 1e-5 | 1.00 all | 0.00 / 0.06 / 0.07 / 0.06 | 0.48 / 0.67 / 0.64 / 0.66 | 0.52 / 0.64 / 0.98 / 0.98 |
| 100x100 | 1.00 at 1e-6 | 0 | 0 | 0.28-0.60 |

Fakes 0-4 per setting (<= 0.08 / Mpix). X-rays binned 0.92-0.98 (were 0.81-0.85) except 100x1 / 100x100 (0.4-0.7);
muons whole 0.64-0.91 at 10x10 / 10x1 / 1x100. Full table: wip/binned/results/ceiling_after_1A2.md. Open: 100x100
(images of 6 x 20 bins: nothing found), 100x1 and 1x100 (0.5-0.85), 4-10 e- at 1x100 / 100x1 below the threshold
method, muons whole, the safety rule; the 2 e- limit at 1e-4 is physical.

1A.2 **Bin geometry in the few-electron finder and A_GMX:** an event is point-like in bins (diffusion sigma / b per
axis, anisotropic for 1x100 / 100x1), the dark charge per bin, block sizes and halo in bins; strips of a few rows or
columns as 1-D.

1A.3 **Pooling over all images of a setting by default:** the dark charge per CCD, row and column; hot pixels and
columns (persistent across images, masked); the noise model (asymmetric tails); the false-alarm rate from the
candidates of all images (the Bayesian finder's run mode, made the default for >= 2 images).

1A.4 **Stitching.** Stacking: the pooling above, and an optional summed image of a setting for faint persistent
features. Continuous readout: consecutive images joined end to start (by image number and readout times), so an event
read across a boundary is one cluster, and the edge rows are no longer lost.

1A.5 **Gain:** at these dark currents there is no comb; the gain comes from a calibration image of the same CCD
(LED / high dark current / unbinned run), stored per CCD and used automatically (today `--gain` / `--gain-file`).

### 1B. High occupancy and the other robustness items (Fe-55 benchmark)

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

### Baseline (2026-11-13, `wip/binned/bench_binned.py`, DM09 Image_5 binned in software, current code)

| setting | K-alpha [keV] | clusters > 1 keV (truth 10764) | truth found | fakes | chain |
|---|---|---|---|---|---|
| 1x1 (re-noised) | 5.805 | 10693 | 98 % | 73 | A_GMX |
| 2x1 | 5.780 | 9594 | 54 % | 550 | A_GMX / unmodelled |
| 2x2 | 5.928 | 9949 | 56 % | 900 | crowded / unmodelled |
| 5x5 | 5.835 | 5050 | 29 % | 234 | crowded / threshold |
| 10x10 | 5.783 | 1136 | 6 % | 13 | crowded (core fill 0.9) |

**Root cause: the gain and dark-current fit, not the clustering.** With > 1 e- per bin the fit takes the first two
peaks as 0 and 1 e- under a Poisson law: gain 1.49 instead of 1.34 ADU/e- (2x2 and 5x5), dark charge 0.8-1.0
instead of 2-3 (2x2) and 12-17 e-/bin (5x5); the read noise then looks like 0.5-2 e- and the chain follows from it.
So step 1.0 comes first.

### Plan

1.0 **Gain and dark charge at high occupancy.** Gain from the spacing of the electron comb over many peaks (visible
at 0.15 e- noise up to tens of e- per bin), dark charge from the mean charge per bin / gain (not the 0/1 e- peak
ratio); the existing fit only where the occupancy is low. Measured on the benchmark (gain, dark charge, then K-alpha).

**Step 1.0 done (2026-11-13)** (`autoconfig_panaSKImg.high_occupancy`, `native_pipeline.gain_noise_dc`; only CCDs
with a median charge >= 2 e- per pixel / bin and a resolved comb; unbinned DM09 clusters identical, 12/12):

| setting | gain / truth | noise [e-] | K-alpha [keV] (truth 5.799) | found (before) | fakes (before) | chain now |
|---|---|---|---|---|---|---|
| 2x2 | 1.0002 | 0.152 | 5.803 (5.928) | 80 % (56 %) | 1028 (900) | crowded |
| 2x1 | 1.0014 | 0.152 | 5.785 (5.780) | 79 % (54 %) | 729 (550) | |
| 5x5 | 1.0000 | 0.151 | 5.822 (5.835) | 45 % (29 %) | 734 (234) | crowded |
| 10x10 | 1.0000 | 0.153 | 5.806 (5.783) | 6 % (6 %) | 16 (13) | threshold |
| 1x10 / 10x1 | 1.0000 | 0.151 | 5.795 / 5.801 | 40 % / 33 % | ~1200 | crowded |
| 1x100 / 100x1 | 1.0000 | 0.37 | - / 5.777 | 0 % / 2 % | 2 / 17 | threshold |

The gain is now right everywhere (comb phase, held) and K-alpha within 7 eV up to 10x10 (was +130 eV at 2x2). What
remains is the clustering: crowded mode on every binned CCD (A_GMX's background at several e- per bin, the core-fill
trigger), threshold on 10x10 and the strips; fakes ~700-1300. The fitted dark charge is 0.3-0.9 x the true one (the
histogram is a mixture over the CCD): report the truncated-Poisson estimate instead (to do with 1.1a).

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
