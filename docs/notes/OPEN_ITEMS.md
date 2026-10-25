# Open items (single list, updated 2026-10-25)

Collected from the dated notes in this folder; each item says where its details are. Remove an item here when it
is done (and say so in the note of that day).

## Pipeline and gain correction
- **Gain-correction coverage**: about 20 % of CCDs get no reliable g(n) (the comb fades after 1-3 charge bins).
  Ideas: reuse a higher-statistics image of the same module/day; per-row-block baseline correction.
  (CONTEXT_2026-10-16, section 7.)
- **Residual -0.2 to -0.7 %** after the correction: charge in faint cluster edges below ClusterFinder's threshold
  (back side loses more); not a gain issue. A_GMX recovers it. (CONTEXT_2026-10-16.)
- **High-charge gain**: above about 2000 e- the curve is extrapolation only; coherent pooling on AMT EXT4 moved the
  Am-241 lines down by 0.4-0.7 %. (HANDOFF_2026-10-23.)
- **Windowed images** are supported but the correction is less reliable; **PD06/DM12 Am-241 (2026-03-28)**: 5 of 6
  CCDs have no resolved peaks and are not corrected. (CONTEXT_2026-10-16.)
- **10000-skip image**: repeat the measurement (and a 5000-skip image right after) to separate "more skips" from
  "worse conditions". (CONTEXT_2026-10-16.)
- Not implemented on request: robust row exclusion when the overscan is contaminated in general.
- panaSKImg issues noted, not fixed (read-only): SetParameters 6-of-7 bug, TF1 100-point drawing, `.fits.bz2` name
  parsing (our scripts decompress), ACM ccd_cols conventions.

## Clustering (A_GMX and the study)
- Inspect A_GMX labels / track merging by eye on real Am-241 tracks (only injections so far). (CONTEXT_2026-10-24.)
- Exact pixel integrals for very sharp clusters (the sum + 1/12 approximation moves a few % of a sharp X-ray's charge
  to a diffuse neighbour). (CONTEXT_2026-10-24.)
- A background model for the PD06/DM12 "unmodelled" images, so that GMX can run there (pairs are not resolved now).
- Crowded CCD: re-optimise the sigma_xy cut for A_GMX's width; the 25-28 keV structure (dev/ndf about 2).
- Low-background few-electron work: validated on simulated images and injections into one real CCD; a check on a
  real low-background science image (few-electron rate vs the dark-current expectation) is still to do. In the A_GMX
  files `has_seed` keeps WADERs' meaning (30 sigma seed), so few-electron events have `has_seed = 0`: select them
  with `agmx_cls == 3`.
- The A_GMX performance numbers in docs/aGMX were made before the 2026-10-25 audit fixes (mask-border seeds,
  residual few-electron test); they change only events touching the mask and high-dark-current CCDs, and were not
  re-run.
- Non-skipper (high read noise) images: the read-noise estimate saturates near 0.37 e- and the background model can
  mistake Gaussian noise for dark charge; A_GMX is not meant for them (no guard yet).
- Duplicated helpers in docs/am241_note (propagate, blend_fwhm, gfit, znp, fit_line) could move to common.py.
