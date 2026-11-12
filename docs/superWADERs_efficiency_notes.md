# superWADERs: efficiency notes (2026-11-12) — recorded, not applied

Found in the audit of 2026-11-12. **None of these changes is applied.** They are kept here, with the measurements,
until they have been checked on real data.

Profiled on synthetic images of 4 CCDs × 500 × 1300 with 4 cores. PyROOT was not available, so ROOT files were
written by a stand-in; all other code paths are the real ones.

## Where the time goes (3 images, one process)

| stage | time | notes |
|---|---|---|
| front end + run pooling | 38 s | Bayesian `prepare_run` 20 s, the pooling pre-pass's fits 15 s, characterisation 3 s |
| final pass | 272 s | A_GMX `background_map` (via `characterise.decide_ccd`) **166 s (61 %)**, A_GMX clustering 76 s, pedestal 9 s, cluster table 5 s |

Inside `background_map`:
- **`fit_comb`** (95 s): a Nelder-Mead comb fit on noisy rows, calling `scipy.stats.poisson.pmf` 550,000 times. The
  cost is the distribution object built on every call.
- **`smooth_local_linear`** (21 s): a pure-Python O(n²) LOWESS loop.
- **`truncated_poisson_mle`** with `stats.poisson.isf`: 18 s.

On a CCD with 1.5 e- read noise, `background_map` alone took 25 s, against about 5 s on an electron-counting CCD.

## Ready, not applied: `docs/patches/superWADERs_efficiency_E1-E3.patch`

Apply with `git apply docs/patches/superWADERs_efficiency_E1-E3.patch`, after checking on real data.

| # | change | file | measured effect | outputs |
|---|---|---|---|---|
| E1 | Poisson pmf / isf as closed forms (`pois_pmf`, `pois_isf`): the formulas scipy evaluates, without the object overhead | `clustering/agent_methods.py` | part of the 2.8× below | pmf bit-identical; isf identical on 3000 λ values |
| E2 | LOWESS (`smooth_local_linear`) as five `np.correlate` passes instead of a Python loop over rows | `clustering/agent_methods.py` | 7.6 s → 0.003 s for a 3500-row CCD | equal to 6e-15 |
| E3 | no `background_map` when the read noise is ≥ 1 e-: A_GMX then takes the 'unmodelled' path on the read noise alone and never uses the map | `engine/characterise.py`, `clustering/adaptive.py` | 1.5 e- noise CCD: 25 s → 0.2 s | clusters identical; only the recorded background spread / λ change |

E1 + E2 together: `background_map` on an electron-counting CCD goes from 5.3 s to 1.9 s.

End to end (3 images, `-j 2`, `--sr`, with the former SR re-fit): 182 s → 121 s.

**Not bit-identical.** E2 changes the background map at the 1e-15 level.
- Cluster energies then differ by 1e-14 to 1e-9 (relative).
- On one crowded CCD, 2597 → 2595 clusters, with the total energy equal to 0.1 keV in 136,674 keV: a floating-point
  near-tie in a split.
- Before applying, compare a real run (DM09 Image_5, an Fe-55 Image_8) with `utils/compare_runs.py`, and accept the
  result as the new reference or add a tolerance.

**Already resolved (E4):** `--sr` fitted every image again. Since the campaign design, the SR part is computed in the
worker from the arrays of the main pass. This also removed an inconsistency: SR had run without the front end, so a
'quiet' CCD was missing from its output and a CCD without a comb got a different dark charge.

## Proposed, not prototyped (largest first)

| # | what | expected gain | risk |
|---|---|---|---|
| P1 | **A_GMX `count_threshold`** simulates 10-50 blank copies of every CCD to set the few-electron threshold (30-40 % of the remaining A_GMX time). Cache by (λ, s) bins across images of one run setting, or draw only the occupied pixels. | ~2-3 s per CCD | random draws change, so thresholds change statistically: needs validation |
| P2 | **Fits done twice.** The pooling pre-pass and the `--gain-correction` pre-pass run the full pedestal and fits, then the final pass does them again. Hand the pre-pass results (pedestal, fits, calibrated image) to the final pass. | ~1/3 of the non-clustering time per image | low |
| P3 | **Parallel CCDs** within an image. One big image (packet, full-frame 1-skip) is processed one CCD at a time. | up to 4× on single images | memory |
| P4 | **Gaussian row pedestal:** `least_squares` with a numerical Jacobian, ~1500 calls per image (~3 s). Use the analytic Jacobian, or one vectorised fit for all rows. | ~2-3 s per image | panaSKImg agreement must stay |
| P5 | **Repeated reads.** The worker reads the image again for `ccd_metrics` and `packet_report.image_data`; the defect map reads each image of a group about 5 times. Pass the arrays already in memory. | I/O on big files | low |
| P6 | **Bayesian `prepare_run`** runs the full finder even where every block later goes to A_GMX. | depends on the run | low |

## Campaign mode (since 2026-11-12): costs of an append

Each run processes only the new inputs, but rebuilds every page from the records. Measured on a synthetic campaign
(26 modules, 104 CCDs):

| run | time |
|---|---|
| 13 new images (packet of 13 modules) | 80 s |
| no new image, `--dc-series --defect-map` re-rendered (criteria changed) | 41 s |

Of the 41 s, about 38 s is `dc_report.pdf` (matplotlib, one page per CCD), written again on every run. The other pages
take seconds.

Proposals:
- **dc-series PDF:** write `dc_report.pdf` only when the series changed (a new image in the dark-current tool), or one
  PDF per module, written when that module changes.
- **Pages:** write module pages only for modules with new data. The overviews must always be rewritten.
- **Records:** loaded whole (gzip JSON). A campaign of thousands of images would want an index of the per-CCD
  numbers, so the overviews don't open every record.
