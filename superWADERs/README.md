# superWADERs

`run_superWADERs` is a separate program from `run_WADERs`. It has its own engine (pedestal, fits, clustering,
panaSKImg-format cluster files) and its own commissioning tools. It needs no panaSKImg. It does not use
`bin/run_WADERs`, `src/`, `clustering_tests/` or `wip/`: every module it runs is in this folder
(`tests/test_superwaders.py` checks this).

```bash
run_superWADERs INPUT [-o PARENT] [-j N] [options]   # INPUT: an image, a raw DAQ file, or a .txt list of them
run_superWADERs -h                                   # the flags;   -h -v: the full guide
```

`bin/run_superWADERs` (on your PATH with the rest of `comm_scripts/bin`) starts `superWADERs/run_superWADERs`. Results
go to `PARENT/superWADERs_<run name>/`; read `summary.txt` first. Python: the `damic` conda environment (numpy,
astropy, scipy, numba, scikit-image, matplotlib, ROOT), found automatically (`--env NAME` or `$SUPERWADERS_ENV`).

## Commissioning campaigns: packet by packet (`--campaign DIR`)

Data arrive in packets, so a campaign folder grows run by run. Each run:
- processes only the inputs the campaign does not have yet (or whose file changed);
- keeps their results in `records/`;
- rebuilds the pages of the tools it was given (packet, dark current, defect maps, SR, and the images overview when
  images were processed) from everything the campaign holds;
- rebuilds `index.html`, which combines every tool per CCD.

```bash
run_superWADERs packet_DM01.txt --packet --campaign ~/comm/LSM2026 --module-list modules.csv   # first packet
run_superWADERs packet_DM02.txt --packet --campaign ~/comm/LSM2026 --module-list modules.csv   # adds DM02 only
run_superWADERs series_day3.txt --dc-series --campaign ~/comm/LSM2026                         # dark current: new images only
run_superWADERs images.txt --defect-map --campaign ~/comm/LSM2026      # only defect groups with a new image are redone
run_superWADERs any.txt --campaign ~/comm/LSM2026 --criteria limits.json   # new limits: every page re-judged, no fits
```

- **Inputs:** an input is known by its path, size and modification time; a changed file is processed again.
- **Several tools:** an input given to another tool later gets only that tool's part (its packet viewer, its SR
  streaks, cluster files).
- **Untouched tools:** they keep their pages, and `index.html` still combines them. An unchanged rerun takes about
  1 s.
- **Flags:**
  - `--refresh-all` rebuilds every tool's pages;
  - `--reprocess` processes the given inputs again;
  - `--criteria` replaces the campaign's limits and re-judges everything.
- **Per run only:** the run-pooled dark charge of the few-electron finder, `--gain-correction` curves and `--composite`
  use the images of the current run only.

**Without `--campaign`, a run stands alone**, as before: a fresh folder (`superWADERs_<run name>/`) with only its
inputs and the tools asked for, the same pages and overviews, no comparison with anything else. Every tool can be run
this way, alone or together: `--packet`, `--dc-series`, `--defect-map`, `--sr`, or plain images.

**Where everything is.** Every page of a run or campaign is in its folder, and the pages link to each other with
relative links. Open them from the folder, and copy or zip the whole folder to share it; a page copied or opened on
its own cannot follow its links.

```
LSM2026/                        the --campaign folder (or superWADERs_<run name>/)
  index.html                    campaign overview: every tool per module and CCD; the CCD panel links to each tool's
                                overview (opened at that CCD) and to the CCD's full reports
  summary.txt                   verdict counts, fits per CCD, every run of the campaign
  images/index.html             images overview; full reports: the DC-fit PDFs (*_FitDarkCurrentProcess_DCfit_EXTn.pdf)
  packet/index.html             packet overview;  full reports: packet/<module>_ACM<id>.html
  dc_series/index.html          dark-current overview;  full reports: dc_series/<module>_ACM<id>_dc.html, dc_report.pdf
  defects/index.html            defect overview;  full reports: defects/<module>_ACM<id>_<size>/defect_map.html
  sr/index.html                 serial-register overview;  full reports: sr/<module>_ACM<id>_sr.html
  root/, L0/, L1/, logs/        cluster files, raw-data products, logs
  campaign.json, criteria.json, records/   the campaign's ledger, limits and kept results
```

Every full report links back to its overview ("← overview of all modules").

**The overview pages** (`index.html`, `<tool>/index.html`, self-contained, offline) share one layout:
- **Module × CCD grid.** Every cell shows the CCD's verdict and its main values, in amber / red when outside a limit,
  and the first problem:

  | page | values in the cells |
  |---|---|
  | images | read noise, gain, dark current, hot columns |
  | packets | read noise, single-skip noise, PSD σ_CDS, gain |
  | dark current | last and median rate, plateau, number of images |
  | defect maps | bright / dark columns, hot / cold pixel spots |
  | serial register | streaks per 1000 reads, read noise, dark charge |
  | campaign | every tool's main values, labelled with the tool (IM, PK, DC, DF, SR) |

- **"Cells show" selector:** turns the grid into a map of any one value over all modules and CCDs. Judged values are
  coloured by their limits, the others by a scale, with the min / median / max shown.
- **CCD panel:** click a cell for every value, the reasons, the history of the CCD's images and links to the full
  reports.
- **What needs attention:** findings grouped by type.
- **Distributions:** every judged metric over all CCDs, with the warn / fail limits.
- **Crosstalk matrices** (images).
- **Full table:** every value, sortable, filterable, with CSV download. The campaign page's table has every tool's
  values per CCD.
- **Criteria and definitions.**

The images overview has one view per image setting (e.g. 500-skip 1×1 and 1-skip 1×1).

**Verdicts** come from the criteria file the campaign keeps (`criteria.json`, a copy of
`commissioning/criteria_default.json` or of `--criteria FILE`):
- per metric: warn / fail limits, direction, `abs`; a limit of `null` is shown but not judged;
- per non-numeric check (no charge seen, gain not measured, fit at a limit, trace problems, ...): `fail`, `warn`,
  `note` (shown with the CCD, no effect on the verdict) or `off`.

**The default limits are placeholders** (the pages say so); the group should set them.

| tool | judged per CCD (default criteria) | page |
|---|---|---|
| images | read noise, gain vs module / nominal, dark current, readout charge (row-fit intercept), glow index, hot columns, fit χ²/ndf, baseline step / row-to-row rms, crosstalk (≥ 5σ), saturation; no charge seen, gain not measured, fit at a limit | `images/index.html` |
| packet | read noise (image with the most skips; unresolved ones too when they have enough skips); shown: single-skip noise, PSD σ and σ_CDS, gain, dark charge; trace quality, trace sums | `packet/index.html` |
| dark current | median, last, plateau rate, readout charge; still falling, above `--dc-limit` | `dc_series/index.html` |
| defect maps | bright / dark columns, hot / cold pixel spots; hot pixels not identifiable (note) | `defects/index.html` |
| serial register | streaks per 1000 reads, read noise; shown: dark charge, streak length | `sr/index.html` |

Image metrics (`commissioning/ccd_metrics.py`, computed once per image):
- **Baseline:** row-to-row rms, largest step (in ADU and in units of that rms), drift.
- **Columns:** hot columns against a running median over 31 columns; the glow index (dark charge of the last 10 % of
  the columns over the median).
- **Saturation.**
- **Crosstalk:** between the CCDs of a module, a slope with an intercept where the source CCD has > 200 e-; judged at
  ≥ 5σ.

## Clustering: chosen per CCD (`--clusterer auto`, the default)

Before anything is fitted, every image and CCD is characterised (`engine/characterise.py`):
- data form (skip-resolved raw files are compressed first) and geometry;
- dead / quiet channel and the first-column artefact;
- gain source (comb, `--gain`, `--gain-file`, other images of the same CCD, prior);
- noise, dark charge, electron counting.

After calibration, and before clustering, the chain is confirmed on the numbers A_GMX itself uses: read noise,
background spread and core fill (`characterise.decide_ccd`).

| CCD | chain |
|---|---|
| electron counting, sparse (core fill < 0.10) | **A_GMX** with the **Bayesian few-electron switch**: in sparse, low-dark-charge 100 x 100 blocks, A_GMX's small clusters (<= 20 e-, away from big deposits) are replaced by the Bayesian catalogue. The dark charge is pooled over the run when >= 2 images share the run setting and DC <= 1.5e-2 e-/pix |
| electron counting, crowded (core fill >= 0.10) | **A_GMX crowded mode** |
| read noise >= 1 e-, or a structured background | **A_GMX 'unmodelled'** (hysteresis); if > 30 % of the pixels are above 4.5 e-, the threshold clusterer instead |
| flooded (no resolved peaks, > 10 e-/pix) | **threshold clusterer** above the flooding level (per-row median, subtracted), at 10 / 30 x the charge width (per-row MAD of the active pixels), not the read noise |
| peaks not resolved, not flooded | **threshold clusterer** (`--thresholds`; sigma0 rule) |
| 'no CCD signal' but a sharp zero peak | kept as electron counting ('quiet': cannot be told from a very low dark charge) |
| no CCD signal | skipped |

The first active column is masked in the clustering region when it carries a significant artefact.

**Where each choice is recorded:** `summary.txt` ("clustering per CCD") lists every CCD's chain, why it was chosen,
its cluster count (and how many came from the Bayesian finder), and every fallback. The full record is in
`characterisation/<image>.json` and in each ROOT file (TNamed `characterisation`).

**Cluster file branches:** besides panaSKImg's branches and the `agmx_*` ones, each cluster has:
- `finder`: 0 threshold, 1 A_GMX, 2 Bayes;
- `p_event`, `N_post`, `sigma_post`: the Bayesian posterior; -999 for other clusters;
- `agmx_cls` 12 ('few-e') and `agmx_regime` 5 ('bayes') for Bayesian events.

`--clusterer agmx` or `--clusterer threshold` puts one clusterer on every CCD instead (with A_GMX's own crowded mode;
the Bayesian switch only runs in `auto`).

## Commands

| task | command | output (in the run folder) |
|---|---|---|
| fits and clusters | `run_superWADERs images.txt [--gain-correction] [--composite --module-list modules.csv]` | `index.html`, `images/index.html`, `root/` (+ `root_gaincorr*/`), `summary.txt`, `fit_quality.csv`, plots, `characterisation/` |
| add to a campaign | any of these with `--campaign DIR` | the same, for everything the campaign holds |
| CCDs without a comb | `... --gain 1.3` or `--gain EXT1=1.3,EXT4=0.82` or `--gain-file gains.json` | flagged in `summary.txt` |
| L0 traces / PSD traces | `run_superWADERs traces.txt [--l0-no-html]` | `L0/<trace>/` |
| L1 images (binned or not) | `run_superWADERs l1.txt --l1-skips 1,10,100,1000 [--l1-skip-first K]` | `L1/`: one FITS per N, noise vs N, diagnostics |
| commissioning packet | `run_superWADERs packet.txt --packet --module-list modules.csv` | `packet/` (one page per module) |
| dark current over a series | `run_superWADERs series.txt --dc-series [--dc-limit X] [--no-clusters]` | `dc_series/` |
| defect maps | `run_superWADERs images.txt --defect-map [--defect-sigma 5] [--no-clusters]` | `defects/` |
| serial-register images | `run_superWADERs sr.txt --sr [--sr-threshold 4] [--sr-min-length 3] [--no-clusters]` | `sr/` |

The commissioning tools (inputs, pages, what was checked, the data still needed) are described in
[docs/superWADERs_user_guide.md](../docs/superWADERs_user_guide.md).

## Layout

```
superWADERs/
  run_superWADERs       the command (driver: inputs, front end, g(n), processing, composite, tools, summary)
  engine/               native_pipeline.py (per image: pedestal, fits, calibration, clustering chain, ROOT files),
                        characterise.py (front end), native_kernels / native_rootio / native_plots,
                        autoconfig_panaSKImg, gain_dc_split, gain_vs_charge, fit_quality, naming, pair_energy
  clustering/           A_GMX (adaptive, gmx, agent_methods, crowded, crowded_shapes, mhc, methods2, cluster_methods)
                        and the Bayesian few-electron finder (agmx_switch, finder, noise, background, pipeline)
  commissioning/        campaign (ledger, records, criteria, verdicts, overview pages), worker (one image -> its
                        record), ccd_metrics, overview_template.html, criteria_default.json,
                        raw_data (+ raw_levels/), packet_report (+ module page template), dc_series, defects
                        (+ lsm_defects/), sr_images, compare_dcfit_vs_skips
  tests/                python -m unittest discover -s superWADERs/tests -v
```

Where the files came from (2026-11-12). They are now superWADERs' own and can change without touching `run_WADERs`:
- `engine/native_pipeline.py` and `characterise.py`: `wip/native_pipeline` (the any-image front end, 7980f24) with
  the later changes of `src/native_pipeline.py`;
- the other `engine/` modules: `src/`;
- `clustering/`: `clustering_tests/` and `wip/lowdc/bayes/`;
- `commissioning/`: `src/` and `utils/compare_dcfit_vs_skips.py`.

`raw_levels/` (ccd_opt_ml) and `lsm_defects/` (LSM_Module_Testing_Analysis) are kept in sync with their own
repositories by `utils/sync_from_ccd_opt_ml.py` and `utils/sync_from_lsm_analysis.py` (both update `src/` and
`superWADERs/`; `--dest superWADERs` updates this folder only).

## Efficiency

Recorded, not applied: [docs/superWADERs_efficiency_notes.md](../docs/superWADERs_efficiency_notes.md).
- **Ready as a patch:** three exact or near-exact speed-ups to A_GMX's background map (2.8× on electron-counting CCDs,
  100× on noisy ones), in `docs/patches/superWADERs_efficiency_E1-E3.patch`.
- **Larger items:** the few-electron threshold simulation, double fits, per-CCD parallelism.

## What has been checked

- **Engine (fits, pedestal, threshold clustering, ROOT format):** the same code as `run_WADERs --engine native`,
  validated against panaSKImg (`docs/native_pipeline.md`, `docs/native_pipeline_validation.md`).
- **The any-image front end and `--clusterer auto`:** `wip/native_pipeline/ANYIMAGE.md`: a suite of 16 cases, mostly real data (plus a few synthetic files).
  - DM09 Image_5: identical to A_GMX.
  - Fe-55 Image_8: K-alpha within 1 eV.
  - Injected few-electron events on real L2 noise: 58 / 64 found.
- **This command (2026-11-12), on synthetic images** (4 CCDs per image, each CCD a different case). PyROOT was not
  available, so the ROOT files were written by a test stand-in.
  - Each case went to its chain: Fe-55-like -> A_GMX; low dark charge -> A_GMX + Bayes (run-pooled over 3 images);
    dense tracks -> crowded mode; 1.5 e- noise -> unmodelled; flooded and unresolved -> threshold.
  - Also run: `--clusterer agmx` / `threshold`, `--gain-correction both`, `--no-clusters` with `--dc-series`
    `--defect-map` `--sr`.
- **Campaigns (2026-11-12, synthetic):**
  - 26 modules fed as a packet of 3 modules (`--packet`, 1-skip + 500-skip images), then two packets of 13, then
    `--dc-series --defect-map` over everything.
  - Each run processed only the new images; images first given as a packet got their cluster files when given as
    images.
  - The images overview gave 95 pass / 1 warn / 8 fail on the 500-skip view: the nine injected faults flagged, no
    healthy CCD flagged. The 4 ADU baseline step fails at 76× the row-to-row scatter; the 1-skip view (12 CCDs) all
    passes.
  - A new image of one module redid only that module's defect group; new criteria re-judged everything with no new
    fit; an unchanged rerun took about 1 s.
- **Not yet run with this command on real data** (only through `wip/native_pipeline`). Before relying on it, a
  first run on known data is worth comparing with `run_WADERs --engine native --clusterer agmx`, e.g. DM09 Image_5.
  The open points of the front end are in `wip/native_pipeline/ANYIMAGE.md`:
  - the crowded-mode cut 0.10 (PD06 sits at it);
  - binned images (clustering not adapted to the bin size);
  - skip-resolved data (synthetic file only);
  - sibling gains;
  - the Bayesian switch's halo and block-boundary tests.
