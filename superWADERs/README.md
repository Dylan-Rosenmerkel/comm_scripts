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
| fits and clusters | `run_superWADERs images.txt [--gain-correction] [--composite --module-list modules.csv]` | `root/` (+ `root_gaincorr*/`), `summary.txt`, `fit_quality.csv`, plots, `characterisation/` |
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
  commissioning/        raw_data (+ raw_levels/), packet_report (+ template), dc_series, defects (+ lsm_defects/),
                        sr_images, compare_dcfit_vs_skips
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
- **Not yet run with this command on real data** (only through `wip/native_pipeline`). Before relying on it, a
  first run on known data is worth comparing with `run_WADERs --engine native --clusterer agmx`, e.g. DM09 Image_5.
  The open points of the front end are in `wip/native_pipeline/ANYIMAGE.md`:
  - the crowded-mode cut 0.10 (PD06 sits at it);
  - binned images (clustering not adapted to the bin size);
  - skip-resolved data (synthetic file only);
  - sibling gains;
  - the Bayesian switch's halo and block-boundary tests.
