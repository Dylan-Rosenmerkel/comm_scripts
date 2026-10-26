# clustering_tests — alternative clustering methods and their test scripts

Research code. A_GMX from here is used by `run_WADERs --clusterer agmx|both` and `bin/run_AGMX` (through
`src/agmx_root.py`); everything else is not part of the pipeline. It holds every clustering method compared with the WADERs /
panaSKImg ClusterFinder on 2026-10-23..25 and the scripts that tested them.

- **Recommended method: A_GMX** = `adaptive.adaptive_cluster(Q, valid, engine="gmx")`. Full description (every step,
  equations, figures, performance): `docs/aGMX/aGMX_clustering.tex (Overleaf: docs/aGMX/aGMX_clustering_overleaf.zip)`.
- All methods and all results: `docs/clustering_methods_report.pdf` (built by `make_clustering_report.py`).
- Context / hand-over: `docs/notes/CONTEXT_2026-10-24.md`.
- Saved results: `~/Documents/comm_scripts_data/session_2026-10-23_clustering/clustering_2026-10-25/`.

Scripts read data from `~/Documents/...` (absolute paths on this machine) and write their outputs (JSON / npz /
png / logs, git-ignored) next to themselves; the documentation builders, `am241_lines_flags.py` and
`summ_wide_flags.py` write into `../docs/aGMX/`, `summ_wide.py` writes the committed `wide_summary.md`.
`am241_paths.py`: the runs_29img analysis folder used by the crowded-CCD scripts (its readers, outside the repo) with
the repo's `docs/am241_note/amfit.py` for the constrained fit. Python: the `damic` conda env. Fast unit test (simulated image, no
data): `python -m unittest tests.test_clustering_methods` (from the repo root).

## Methods (modules)

| file | method | one line |
|---|---|---|
| `cluster_methods.py` | WADERs-like, B auto | hysteresis clustering (`wader`), topological clustering in electrons (`topo`), injection (`inject`) |
| `agent_methods.py` | MFD+G | background map, Poisson tails, matched-filter detection + disk + growth (`mf_detect`), Cash / Naylor / multi-fit (`pipeline`) |
| `methods2.py` | hybrid v1 / v2 | merges of MFD+G pieces (B region, saddle, `mfdg_hybrid`), track-first, counting energies |
| `mhc.py` | MHC | multi-hypothesis clustering: specialised finders + chi2 / BIC arbitration + classes |
| `gmx.py` | GMX | Gaussian-mixture EM deblending (`general_cluster`) - from the exploratory agent, verified; quality flags `blend`, `ped`, `peakfrac`, `chi2` |
| `pws.py` | PWS | persistence-simplified watershed (control in the GMX study; not recommended; its test harness is in the data folder, `clustering_2026-10-25/general_agent/`; import it from this folder) |
| `adaptive.py` | adaptive / **A_GMX** | regime switch: 'unmodelled' -> hysteresis; engine 'mhc' (crowded -> hysteresis, sparse -> MHC) or 'gmx' (GMX everywhere) |
| `fe_candidates.py` | few-electron candidate catalogue | runs after A_GMX (low-background / DAMIC-M style): every 2-20 e- group with its null statistic and expected background, mask flags (halo, trails, hot columns, edge, A_GMX clusters), per-image single-electron rate and expected dark-charge counts per class; written by `src/agmx_root.py` to `fe_candidates/` for electron-counting images |

## Test scripts

| script | test | data |
|---|---|---|
| `wide_test.py` (+ `summ_wide.py`), `wide_run.py adaptive / GMX / adaptive+GMX` | every method on every WADERs-calibrated image on the machine (878 image-CCDs): injection + real Ka / 59.5 keV | all data sets |
| `run_study.py` | the earlier study's loader and B / WADERs comparison (`study.json`); imported by `compare.py` and `wide_test.py` | DM09 / DM10 |
| `adaptive_validate.py` (+ `run_gmx_validate.sh`, `run_adaptive_v2.sh`, `run_rest.sh`) | tracks, few-electron, Fe-55 and crowded-CCD validation of MHC / adaptive / GMX / A_GMX | AMTEst, simulated, DM09/DM10, runs_29img |
| `compare.py` (-> `results/` or `$RESULTS_DIR`), `mhc_fe55.py` (-> `results_fe55/`, read by `summ_fe55.py`) | Fe-55: injection at 4 widths, pairs, real Ka | DM09 / DM10 Image 5 |
| `am_test.py` | Am-241: real lines, injected 17.75 / 59.5 keV, big-event splitting | AMTEst 10 images |
| `tracks_test.py` (-> `results4/`, read by `summ_tracks.py`), `extscan.py` (-> `extscan.json`: hybrid B-region size scan), `mhc_tracks.py` (+ `summ_mhc.py`) | injected muons, worms, blobs, alphas, hot columns, X-ray pairs | AMTEst EXT1 / EXT3 |
| `lowE_test.py`, `mhc_lowE.py` (+ `summ_lowE.py`) | few-electron events: efficiency, exact charge, fakes | simulated + AMTEst EXT1 |
| `agmx_root_check.py` | `src/agmx_root.py` writes the WADERs format: real panaSKImg clusters replayed through the writer, every branch compared | runs_29img |
| `am241_29img_recluster.py`, `am241_29img_fit.py`, `adaptive_tune_am241.py`, `adaptive_thr_scan_am241.py`, `adaptive_crowded_scan_am241.py`, `make_crowded_table.py` | crowded Am-241 CCD: constrained Lepy fit for every clustering, occupancy / threshold scans | runs_29img EXT4 |

## Documentation builders

| script | output |
|---|---|
| `make_agmx_figures.py` | `docs/aGMX/figures/fig00-13` (every A_GMX step run on a real image) |
| `make_agmx_perf.py` | `docs/aGMX/figures/fig20-26`, `docs/aGMX/perf_tables.tex` |
| `make_clustering_report.py` (+ `report_text_methods.py`, `report_figs.py`) | `docs/clustering_methods_report.pdf` |
| `make_agmx_contamination_figs.py` | `docs/aGMX/figures/fig30-34` (the 59.5 keV tail, its cause, the quality flags) |
| `make_overleaf_zip.py` | `docs/aGMX/aGMX_clustering_overleaf.zip` |

## 2026-10-25 additions

- `gmx.py`: quality flags `ped` (local pedestal), `peakfrac`, `chi2` on every cluster; satellite merging only for
  fragments < 1 keV (`sat_e_max`). Energies unchanged. Why: docs/aGMX, section "Contamination".
- `am241_lines_flags.py`: every Am-241 line fitted alone, old vs new code, with / without the flag cuts (EXT1/3/4).
- `wide_agmx_flags.py` (+ `summ_wide_flags.py`): Fe-55 K-alpha of every data set with the flag cuts, and the injection
  test, for the new code.

## 2026-10-25 audit

- `gmx.py`: peak search ignores masked pixels (X-rays on the mask border were lost to the mask); residual
  few-electron seeds tested against the right background; `fill_flags` gives every cluster the quality flags.
- `adaptive.py`: extra keywords go to the engine; unknown engine / `return_pixels` with MHC raise.
- One `simple_bg` (agent_methods); `wide_run.py` replaces `wide_adaptive.py`, `wide_gmx.py`, `wide_adaptive_gmx.py`.
- Details in the git log ("Audit part A-E") and `docs/notes/OPEN_ITEMS.md`.
