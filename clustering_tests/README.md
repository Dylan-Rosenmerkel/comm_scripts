# clustering_tests — alternative clustering methods and their test scripts

Research code (not used by `run_WADERs`). It holds every clustering method compared with the WADERs /
panaSKImg ClusterFinder on 2026-10-23..25 and the scripts that tested them.

- **Recommended method: A_GMX** = `adaptive.adaptive_cluster(Q, valid, engine="gmx")`. Full description (every step,
  equations, figures, performance): `docs/aGMX/aGMX_clustering.tex (Overleaf: docs/aGMX/aGMX_clustering_overleaf.zip)`.
- All methods and all results: `docs/clustering_methods_report.pdf` (built by `make_clustering_report.py`).
- Context / hand-over: `docs/CONTEXT_2026-10-24.md`.
- Saved results: `~/Documents/comm_scripts_data/session_2026-10-23_clustering/clustering_2026-10-25/`.

Scripts read data from `~/Documents/...` (absolute paths on this machine) and write their outputs (JSON / npz /
png / logs, git-ignored) next to themselves. Python: the `damic` conda env. Fast unit test (simulated image, no
data): `python -m unittest tests.test_clustering_methods` (from the repo root).

## Methods (modules)

| file | method | one line |
|---|---|---|
| `cluster_methods.py` | WADERs-like, B auto | hysteresis clustering (`wader`), topological clustering in electrons (`topo`), injection (`inject`) |
| `agent_methods.py` | MFD+G | background map, Poisson tails, matched-filter detection + disk + growth (`mf_detect`), Cash / Naylor / multi-fit (`pipeline`) |
| `methods2.py` | hybrid v1 / v2 | merges of MFD+G pieces (B region, saddle, `mfdg_hybrid`), track-first, counting energies |
| `mhc.py` | MHC | multi-hypothesis clustering: specialised finders + chi2 / BIC arbitration + classes |
| `gmx.py` | GMX | Gaussian-mixture EM deblending (`general_cluster`) - from the exploratory agent, verified |
| `pws.py` | PWS | persistence-simplified watershed (control in the GMX study; not recommended) |
| `adaptive.py` | adaptive / **A_GMX** | regime switch: 'unmodelled' -> hysteresis; engine 'mhc' (crowded -> hysteresis, sparse -> MHC) or 'gmx' (GMX everywhere) |

## Test scripts

| script | test | data |
|---|---|---|
| `wide_test.py` (+ `summ_wide.py`), `wide_adaptive.py`, `wide_gmx.py`, `wide_adaptive_gmx.py` | every method on every WADERs-calibrated image on the machine (878 image-CCDs): injection + real Ka / 59.5 keV | all data sets |
| `adaptive_validate.py` (+ `run_gmx_validate.sh`, `run_adaptive_v2.sh`, `run_rest.sh`) | tracks, few-electron, Fe-55 and crowded-CCD validation of MHC / adaptive / GMX / A_GMX | AMTEst, simulated, DM09/DM10, runs_29img |
| `compare.py`, `mhc_fe55.py` (+ `summ_fe55.py`) | Fe-55: injection at 4 widths, pairs, real Ka | DM09 / DM10 Image 5 |
| `am_test.py` | Am-241: real lines, injected 17.75 / 59.5 keV, big-event splitting | AMTEst 10 images |
| `tracks_test.py`, `extscan.py` (+ `summ_tracks.py`), `mhc_tracks.py` (+ `summ_mhc.py`) | injected muons, worms, blobs, alphas, hot columns, X-ray pairs | AMTEst EXT1 / EXT3 |
| `lowE_test.py`, `mhc_lowE.py` (+ `summ_lowE.py`) | few-electron events: efficiency, exact charge, fakes | simulated + AMTEst EXT1 |
| `am241_29img_recluster.py`, `am241_29img_fit.py`, `adaptive_tune_am241.py`, `adaptive_thr_scan_am241.py`, `adaptive_crowded_scan_am241.py`, `make_crowded_table.py` | crowded Am-241 CCD: constrained Lepy fit for every clustering, occupancy / threshold scans | runs_29img EXT4 |

## Documentation builders

| script | output |
|---|---|
| `make_agmx_figures.py` | `docs/aGMX/figures/fig00-13` (every A_GMX step run on a real image) |
| `make_agmx_perf.py` | `docs/aGMX/figures/fig20-26`, `docs/aGMX/perf_tables.tex` |
| `make_clustering_report.py` (+ `report_text_methods.py`, `report_figs.py`) | `docs/clustering_methods_report.pdf` |
