# Bringing comm_scripts into WADERs: plan (draft, 2026-11-02)

Branch `waders-integration`. `WADERs/` is an editable copy of pysimdamicm `waders_LBC` @ ed622e7
(git subtree, squashed) plus the local fixes from the working install (np.longdouble, TH1 SetDirectory(0),
FitDarkCurrentProcess.x_max_plot). The installed `~/Documents/Software/pysimdamicm` is untouched, so the
current run_WADERs keeps working while this is developed.

Update from upstream: `git subtree pull --prefix=WADERs https://gitlab.in2p3.fr/damicm/pysimdamicm.git waders_LBC --squash`.

## How WADERs is organised (what we plug into)

- `panaSKImg` (scripts/anaSKImg.py) loops over input images; for each one `ProcessManager.execute_process`
  runs the active processes in order of their `__sequence_id__`, per amplifier where `_per_amp` is set.
- A process is a class in `pysimdamicm/processes/` with `__name__`, `__sequence_id__`, parameters set from
  its JSON block (`set_parameters`, keys must be in `__units__`), and `execute_process(rawdata)` that reads
  and adds images on the raw-data object.
- It must be listed in `ProcessManager.__valid_processes__` to be usable from the JSON.
- Current order: compress 10, crosstalk 11, add background 15, pedestal 20, noise 21, Gaussian fit 22,
  calibration 25, 2D-Gauss 28, dark current 30-32, ClusterFinder 500 (writes `panaSKImg_clustersRec_*.root`
  through `OutputDataManager`), BuildClusterMask 550, ApplySelectionCuts 600; CTI/halo 510/511.

## Two layers

WADERs works one image at a time. Some comm_scripts steps need many images (the gain curve is measured on
all images of a module) or a whole run (composite, summary). So:

1. **Per-image processes inside panaSKImg** (configured by JSON, run by ProcessManager).
2. **A run-level driver** (today `bin/run_WADERs`) that configures, runs panaSKImg per CCD, measures across
   images and summarises. It moves into `WADERs/bin/` and calls the processes above.

## Where each comm_scripts piece goes

| comm_scripts | today | in WADERs | form |
|---|---|---|---|
| autoconfig_panaSKImg.py, gain_dc_split.py | step 1, writes a JSON per CCD | `pysimdamicm/autoconfig/`; `panaSKImg --autoconfig` builds the JSON in memory and saves the one it ran | config generator (keeps the JSON as the record) |
| gain_vs_charge.py: measure | step 3b, on saved calibrated images | run-level tool `bin/gain_vs_charge` → model JSON per module | needs all images of a module |
| gain_vs_charge.py: apply | step 3b, rewrites cluster ROOT files | new process `GainNonlinearityProcess` (seq 26, after calibration 25): x → n = A⁻¹(x) on the calibrated image; JSON key `model_file` | energies corrected at source; no ROOT rewriting |
| aGMX: gmx.py, adaptive.py, crowded.py, crowded_shapes.py | step 3d, then agmx_root.py converts | `pysimdamicm/agmx/` + `ClusterFinder.method: "agmx"` (or a sibling `AGMXClusterFinder`, seq 500, exclusive), writing clusters through the same `Cluster`/`OutputDataManager` | removes agmx_root.py; JSON keys `crowded`, `crowded_shapes`, … |
| fit_quality.py | step 4 | errors written by FitDarkCurrentProcess itself, or stays run-level | small |
| summary.txt, root_files.txt | step 3/4 | run-level driver | small |
| composite (MakeComposite_Fe55_Clusters_Extensions.cc) | step 3c | stays external (LSM repo, not ours); driver calls it | unchanged |
| naming.py, pair_energy.py | helpers | `pysimdamicm/utils/` | ε(T) used by the energy |

## Gain correction: one pass (done)

The g(n) curve is measured from the RAW images (gain_vs_charge uses autoconfig's own pedestal and gain), so it
can be measured before panaSKImg runs. `GainNonlinearityProcess` (after CalibrationProcess) attaches the curve;
ClusterFinder corrects the energies of the clusters it found with exactly the formula of `gain_vs_charge apply`
(WADERs clusters), or converts the image before clustering (A_GMX), as run_WADERs does. No second pass.

## Steps

0. ✔ Editable copy on `waders-integration`.
   ✔ (2026-11-02) WADERs/doc/comm_scripts_additions.md: GainNonlinearityProcess, bin/gain_vs_charge (with the
   g(n), nonlinearity and new A(n) plots), ClusterFinder `algorithm: "agmx"` (+ `agmx` options) and
   `spr_threshold_auto`; comm_scripts modules synced into pysimdamicm/comm/ by utils/sync_to_waders.py.
   Validated on L2 image 9 EXT1 (see that file). Merge request style: additive, defaults unchanged.
1. A development environment that imports `WADERs/` (e.g. a clone of `damic` with `pip install -e WADERs`),
   so the production `damic` env is not changed. Check panaSKImg on one image gives identical output.
2. `GainNonlinearityProcess` (smallest, self-contained). Regression: cluster energies equal to
   run_WADERs --gain-correction on DM09/DM10 Image 5 (front-side Fe-55) and the Am-241 set.
3. aGMX as a ClusterFinder method (numba becomes a dependency). Regression: agmx_root_check.py-style
   comparison against root_agmx/ on L2 image 9 and Am-241 EXT4.
4. Autoconfig inside panaSKImg; the saved JSON must reproduce the run.
5. Driver into `WADERs/bin/`; `comm_scripts/bin/run_WADERs` becomes a thin wrapper, then retires.
6. Docs; then decide whether to propose upstream (merge request to gitlab waders_LBC).

## Open questions

- Decided: merge request to the DAMIC-M gitlab, so changes stay additive (new modules, new options, defaults unchanged).
- g(n) incoherent combination still depends on image order (open item); fix before it becomes a process.
- Dependencies: numba (aGMX), scipy versions in the panaSKImg environment.
- The comm_scripts GitHub repo is private; this branch contains pysimdamicm code. Push only if that is fine.
