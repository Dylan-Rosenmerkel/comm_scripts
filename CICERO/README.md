# CICERO

**CCD Image Calibration, Event Recognition, and Optimization**: adaptive software for skipper-CCD image calibration
and energy-event analysis.

From averaged images or raw DAQ files, `run_CICERO` produces:
- calibrated fits per CCD (pedestal, gain, read noise, dark current);
- clusters in panaSKImg's ROOT format, with the clustering chosen per CCD from what the image is;
- the gain-vs-charge correction g(n), Fe-55 composites, and the commissioning reports (packets, dark-current series,
  defect maps, serial-register images, L0 / L1 data).

It is independent of panaSKImg and of the rest of comm_scripts: everything it runs is in this folder.

## Install

1. Copy this folder anywhere.
2. Make the Python environment once (conda, mamba or micromamba; conda-forge; Linux or macOS):
   ```bash
   conda env create -f CICERO/environment.yml
   ```
3. Run `CICERO/run_CICERO` (or `comm_scripts/bin/run_CICERO`). It finds the environment named `damic` by itself;
   `--env NAME` or `$CICERO_ENV` chooses another one.

## Run

```bash
run_CICERO INPUT [-o PARENT] [-j N] [options]   # INPUT: one file, or a .txt list (one path per line)
run_CICERO -h                                   # every flag;   -h -v: the full guide
```

Results go to `PARENT/CICERO_<input name>/`. Read `summary.txt` first; `index.html` is the overview page.

**Every command and option:** [HOW_TO_USE.md](HOW_TO_USE.md).

**Inputs** (any mix, in any order): averaged images (`.fits`, `.fz`, `.fits.gz`, `.fits.bz2`; any size, binning or
number of skips), skip-resolved raw FITS files (averaged first), level-0 traces and PSD traces (`trace_*` / `psd_*`
`.RJ45ADCDATA<ch>`), and level-1 images (`*_L1_*.RJ45ADCDATA0..3` with their `.meta`).

| task | command | output |
|---|---|---|
| fits and clusters | `run_CICERO images.txt` | `root/`, `images/`, `summary.txt`, `fit_quality.csv` |
| gain-corrected energies (g(n)) | `... --gain-correction [pooled\|per-image\|both]` | `gain_vs_charge/`, `root_gaincorr*/` |
| Fe-55 analysis | `... --composite --module-list modules.csv` | `composite/`: K-alpha / K-beta fits front and back per CCD, CTI, plots, `energy_summary.csv` |
| fits only | `... --no-clusters` | no cluster files (much faster) |
| CCDs without an electron comb | `... --gain 1.3`, `--gain EXT1=1.3,EXT4=0.82` or `--gain-file gains.json` | flagged in `summary.txt` |
| L0 traces / PSD traces | `run_CICERO traces.txt` | `packet/` (trace view and noise spectrum per module), `L0/` |
| L1 images | `run_CICERO l1.txt --l1-skips 1,10,100,1000` | `L1/`: one FITS per number of skips, noise vs N; then processed as images |
| commissioning packet | `run_CICERO packet.txt --packet --module-list modules.csv` | `packet/` |
| dark current over a series | `run_CICERO series.txt --dc-series [--dc-limit X]` | `dc_series/` |
| defect maps | `run_CICERO images.txt --defect-map` | `defects/` |
| serial-register images | `run_CICERO sr.txt --sr` | `sr/` |
| continuous readout | `... --stitch time` | consecutive images joined row by row before clustering |

Every task runs on its own (a fresh folder per run) or inside a campaign (below); the tools can be combined in one
run, except `--packet`, which makes only the packet reports. `--composite` needs `Image_<N>` in the file names (or
`--image-number N`). There is no dedicated Am-241 fit: the composite plots the 10-30 keV spectrum, and the cluster
files carry the energies for your own fits.

## Clustering, chosen per CCD

Each image and CCD is characterised before fitting (`engine/characterise.py`: geometry, dead channels, gain source,
noise, dark charge, occupancy). The chain is then chosen per CCD (`--clusterer auto`, the default):

| CCD | chain |
|---|---|
| electron counting, sparse | A_GMX, with the Bayesian few-electron finder in sparse, low-dark-charge regions |
| electron counting, crowded (core fill >= 0.10) | A_GMX crowded mode |
| read noise >= 1 e-, or a structured background | A_GMX 'unmodelled' (hysteresis) |
| flooded or peaks not resolved | threshold clusterer |

`--clusterer agmx` or `--clusterer threshold` uses one clusterer everywhere. Every choice and fallback is written to
`summary.txt`, `characterisation/<image>.json` and each ROOT file. Besides panaSKImg's branches, clusters carry
`finder` (0 threshold, 1 A_GMX, 2 Bayes), the A_GMX branches (`agmx_*`) and, on binned images, `agmx_sxy_px` (the
width in unbinned pixels: use it for front / back cuts).

## Campaigns: data arriving packet by packet

```bash
run_CICERO packet_DM01.txt --packet --campaign ~/comm/LSM2026 --module-list modules.csv
run_CICERO series_day3.txt --dc-series --campaign ~/comm/LSM2026
```

With `--campaign DIR`, each run processes only inputs that are new or changed, and rebuilds the pages from
everything the folder holds. Useful flags: `--dataset NAME` (a module measured again, e.g. after a repair),
`--criteria FILE` (new pass / warn / fail limits; re-judges everything without new fits), `--reprocess`,
`--refresh-all`.

- **Overview pages:** `index.html` (every tool per CCD) and one per tool, each with a module × CCD grid of verdicts
  and values, distributions, a sortable table with CSV, and the criteria.
- **Verdicts:** come from `criteria.json` (default `commissioning/criteria_default.json`). The default limits are
  placeholders until the group sets them.
- **Reviews:** a name, a status and a comment, typed on any page, are saved into `DIR/reviews.js`. A small local
  helper started by each run does the saving (`--no-review-helper` to skip it); `run_CICERO --campaign DIR` with no
  input starts it on another computer.
- **Sharing:** copy or zip the whole folder; the pages link to each other.

## Status

Checked on real data: the engine against panaSKImg, the clustering on Fe-55, muon and Am-241 images, g(n), L0
traces, L1 files and the packet and dark-current tools. Few-electron, crowded, heavily binned and skip-resolved data
run, but parts of those chains are checked only on synthetic or injected data. Details and open items:
[docs/STATUS.md](docs/STATUS.md).

## Layout

```
CICERO/
  run_CICERO       the command
  engine/          per image: pedestal, fits, calibration, clustering chain, ROOT files; characterisation; g(n)
  clustering/      A_GMX (incl. crowded mode) and the Bayesian few-electron finder
  commissioning/   campaigns, overview pages and reviews; packet, dark-current, defect-map, SR and L0 / L1 tools;
                   copies kept in sync with their sources: raw_levels/ (ccd_opt_ml DAQ decoders),
                   lsm_defects/ and lsm_composite/ (LSM_Module_Testing_Analysis)
  HOW_TO_USE.md    every command and option
  docs/            STATUS.md, the user guide (every page and number explained), plans and notes
  tests/           python -m unittest discover -s CICERO/tests
  environment.yml  the conda environment
```

The copies in `commissioning/` are updated with `utils/sync_from_ccd_opt_ml.py` and
`utils/sync_from_lsm_analysis.py` (in comm_scripts).
