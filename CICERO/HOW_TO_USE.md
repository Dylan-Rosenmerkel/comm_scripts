# How to use CICERO

Every command and option of `run_CICERO`. For what the program is and how it decides, see [README.md](README.md); for
what every page, plot and number means, see [docs/CICERO_user_guide.md](docs/CICERO_user_guide.md).

## 1. Setup (once)

```bash
conda env create -f CICERO/environment.yml     # or: mamba env create -f ... / micromamba create -f ...
```

Then run `CICERO/run_CICERO` from anywhere. It finds the environment called `damic` by itself; you do not need to
activate it. Inside comm_scripts, `comm_scripts/bin/run_CICERO` does the same and can be put on your `PATH`.

## 2. The command

```bash
run_CICERO INPUT [options]
```

`INPUT` is one file, or a `.txt` list with one path per line (relative paths are relative to the list; `#` starts a
comment; any order, any mix of types):

| input | files |
|---|---|
| averaged images | `.fits`, `.fz`, `.fits.gz`, `.fits.bz2`; any size, binning or number of skips |
| skip-resolved raw FITS images | averaged over the skips first (`--skip-start K`) |
| level-0 traces, PSD traces | `trace_*.RJ45ADCDATA<ch>`, `psd_*.RJ45ADCDATA<ch>` (+ `.meta`) |
| level-1 images | `*_L1_*.RJ45ADCDATA0..3` + `.meta` |

Results go to `CICERO_<name>/` (name of the image or the list) in the current folder, or in `-o PARENT`. Read
`summary.txt` first; open `index.html` for the pages.

## 3. Recipes

```bash
# fits and clusters (clustering chosen per CCD)
run_CICERO images.txt
run_CICERO images.txt -o ~/results -j 4               # elsewhere, 4 images at a time
run_CICERO images.txt --overwrite                     # replace an earlier CICERO_images/

# fits only, no cluster files (much faster)
run_CICERO images.txt --no-clusters

# gain-corrected energies: measure g(n), write corrected cluster files too
run_CICERO images.txt --gain-correction               # one curve per module (pooled)
run_CICERO images.txt --gain-correction both          # pooled and per image

# Fe-55: the group's composite (K-alpha / K-beta fits front and back per CCD, CTI, plots)
run_CICERO images.txt --gain-correction --composite --module-list modules.csv
run_CICERO images.txt --composite --image-number 5    # file names without Image_<N>

# CCDs whose electron comb cannot be measured (e.g. ultra-low dark current): give the gain
run_CICERO l2.txt --gain 1.3
run_CICERO l2.txt --gain EXT1=1.3,EXT2=1.3,EXT3=1.3,EXT4=0.82
run_CICERO l2.txt --gain-file gains.json              # or an earlier run's native_summary.json

# one clusterer on every CCD instead of the automatic choice
run_CICERO images.txt --clusterer agmx
run_CICERO images.txt --clusterer threshold --thresholds 2.5,9.5

# continuous readout: join consecutive images row by row before clustering
run_CICERO images.txt --stitch time

# raw DAQ data
run_CICERO traces.txt                                 # L0 traces / PSD traces -> packet/ pages and L0/
run_CICERO l1.txt --l1-skips 1,10,100,1000            # L1 images -> one FITS per number of skips -> processed

# commissioning tools
run_CICERO packet.txt --packet --module-list modules.csv
run_CICERO series.txt --dc-series --dc-limit 0.01 --no-clusters
run_CICERO images.txt --defect-map --no-clusters
run_CICERO sr.txt --sr --no-clusters

# campaigns: data arriving packet by packet into one folder
run_CICERO packet_DM01.txt --packet --campaign ~/comm/LSM2026 --module-list modules.csv
run_CICERO packet_DM02.txt --packet --campaign ~/comm/LSM2026        # adds DM02 only
run_CICERO series_day3.txt --dc-series --campaign ~/comm/LSM2026     # new images only
run_CICERO retest.txt --packet --dataset retest --campaign ~/comm/LSM2026
run_CICERO any.txt --campaign ~/comm/LSM2026 --criteria limits.json  # new limits: everything re-judged, no new fits
run_CICERO --campaign ~/comm/LSM2026                                 # open a campaign (and save reviews) elsewhere
```

Tools can be combined in one run (e.g. `--dc-series --defect-map --sr`), except `--packet`, which makes only the
packet reports. Everything works with or without `--campaign`.

## 4. All options

### General

| option | default | what it does |
|---|---|---|
| `-o DIR`, `--parent DIR` | current folder | where `CICERO_<name>/` is made |
| `-j N`, `--jobs N` | 2 | images processed at the same time (one process each) |
| `--overwrite` | off | replace an existing `CICERO_<name>/` (not with `--campaign`) |
| `--env NAME` | `$CICERO_ENV`, else `damic` | conda environment to run in |
| `-h`, `-h -v` | | the flags; the full guide |

### Clustering and calibration

| option | default | what it does |
|---|---|---|
| `--clusterer auto\|agmx\|threshold` | `auto` | `auto`: one chain per CCD (A_GMX, Bayesian few-electron finder, crowded mode, 'unmodelled', threshold); `agmx` / `threshold`: that clusterer everywhere |
| `--thresholds auto\|sigma0\|PIX,SEED` | `auto` | threshold clusterer: from the dark charge (auto), panaSKImg's 10 / 30 sigma0, or pixel and seed thresholds in electrons |
| `--no-clusters` | off | fits only, no cluster files |
| `--gain G` or `EXT1=G1,...` | | gain [ADU/e-] for CCDs without a measurable comb (flagged) |
| `--gain-file JSON` | | gains: `{EXTn: G}`, `{acm: {EXTn: G}}` or a `native_summary.json` |
| `--skip-start K` | 0 | skip-resolved raw files: first skip used in the average |
| `--tune` | off | per CCD: pedestal method and fit window chosen by the smallest fit error |
| `--autoconfig-args "ARGS"` | | extra geometry / measurement settings, quoted, e.g. `"--ccd-cols 6200"` |
| `--stitch time\|order` | off | join consecutive images of one setting row by row (by DATEINI / DATEEND, or list order) |
| `--stitch-gap S` | 5 | `--stitch time`: largest gap [s] between images |

### Gain vs charge, g(n)

| option | default | what it does |
|---|---|---|
| `--gain-correction [pooled\|per-image\|both]` | off (`pooled` if given alone) | measure g(n) and write corrected cluster files (`root_gaincorr/`, `root_gaincorr_per_image/`) |
| `--gain-model smooth\|interpolated\|broken\|linear` | `smooth` | form of g(n) |
| `--gain-combine incoherent\|coherent` | `incoherent` | several images of a module: comb powers added, or pixels pooled |
| `--gain-memory off\|auto` | `off` | remove the horizontal readout memory before measuring g(n) |
| `--gain-periods auto\|off` | `auto` | a curve per gain period where a module's gain steps by > 1 % |
| `--gain-anchor comb\|fit` | `comb` | low-charge gain from the comb, or each image's fitted gain |
| `--temperature K` | `$COMM_TEMPERATURE_K`, else 130 | CCD temperature for the energy per electron, epsilon(T) |

### Fe-55 composite and module names

| option | default | what it does |
|---|---|---|
| `--composite` | off | run the Fe-55 composite macro per image type and module (needs cluster files) |
| `--composite-macro PATH` | `$MAKECOMPOSITE_MACRO`, else `commissioning/lsm_composite/...` | another version of the macro |
| `--image-number N` | `Image_<N>` from the name | image type for the macro |
| `--module ACM=NAME` | | module name of an ACM (repeat for several) |
| `--module-list FILE` | | module names per ACM or CCD (format in section 6); kept in a campaign |

Module names come from `--module-list`, `--module`, a `..._DMa_DMb_DMc_...` data folder name, else `ACM<id>`.

### Raw data and commissioning tools

| option | default | what it does |
|---|---|---|
| `--l1-skips auto\|N,N,...` | `auto` (1, 2, 5, 10, ... and all) | level-1 images: one FITS image per number of skips |
| `--l1-skip-first K` | 0 | level-1 images: leave out the first K skips of every pixel |
| `--l0-no-html` | off | level-0 traces: no interactive viewer in `L0/` |
| `--packet` | off | the list is a commissioning packet: one page per module (`packet/`), no clustering |
| `--dc-series` | off | dark current over the series (`dc_series/`) |
| `--dc-limit E` | | with `--dc-series`: flag CCDs above E e-/pix/day |
| `--defect-map` | off | median composites and defects per module and image size (`defects/`) |
| `--defect-sigma N` | 5 | hot / cold pixels at N × the local spread |
| `--sr` | off | serial-register images: streaks along a read (`sr/`) |
| `--sr-threshold N` | 4 | with `--sr`: N × noise |
| `--sr-min-length PIX` | 3 | with `--sr`: shortest streak |

### Campaigns and reviews

| option | default | what it does |
|---|---|---|
| `--campaign DIR` | off | add the inputs to campaign folder DIR (made on first use); only new or changed inputs are processed |
| `--criteria FILE` | `commissioning/criteria_default.json` | pass / warn / fail limits; replacing them re-judges everything without new fits |
| `--reprocess` | off | process the given inputs again |
| `--refresh-all` | off | rebuild every tool's pages, not only this run's |
| `--dataset NAME` | default data set | a module measured more than once (e.g. after a repair): own report and view per data set |
| `--reviews FILE ...` | | merge review files downloaded from the pages into `DIR/reviews.js` |
| `--no-review-helper` | off | do not start the local helper through which pages save reviews |
| `--port PORT` | 8765 | first port the review helper tries |

Reviews can also be added from the command line:

```bash
python CICERO/commissioning/reviews.py DIR --add --reviewer NAME --status concern --module DM03 --ext EXT2 --comment "..."
```

## 5. Environment variables

| variable | what it sets |
|---|---|
| `CICERO_ENV` | the conda environment (as `--env`) |
| `MAKECOMPOSITE_MACRO` | the composite macro (as `--composite-macro`) |
| `COMM_TEMPERATURE_K` | CCD temperature for epsilon(T) (as `--temperature`) |

A_GMX also reads expert switches (`AGMX_CROWDED`, `AGMX_CROWDED_SHAPES`, `AGMX_LINK_STD`, `AGMX_XSAT`, `AGMX_XSAT_STD`,
`AGMX_PHYS_FIXES`, `AGMX_PIX_EXACT`, `AGMX_NUMBA`). Their defaults are the validated settings; change them only to
compare with earlier behaviour.

## 6. Files

**Module list** (`--module-list`): one line per ACM, or per CCD; comma, tab or space separated:

```
# ACM, module[, EXT<n> or CCD name]
101, DM24
103, PA03, EXT2
```

**Output folder** (`CICERO_<name>/` or the campaign folder):

| file | content |
|---|---|
| `summary.txt` | read first: fits per CCD, the clustering chain of every CCD and why, verdicts, every run |
| `index.html` | overview of every tool per module and CCD |
| `images/`, `packet/`, `dc_series/`, `defects/`, `sr/` | each tool's overview (`index.html`) and module pages |
| `root/` | cluster files, one per CCD (`panaSKImg_clustersRec_<image>_ext<n>.root`); `root_files.txt` lists them |
| `root_gaincorr*/`, `gain_vs_charge/` | with `--gain-correction` |
| `composite*/` | with `--composite` (`Fe_55_Analysis_Plots/<module>/...`, `energy_summary.csv`) |
| `fit_quality.csv` | gain, noise, dark current per CCD, errors scaled by sqrt(chi2/ndf) |
| `native_summary.json` | everything per image and CCD |
| `characterisation/<image>.json` | the front end's record per image and CCD |
| `*_DCfit_EXTn.pdf`, `*_Baseline_EXTn.pdf`, `*_Sigma_EXTn.pdf` | fit and pedestal plots |
| `L0/`, `L1/`, `compressed/` | raw-data products |
| `reviews.js`, `reviews.csv`, `reviews.txt` | the reviews written on the pages |
| `campaign.json`, `criteria.json`, `records/`, `modules.csv` | the campaign's ledger, limits, kept results, module list |
| `logs/` | failures and composite output |

The pages work offline. To share results, copy or zip the whole folder.

## 7. Tests

```bash
cd CICERO && python -m unittest discover -s tests
```
