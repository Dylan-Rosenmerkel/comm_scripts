# run_superWADERs: user guide (2026-11-12)

`run_superWADERs` is a separate program (`superWADERs/`; overview, clustering and layout in
[superWADERs/README.md](../superWADERs/README.md)): its own engine (no panaSKImg), clustering chosen per CCD, and the
commissioning tools below. It uses nothing from `run_WADERs` or `src/`. Python: the `damic` conda environment (found
automatically).

```
run_superWADERs INPUT [-o PARENT] [-j N] [options]       # INPUT: one file, or a .txt list (one path per line)
run_superWADERs -h            # all flags          run_superWADERs -h -v     # the full guide
```
Results: `PARENT/superWADERs_<input name>/` (`--overwrite` to replace an existing one), `summary.txt` first.

## 1. What it takes as input

| file | what it is | how it is used |
|---|---|---|
| `avg_*.fits`, `.fits.bz2`, `.fz` | averaged images from the DAQ | the image pipeline (pedestal, fits, clusters) |
| `trace_*.RJ45ADCDATA<ch>` (+ `.meta`) | L0 trace: raw video samples of one channel | decoded, checked, plotted (`L0/`) |
| `psd_*.RJ45ADCDATA<ch>` (+ `.meta`) | L0 noise trace (no skip windows) | samples and spectra (`L0/`, packet PSD tab) |
| `*_RxCxS_BxB_L1_*.RJ45ADCDATA0..3` + `.meta` | L1 image: every skip of every pixel (binned or not) | FITS images with N skips -> pipeline (`L1/`) |

In a list, files may come in any order; the four channel files of an L1 image are grouped by name. A trace's channel
file without samples (channels not recorded) is listed as "not recorded" and skipped. DAQ binaries of averaged (L2)
images are not read: give their `avg_*.fits`.

**Module names** (`--module-list FILE`): a correlation list, one line per ACM, or per CCD where a board carries CCDs
of different modules. Comma, tab or space separated; `#` comments and a header line are ignored:
```
# ACM, module[, EXT<n> or CCD name]
101, DM24
102, DM25
103, PA03, EXT2
```
Without an entry: `--module ACM=NAME`, then the data folder name (`..._DMa_DMb_DMc_...`), else `ACM<id>`.

## 2. The commands

### Fits and clusters
```
run_superWADERs images.txt --gain-correction --composite --module-list modules.csv
```
Clustering: `--clusterer auto` (default; one chain per CCD: A_GMX + Bayesian few-electron switch, A_GMX crowded mode,
A_GMX 'unmodelled', threshold clusterer; see superWADERs/README.md), `agmx` or `threshold` (one clusterer everywhere).
`--thresholds auto|sigma0|PIX,SEED` (threshold clusterer), `--gain` / `--gain-file` (CCDs without a comb), `--tune`,
`--no-clusters`, `--gain-*`, `--composite`: `-h`. Every CCD is processed (flooded CCDs go to the threshold clusterer).
Per CCD, `summary.txt` lists the chain and why; `characterisation/<image>.json` has the full record.

### L1 images: images with N skips (`--l1-skips`)
```
run_superWADERs l1.txt --l1-skips 1,10,100,1000 [--l1-skip-first 20] [--no-clusters]
```
`l1.txt` lists the four `.RJ45ADCDATA0..3` files (the `.meta` must be next to them). Output `L1/`:
- `avg_img_<rows>x<cols>x<N>_..._L1_....fits`: one image per N (mean of skips K..K+N-1), DAQ header from the .meta;
  all go through the pipeline (every CCD processed, also unresolved ones);
- `noise_vs_skips.csv/.png`: noise and gain per CCD against N, with the 1/sqrt(N) line;
- `compare/<image>/`: the DC-fit, baseline and sigma plots of every N side by side (PDF);
- `<image>_diagnostics/ch<k>.png` + `diagnostics.json`: start-of-pixel transient per skip (and the suggested
  `--l1-skip-first`), single-skip noise sigma_1, noise vs N within a pixel (Allan deviation), spectrum of the skip
  sequence, drift of a pixel's level during its skips, level along rows and columns.

### L0 traces
```
run_superWADERs traces.txt [--l0-no-html]
```
`L0/<trace>/overview.png`, `viewer.html` (interactive, whole trace), `summary.json`; the summary lists for each trace:
samples, pixels x skips, skip cycle, inline vs raw sums (agree = decoder and firmware consistent), quality.

### Commissioning packet (`--packet`)
```
run_superWADERs packet.txt --packet --module-list modules.csv
```
`packet.txt`: the packet's files in any order (traces, PSD traces, images, L1 files). Output `packet/`:
`index.html` (one line per module: ACM, extensions, CCD names, flags; filter box, click a column title to sort) and
one report per module `<module>_ACM<id>.html` with check boxes for its CCDs (see section 3). No clustering. Tabs
(only those the packet has data for):
- **Overview**: identity, files, the main numbers of every image.
- **Trace** (the ticked CCDs one below the other): the whole trace (zoom down to single skip cycles; pedestal / signal windows shaded red / green),
  per-skip window means, CDS per skip, the folded skip cycle (mean +- 1 sigma), noise spectrum, decoder checks.
- **PSD**: ADC trace, periodogram, Welch (linear and log) with sigma_ADU and sigma_CDS (CDS of windows of the
  delayInt length); the ticked CCDs overlaid, one colour per CCD.
- **Image summary**: per image gain, noise, dark current (fit and 0e-/1e- counts), chi2, pedestal method, gain
  choice (the CCDs side by side); the pixel-charge histogram with the fit (one per CCD); pedestal and sigma per row,
  dark charge per row, column and row medians (CCDs overlaid). Numbers outside the usual range are red, and the tab gets a red dot.
- **One viewer per image** (ds9-like, a slider switches between the CCDs as ds9 frames): see section 3.

### Dark current over a series (`--dc-series`)
```
run_superWADERs series.txt --dc-series [--dc-limit 0.01] [--no-clusters] --module-list modules.csv
```
The usual run, plus `dc_series/`: `dc_report.pdf` (summary table; one page per module: rates of its CCDs vs time;
one page per CCD: dark charge per image (fit and 0e-/1e- counts), rate vs time with the settling fit, dark charge per
row vs the row's waiting time), `dc_series.csv`, `index.html` + one interactive page per module `<module>_ACM<id>_dc.html` (tabs Rates, Fits, Per
row, Data, Method; the ticked CCDs overlaid in the plots, side by side in the tables; Fits: every image's
pixel-charge histogram with the dark-current fit, per CCD).
Rates in e-/pix/day: DC / (exposure + readout/2); the row fit gives the rate from the growth along the rows
(slope) and the charge that does not grow with time (intercept: made during the readout). Flags: above
`--dc-limit`, still falling at the end. A settling fit (plateau + exponential) is shown with 5 or more images.

### Defect maps (`--defect-map`)
```
run_superWADERs images.txt --defect-map [--defect-sigma 5] [--no-clusters] --module-list modules.csv
```
Per module and image size: all images of one ACM with the same rows x columns and binning form one group, whatever their type or number of skips; `defects/<module>_ACM<id>_<rows>x<cols>_<binning>/`:
- the group's `Image_4` (D. Venegas-Vargas, copy in src/lsm_defects/): composite median of all images (PNG + log
  with column defects and the event rate per pixel per day), with its geometry set from the images' header; shown
  in the page's "Image_4 plot" tab;
- `defects.csv`, the defect log: type (bright / dark column, hot / cold pixel spot), column, row, size, value,
  number of images it is seen in; adjacent flagged columns are one entry (their column range). A column flagged by
  Image_4 on the composite is kept only if it is also bright / dark in enough of the single images (a track along
  a column in one image is not a defect; Image_4's own plot and log are not filtered). The first active column is
  usually flagged in every image: a known readout artefact, not a CCD defect;
- hot / cold pixels must be persistent: present in enough of the images that chance coincidences of events or
  dark electrons are unlikely (< 0.1 per CCD, from each image's measured occupancy, at charge levels and bands
  1-16 x the threshold). Few or busy images can therefore give none ("none identifiable", logged on the page with
  the rule used). Checked on simulated images (planted hot / cold pixels, 1e-5 to 1 e-/pix, 0-30 % event pixels,
  2-10 images: no false hot pixels; at 1e-5..1e-3 e-/pix with 4 images 50-75 % of +1 e- and all +4 e- hot pixels
  found) and on real data (DM26 Image 5, L2 ACM 101 and the very-low-DC L2 ACM 103 noise test: no hot or cold pixels);
- images without an overscan: Image_4 still runs, with pedestal and noise from the active region (CTI n/a);
- `composite_EXT<n>.fits`;
- `defect_map.html`: the composite with the defects marked (tick boxes per type), one CCD at a time with the frame
  slider, the defect list of that CCD (click a line to jump to it), and a tab with the scripts' figures and logs. `defects/index.html` lists the modules.

### Serial-register images (`--sr`)
```
run_superWADERs sr_images.txt --sr [--sr-threshold 4] [--sr-min-length 3] [--no-clusters]
```
Streaks = serial-register events (readout artifacts): runs of >= `--sr-min-length` pixels above `--sr-threshold` x
noise along one read (row), confined to it. `sr/`: `sr_streaks.csv` (one line per streak), `sr_summary.csv` (per
image and CCD: streaks, per 1000 reads, median length, DC), per CCD a PNG (counts per image, lengths, positions
along the register, dark charge per read) and one interactive page per module `<module>_ACM<id>_sr.html`:
**Diagnostics** (per image: streaks, streaks per 1000 reads, dark charge from the fit and the 0e-/1e- counts, read
noise, gain; dark charge per read; streak lengths and start positions; a table of all numbers; the CCDs overlaid)
**DC fits** (every image's pixel-charge histogram with its dark-current fit, the numbers side by side, the profiles)
and one tab per image with the streaks marked (frame slider over the CCDs). The streak list is only in
`sr_streaks.csv`, not on the page.

## 2b. Campaigns and overview pages

Commissioning data arrive packet by packet. Give every packet (or list) the same `--campaign DIR`:
- **Only new inputs are processed** (new path, or the file changed). Their results are kept in `DIR/records/`.
- **The tools of the run rebuild their pages** from everything the campaign holds; tools not touched keep theirs.
- **`DIR/index.html` combines every tool per CCD.**
- **Without `--campaign`, a run stands alone:** a fresh folder with only its inputs and tools.

Details, the verdicts and the criteria file are in [superWADERs/README.md](../superWADERs/README.md), section
"Commissioning campaigns".

```
run_superWADERs packet_DM01.txt --packet --campaign ~/comm/LSM2026 --module-list modules.csv
run_superWADERs packet_DM02.txt --packet --campaign ~/comm/LSM2026 --module-list modules.csv
run_superWADERs series.txt --dc-series --campaign ~/comm/LSM2026
run_superWADERs packet_DM01_retest.txt --packet --dataset retest --campaign ~/comm/LSM2026   # DM01 measured again
```

**Several data sets per module** (`--dataset NAME`: a second packet, a second cooldown's dark-current run, ...): each
data set gets its own report and its own view in the tool's overview (newest first); the campaign overview shows the
newest one per CCD and links to all of them.

Every tool (images, packet, dark current, defect maps, SR) has an overview page, `<tool>/index.html`:
- a module × CCD grid with pass / warn / fail verdicts from the criteria file (`criteria.json`; the default limits
  are placeholders);
- what needs attention, distributions against the limits, the full table (CSV), the criteria;
- per CCD (click a cell): the reasons, the reviews, the links to the full reports and the main numbers; every other
  number and the history over images folded below.

Module pages link back to their overview. The campaign page's cells show read noise and gain (images, else packet),
dark current (latest image), the dark-current rate of the series (plateau, else latest) and the serial-register
dark-current rate, plus each CCD's review status.

**Reviews.** Every overview (per CCD) and every module report (per tab, for the whole module or one CCD) has a review
box: your name, a status (approved / concern / rejected / note) and a comment. Open the pages with
`python3 DIR/review_server.py` (written into every campaign folder; or `run_superWADERs --campaign DIR --serve`) and
every review is saved into `DIR/reviews/` the moment it is added, in any browser. On double-clicked pages: Chrome /
Edge **Save to campaign folder** (asks for the folder once, then saves as you go); other browsers **Download**, then
`run_superWADERs --campaign DIR --reviews FILE` (no input). Saved reviews show on every page of the campaign and are listed in
`DIR/reviews.csv` and `DIR/reviews.txt`. A review keeps the data and numbers it was written on; when newer data
arrive it is marked "data changed since". Details: [superWADERs/README.md](../superWADERs/README.md), "Reviews".

## 3. Using the HTML pages

Every page works offline (no network) in Chrome, Safari or Firefox; open it by double-click.
- **First visit**: a blue hint line; **? Help** (top right) explains everything; Esc closes it.
- **CCDs** (the row below the tabs, on every module page): tick the CCDs to show; **all CCDs in the module** ticks
  them all; **only** next to a CCD shows that one alone. Every tab follows: traces one below the other; PSD,
  profiles and series plots overlaid (one colour per CCD; with two or more series per CCD the line style is
  explained above the plot); numbers side by side (a value the same for every CCD is shown once, grey); fit
  histograms and figures one per CCD; images one CCD at a time. The choice is remembered in the browser. A red dot
  next to a CCD: it has numbers outside the usual range.
- **Plots**: drag a box to zoom; shift + drag to move; mouse wheel zooms x (alt + wheel: y); double-click or
  **reset** shows everything again; **log y** / **lin y** switches the y axis; **PNG** saves the plot. Hover to
  read values.
- **Images**: wheel or **+** / **-** zoom, drag to move, **fit** / **1:1**; click a pixel: its row and column are
  plotted below (the middle row / column at start); shift + drag a box: mean, median, std, min, max of the region;
  **scale** (99.5 % hides the brightest pixels; manual: type min / max), **colormap**, **asinh** (faint
  structure), colour bar below; row 0 at the bottom as in ds9; **PNG** saves the view. **Frames**: the slider,
  the arrow buttons or the left / right arrow keys switch between the ticked CCDs, keeping zoom, position, scale
  mode and the cut (to compare the same spot); **blink** switches every second. Defect pages: tick boxes per
  defect type, "show markings", click a defect in the list to jump to it. Long lists are laid out in columns of 40
  lines, with a type filter and a sort (largest value / position).
- **Reviews**: one box per tab (name, status, whole module or one CCD, comment); the line under the title lists every
  review of the module. Saving: see section 2b.
- **Export PDF**: every tab on its own page with the reviews; choose "Save as PDF" in the print dialog.
- **Log**: at the bottom of each tab: files, header settings, decoder checks, fit results, warnings.
- A **red dot** on a tab / red numbers: outside the usual range (noise > 0.3 e-, peaks not resolved, defects
  found, DC above the limit, ...).

## 4. What has been checked, and on what

| feature | checked on | result |
|---|---|---|
| native engine vs panaSKImg | DM09 Image 5; board-103 skip-noise images (80 CCD-images); L2 surface run | same gain / noise / DC (median diff 0.00 %), same clusters with --thresholds sigma0 |
| pedestal methods, --tune | L2 surface run (charged overscan), 100-skip images | per row within 1e-4 ADU of panaSKImg; same --tune choices |
| L1 -> FITS (unbinned) | synthetic L1 file from a DM09 image (decode_l1.write_l1) | exact round trip, noise 1/sqrt(N) |
| L1 binned | real 20x20 file of 2026-10-06 | 80/80 rows; levels match a DAQ 20x20 image; ccd_opt_ml tests pass |
| L1 diagnostics | the same two files | sigma_1 = 4.00 ADU injected / found; real file: transient +220 ADU in skip 0, drift 1.2 ADU/row |
| L0 traces / PSD | 2026-09-16 PA03/DR03 traces and PSD traces | decoder checks agree; plots and pages render |
| packet report | 2026-09-16 PA03/DR03 files (68, shuffled; 20x20 images) | 8 reports, 17 s, no script errors |
| DC series | DM24-26 module 106, 6 Image 5 over 19 h | rates 3-9 e-/pix/day, row fit consistent; PDF, pages |
| defect map | L2 surface run (with overscan), DM26 Image 5 (no overscan) | group scripts run (L2); thresholds not yet validated |
| SR images | (no SR images yet) | runs; thresholds untested |

## 5. Data needed to confirm (please provide)

1. **A real commissioning packet** (per module): the trace, the PSD trace, the 5 min 1x1 1-skip image, the 5 min
   10x10 500-skip image (with their .meta), as it will be taken - ideally for 2-3 modules, and the correlation list
   file as it exists (to match its format).
2. **A complete unbinned L1 image** (all four channel files + .meta, not cut short), and, if possible, an averaged
   image taken right before / after with the same settings: to confirm the L1 -> FITS orientation, sign and levels
   against the DAQ's own averaging, and the transient / drift diagnostics.
3. **A binned L1 image with more skips** (the 2026-10-06 ones have 10 skips): to check the binned path with resolved
   peaks.
4. **A dark-current series**: the images of one cool-down or a long run (with DATEINI/DATEEND), plus the dark-current
   limit(s) the group uses, and (if logged) the cool-down time.
5. **Defect-map images**: any images of the same size (several per module; e.g. the Image 2 / Image 4 runs), with a module whose defects are known (a list or a previous Image_4 log) to tune
   `--defect-sigma` and compare the defect log.
6. **Serial-register images**: a few SR images (with .meta), and an example of the plots / numbers the group makes
   for them now, to tune the streak finder (threshold, minimum length) and the DC-per-read plots.
7. Optional: an Am-241 or Fe-55 run taken with run_WADERs before, to keep the native engine's energy scale checked.

## 6. Where things are

| what | file |
|---|---|
| command | `superWADERs/run_superWADERs` (`bin/run_superWADERs` starts it) |
| engine | `superWADERs/engine/`: `native_pipeline.py`, `characterise.py`, `native_kernels.py`, `native_rootio.py`, `native_plots.py`, and its own autoconfig / gain modules |
| clustering | `superWADERs/clustering/` (A_GMX, Bayesian few-electron finder) |
| L0 / L1 | `superWADERs/commissioning/raw_data.py`; readers copied from ccd_opt_ml in `raw_levels/` (`utils/sync_from_ccd_opt_ml.py [--check]`) |
| packet report and the module page template | `superWADERs/commissioning/packet_report.py`, `packet_report_template.html` |
| campaign, verdicts, overview pages | `superWADERs/commissioning/campaign.py`, `overview_template.html`, `criteria_default.json`, `worker.py`, `ccd_metrics.py` |
| DC series | `superWADERs/commissioning/dc_series.py` |
| defects | `superWADERs/commissioning/defects.py`; group scripts copied in `lsm_defects/` (`utils/sync_from_lsm_analysis.py [--check]`) |
| SR images | `superWADERs/commissioning/sr_images.py` |
| docs | this guide; `superWADERs/README.md`; `docs/native_pipeline.md`, `docs/native_pipeline_validation.md` (engine validation), `docs/superWADERs_commissioning_next_steps.md` |
