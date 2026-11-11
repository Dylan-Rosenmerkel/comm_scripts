# superWADERs: commissioning additions, plan of attack (2026-11-11)

From "SuperWADERs Additions" (Document1.pdf). Status of run_superWADERs today: native engine (no panaSKImg),
panaSKImg-style plots, all pedestal methods, --tune, L0 traces (decode / check / plot / html viewer), unbinned L1
images -> FITS per number of skips -> pipeline -> noise vs N (docs/native_pipeline.md). Everything below builds on
`src/native_pipeline.py` (fits per CCD), `src/raw_data.py` (L0 / L1) and `src/raw_levels/` (ccd_opt_ml readers).

## Status (2026-11-11)

- **1. Binned L1: done.** `raw_data.l1_skip_means` finds each row after its run of LINE markers; works for binned and
  unbinned files. 20x20 file of 2026-10-06: 80 of 80 rows; at N = 1 its prescan / active / overscan levels (316 / 183
  / 149 ADU on EXT1) match a DAQ avg 20x20 1-skip image of the same board (349 / 184 / 143, other time); unbinned
  round trip unchanged (2e-4 ADU). Not yet ported to ccd_opt_ml's decode_l1.
- **3. Packet report: first version done** (`run_superWADERs PACKET.txt --packet [--module-list FILE]`,
  src/packet_report.py + src/packet_report_template.html). Inputs in any order, classified by file type (trace_,
  psd_, avg_*.fits, level-1 channel files); module names from the correlation list (lines `ACM, module[, EXT<n> or
  CCD name]`); one self-contained HTML per CCD (`packet/<module>_ACM<id>_EXT<n>_<CCD>.html`) + `packet/index.html`.
  Tabs (only those with data): Overview, Trace (whole trace zoomable with window shading, per-skip means, CDS,
  folded cycle, spectrum), PSD (ADC trace, periodogram, Welch linear / log with sigma_ADU and sigma_CDS; several PSD
  files overlaid), Image summary (native fits, DC-fit histogram, pedestal and sigma per row, dark charge per row,
  column / row medians), one ds9-like viewer per image (pan / zoom, scale, colormaps, asinh, pixel readout, region
  statistics, row / column cuts). Comments per tab (browser storage + JSON save / load), a log per tab, Export PDF
  (print of all tabs with the comments). CCDs without CCD signal: image and raw statistics still shown. Tested on
  the 2026-09-16 PA03/DR03 files (68 files shuffled: 8 reports of ~8.5 MB, 17 s; rendered in headless Chrome, no
  script errors). To do: more image examples (5 min 1x1 1-skip, 10x10 500-skip), report size for full-size images
  (viewer bins 2x2 above 4 M pixels), PSD comparisons across dates.
- **6. SR images**: serial-register events = streaks in the serial register, readout artifacts (horizontal streaks
  along a row) - the finder will look for those, not for particle clusters.

## Order of work

| # | item | why this order | size |
|---|---|---|---|
| 1 | Binned L1 decoding | small, unblocks L1 of the 10x10 / 20x20 packet images | 0.5 day |
| 2 | L1 diagnostics | needs 1; reused by the packet report | 1 day |
| 3 | Commissioning packet report (`--packet`) | the main deliverable; needs 1-2 | 3-4 days |
| 4 | DC commissioning (`--dc-series`) | reuses the per-CCD fits and the report machinery of 3 | 1-2 days |
| 5 | Defect map (`--defect-map`) | independent; port of two existing scripts + viewer of 3 | 1-2 days |
| 6 | Serial-register images (`--sr`) | needs example SR images to define "events" | 1-2 days |

Common pieces built once in step 3 and reused: the HTML report shell (tabs, comment boxes, logs, PDF export), the
image viewer, the per-CCD identity (ACM, extension, CCD name, module name).

## 1. L1 images with binning (NxM)

What the binned files look like (img_80x320x10_20x20_L1_101_..._3, checked 2026-11-11):
- 1600 `LINE` markers for 80 rows: one LINE per vertical shift, i.e. NPBIN (20) LINEs in a row, the first 19
  with no pixels after them (vertical binning), then the row's pixels.
- 25600 `EPIX` = 80 rows x 320 pixels; within a row the records are 55 words = 4 x NDCM(10) + 2 + 13 (the
  unbinned per-pixel layout). NCOL in the name / .meta is already the number of BINNED columns.
- So the per-pixel record is unchanged; only the row start is not at a fixed offset.

Plan:
1. `raw_data.l1_skip_means`: find row starts as the last LINE of each run of consecutive LINE markers (or: LINE
   followed by a pixel header word, top nibble 5), instead of `first + r * row_len`. Pixels: the next NCOL
   records of 4 x NDCM + 2 + 13 words, same checks (skip headers, EPIX, per-skip sums vs trailer).
2. Geometry from the name pattern `_<R>x<C>x<S>_<B1>x<B2>_L1_` (any binning) and/or the .meta (NROW, NCOL, NDCM,
   NPBIN, NSBIN); the written FITS carries NPBIN / NSBIN from the .meta, so autoconfig's geometry (prescan and
   overscan columns in binned units: PRESCAN / NSBIN, overscan = PHYSCOL - NCOLPRE ... / NSBIN) works as for the
   DAQ's binned avg images. Check against an avg_ binned FITS of the same settings (avg_img_80x320x1_20x20_L2_*
   exist in the 2026-10-06 run): same ccd_cols, prescan, overscan.
3. Skipped serial binning check: verify the per-pixel sample count n and the trailer word 0x7.. (n x NDCM) are the
   same with NSBIN > 1 (the 75-word gaps seen are the 20 LINEs between rows, not bigger pixel records).
4. Port the fix to ccd_opt_ml's `decode_l1.read_l1` too (the comm_scripts copy is synced from there:
   utils/sync_from_ccd_opt_ml.py), with a test on the 20x20 file; drop the "1x1 only" restriction in raw_data.
5. Truncated files (the 2026-10-06 1x1 file ends after 80 pixels): keep "rows cut by the end are left out"; add
   an option to keep a partial last row padded with the row median for very short files (diagnostics only).

## 2. L1 data diagnostics

Per CCD, from the per-skip values (no FITS needed), written to `L1/<image>/diagnostics.json` + plots, and a
one-line verdict per number (ok / warn) in summary.txt:
- **Transient after charge arrives**: mean of skip k minus the mean of the late skips, against k (all pixels and
  charged pixels separately); number of skips until within 1 sigma -> recommended `--l1-skip-first`.
  (skip_noise_L1.py found ~+170 e- in skip 0, ~15 skips to settle on board 101.)
- **Noise vs N within a pixel** (Allan deviation over blocks of N skips; from skip_noise_L1.py): separates white
  noise from slow drifts; the image noise vs N (already made) adds row-to-row drifts.
- **Drift across a row / column**: pedestal (overscan median) per row and per column, slope and rms; drift of the
  per-skip pedestal within a pixel (first vs last skips).
- **Baseline steps** between rows (bright events spilling into the overscan; seen at 10000 skips).
- **Spectrum of the skip sequence** (per-skip value vs skip number, FFT): pickup lines in skips.
- **Layout health**: pixels failing the checks, rows cut, window sample counts N / N+1.
Reuse skip_noise_L1.py's functions (move its numerics into raw_data.py; keep the CLI as a thin wrapper).

## 3. Commissioning packet (`--packet PACKET.txt`)

Input: a .txt naming the packet's files (a trace, a PSD trace, a ~5 min 1x1 1-skip image, a ~5 min 10x10 500-skip
image; FITS and/or raw .RJ45ADCDATA files; a `# role: trace|psd|image1skip|image500skip` comment per line
optional, else roles from the names: trace_, psd_, x1_1x1_, x500_10x10_). No clustering (fits only).

Output: one self-contained HTML report per CCD: `packet/<module>_ACM<id>_EXT<n>_<ccdname>.html` (26 modules x 4
CCDs = up to 104 reports), plus an index page per run linking them with one summary row per CCD (noise, gain, DC,
flags). CCD identity: ACM id from the file name / .meta ACMID, extension number, the channel's CCD name from the
.meta (ch0: CCD_D ...), module name (DM#, PA#, ... from --module ACM=NAME or the folder name, as --composite).
Note: a trace / PSD file holds one channel; it belongs to the CCD read on that channel.

Tabs (a tab is left out when its input is missing from the packet):
1. **Trace**: interactive decoded trace with zoom / pan over the whole trace (view_trace.py's viewer, already
   generated as viewer.html: embed it), the window flags, per-skip means, folded cycle, quality checks.
2. **PSD**: interactive versions of the four panels of the PDF example: ADC trace, periodogram, Welch (linear),
   Welch (log) with sigma_ADU and sigma_CDS in the legend; overlay of several PSD traces (e.g. two dates); zoom
   on any range. Plotly (one JS file inlined) or uPlot; downsample the trace for display (min/max per pixel
   column) so that 3 M samples stay responsive.
3. **Summary per image**: noise, gain, DC (with errors), chi2/ndf, pedestal per row, sigma per row, the DC-fit
   histogram (interactive), dark charge per row (lambda by row) and per column, hot columns; the 1-skip image:
   noise in ADU / single-skip noise; the 500-skip image: resolved peaks, gain, DC.
4. **Image viewer per image** (ds9-like): canvas with pan / zoom (wheel, drag), contrast (zscale, min/max,
   percentiles), colour maps, pixel value and (row, col) under the cursor, region box with statistics, row /
   column cuts. Data embedded as a compressed float32 / uint16 array (base64) decoded in the browser; for big
   images an image pyramid (2x2 mean levels) so the first view is fast.
On every tab:
- **Comments**: a text box per tab and one for the whole report, saved in the browser (localStorage keyed by the
  report name) and with "save comments" writing a small JSON file the user keeps next to the report (and
  "load comments" to bring them back); the PDF export includes them.
- **Log**: the relevant lines for that tab (file names, settings from the .meta, decoder checks, fit results,
  warnings, versions, timing).
- **Export to PDF**: print stylesheet (each tab on its own page, interactive plots rendered to static images
  first, comments included) + a button calling the browser's print; optionally a server-side PDF (matplotlib
  versions of the same plots + comments JSON) via `--packet-pdf` for batch use.
Implementation: `src/packet_report.py` (data gathering per CCD -> one JSON) and a single HTML template with
inlined JS (no network needed at the LSM). Build and test on the 2026-09-16 traces / PSDs and the 2026-10-06
images first; check sizes (aim < 20 MB per report).

## 4. DC commissioning (`--dc-series`)

Input: a series of images of the same CCDs (times from DATEINI/DATEEND). Per CCD and image: DC from the fit and
from the 0e-/1e- counts (lam01), per row (lambda by row: exposure + readout time), in e-/pix/day using each row's
exposure (EXPOSURE + row x readout time per row, as worked out for the 2026-11-06 skip-noise images).
Output: the usual plots, plus `dc_report.pdf` (matplotlib, one page per module + per CCD): DC vs time (time since
cool-down / since start, with errors), DC per image vs exposure (slope = DC, intercept = per-readout charge such as
CIC / serial-register charge), DC per row (rows read later see more time), a table of DC numbers (mean, trend,
fit of an exponential settling towards a plateau), and flags (DC above a set limit, still falling). With very low
DC (< 1e-3 e-/pix/day) use counts of single-electron pixels with Poisson errors, as in the 2026-11-06 estimate.
Report PDF: matplotlib PdfPages; the same numbers also as CSV.

## 5. Defect map (`--defect-map`)

Existing scripts (LSM_Module_Testing_Analysis/src, D. Caminos's repository, to be ported, not edited there):
- `Image_4.py`: median composite of several .fz images per channel (pedestal subtraction with overscan noise,
  composite median image, event rate per pixel per image, plots).
- `Image_2.py`: column defects / traps on a 4-channel image (pedestal, running median / MAD along rows, trap
  candidates).
Plan: port both into `src/defects.py` as functions (same numbers as the scripts: test on the same input files and
compare outputs), run per CCD under `--defect-map` on the run's images (Image 4 / Image 2 types), and write:
- the composite median FITS, the defect mask FITS (hot pixels, hot / bright columns, traps),
- `defects.csv` / defect log: one line per defect (CCD, type, column / row / pixel, strength, number of images
  in which it appears), and a short summary per CCD,
- an interactive defect map: the image viewer of step 3 with the defect mask as an overlay (toggle per type),
  click on a defect to see its column / row profile and its entry in the log.

## 6. Serial-register (SR) images (`--sr`)

Images of the horizontal register only (no vertical transfers: NROW = number of readouts of the register, each
"row" is the same register read again). Differences to handle:
- geometry: no active-area rows; the "image" is register pixels x reads; prescan / overscan from the .meta;
  autoconfig geometry may need an SR mode (no hot-column logic across rows, rows not independent exposures).
- DC: charge per register pixel per read time (exposure per row = time the register is held); fit per row as now,
  DC vs read number and vs register position (hot register pixels).
- serial-register events: streaks in the serial register (readout artifacts): runs of charged pixels along a row,
  confined to that row; count per image and per read, length / charge distributions, position along the register,
  figure of the number of SR events found.
Need: example SR image files and the group's current SR plots to match (to be provided); then add `--sr` to the
native engine (fits per row, SR event finder = threshold clusters restricted to 1 row, report page).

## Open questions (to settle before / while building)

1. Packet roles: is the packet .txt always 4 files in a fixed order, or should roles come from the names / a
   comment column? Which module-name source (DM#/PA#) is authoritative: .meta, folder name, or a lookup table?
2. PSD tab: which traces are compared by default (same channel, earlier date?), and should sigma_CDS use the
   delayInt window length from the .meta?
3. DC limits for flags (per module type), and whether time = since cool-down needs a logged cool-down time.
4. Defect map: which image types and thresholds the group uses now (from Image_2 / Image_4 defaults), and the
   defect log format expected downstream.
5. SR images: example files and the expected plots.
6. Report storage: comments as JSON next to the HTML (portable) vs a shared Artifact with saved state (needs
   network) - assume local files for the LSM.

## Testing data available locally

- L0: JH1_Testing/2026-09-16-...PA03_DR03_135K_Test1 (trace_*, psd_*; also .meta).
- L1: 2026-10-06 installation check (1x1 file cut after 80 pixels; 20x20 binned files, complete) -> binned
  decoder test; synthetic L1 from decode_l1.write_l1 for the rest.
- Images: production Image 5 (DM*), skip-noise runs (board 101/103), L2 surface run; no SR images yet.
