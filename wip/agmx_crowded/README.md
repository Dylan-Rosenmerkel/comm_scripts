# A_GMX on crowded CCDs — work in progress

Scratch study of A_GMX on very crowded images (L2 ACM101, run 2026-09-22, images 9 and 10). Nothing here is
used by `run_WADERs` / `adaptive.py` yet: every change is applied by wrapping (monkey-patching) the
`clustering_tests` modules at run time. Status, numbers and figures are in `report/muon_report.pdf`. Its
lower-background section is still pending.

Scripts expect to be run from this folder (`python script.py ...`). They import `clustering_tests` and `src`
from `~/Documents/Software/comm_scripts` and read data from `~/Documents/comm_scripts_data`. Outputs go
to `out/`, `reg/` and `inj/` in the working directory.

## The pieces (cumulative "package")

| file | what it does |
|---|---|
| `harness.py` | loads L2 images in electrons with g(n); `link_tracks` (collinear linking of straight-track fragments) |
| `bgv3.py` | `regional_v3`: background override for crowded CCDs (bad-row pedestals from good-row consensus, per-cell truncated-Poisson dark charge); only when read noise < 0.45 e- |
| `xsplit.py` | splits star-shaped clusters where two straight lines cross (RANSAC on cloud centres); wraps `gmx.merge_graph` / `island_arbitration` |
| `streaks.py` | serial-register horizontal streaks found and removed before clustering, reported as `artifact-row` |
| `muons3.py` | **current best straight-track (muon) finder**, see below |
| `muons2.py` | earlier version (cap-rule sharing; cuts crossing tracks, leaves crumbs) |
| `muons.py` | first version (probabilistic Hough; fails on L2) |

## Straight-track finder (`muons3.py`)

1. **Find.** Core = smoothed charge above max(60 e-, 5 noise). Skeleton of the core, standard Hough per
   160-px tile. Each line is followed through the image with a ridge test, allowing crossings with earlier
   tracks.
2. **Accept.** The track is kept if it passes all of these:
   - length ≥ 40 px;
   - occupancy ≥ 0.85;
   - median transverse rms ≤ 2.2 px;
   - not a duplicate of an earlier track;
   - axis continuity (fraction of steps below 0.5× and 0.3× the median axis charge ≤ 0.15 and 0.08). This
     rejects chains of X-ray blobs in dense fields.
3. **Share charge (EM).** In the tube, the muon model is m = A(t)·G(d; σ(t)). The other tracks are modelled
   as o = residual outside the tube, Laplace-filled into it. Each pixel gives the muon R·m/(m+o); A(t) is
   refitted and the step repeated 5 times. The remainder stays in the image for A_GMX, so crossing tracks stay
   whole.
4. **Install.** `muons3.install(adaptive)` runs the finder before `adaptive_cluster` and appends the muon
   clusters (`cls="muon"`, `regime="muon-finder"`).

**Proposed trigger (not implemented).** Enable the finder only when the core fill is above 10 %. Core fill is
27–31 % on L2, 43–47 % on Am-241 and 1.5–2 % on Fe-55 production CCDs. On low-background CCDs the finder
mostly picks straight pieces of electron tracks.

## Validation and figures

| file | purpose |
|---|---|
| `inject_mu.py`, `inj_analyse.py` | 25 simulated muons per image injected into L2, full clustering, per-muon score |
| `em_eval.py`, `joint_eval.py`, `mu_detect*.py` | pixel-level check of the charge sharing against the injected truth (incl. oracle and GMX-as-other tests) |
| `cont_check.py`, `fill_check.py` | continuity cut calibration; core fill per image type (trigger) |
| `regress.py`, `reg_analyse.py` | current vs package (background etc.) on 192 CCD images |
| `regress_mu.py`, `regress_mu3.py` | package vs package + finder (v2 / v3) on 32 lower-background CCD images |
| `full_pkg.py IMG EXT [pkg|pkgmu2|pkgmu3]`, `fullmaps.py`, `fe55_new.py` | L2 label images, maps, Fe-55 |
| `report/figs_*.py`, `report/fe55_l2.py` | figures of `report/muon_report.tex` |
| other `*.py` | one-off diagnostics from the investigation (blob, splits, linking, streaks) |

## Open items

- False positives on straight stretches of electron tracks; a test for charge that continues past a track's
  end along a curve.
- X-rays on a muon's path go to the muon (5–8 % of clean Fe-55 events on L2).
- At muon–muon crossings, the second muon can't take back pixels already given to the first.
- Lower-background regression of v3 to finish.
- Then a proposal for `adaptive.py`: the package, the finder and the trigger.
