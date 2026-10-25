"""Method descriptions for docs/clustering_methods_report.pdf: what each method does, where the ideas come from and
why it works (or where it fails). Numbers live in the results pages."""

METHODS = [
    ("WADERs (panaSKImg ClusterFinder) - the reference", [
        "What: pixels above n_pix x sigma that touch (8 neighbours) form a cluster; it is kept if one pixel is above "
        "n_seed x sigma (10 and 30). sigma is the width of the pedestal fit of the CCD. Energy = sum of the cluster pixels.",
        "Origin: two-threshold (hysteresis) segmentation, as in Canny's edge detector (IEEE PAMI 8, 679, 1986) and the "
        "detection / analysis thresholds of SExtractor (Bertin & Arnouts 1996, A&AS 117, 393).",
        "Why it works / fails: simple and fast. Its thresholds are in units of the pedestal sigma, which on noisy or "
        "high-dark-current CCDs is 5-15x the read noise (7-21 e-): the tails of diffuse clusters are cut (energies "
        "8 % low at sigma_xy 1.8 px), close X-rays merge, few-electron events are invisible. On a crowded CCD the same "
        "inflated sigma (8 e- on Am-241 EXT4) happens to give the right thresholds (80 / 240 e-).",
    ]),
    ("B auto (--cluster-threshold-e auto, in run_WADERs since commit acbac7d)", [
        "What: as WADERs but the pixel threshold is set in electrons from the dark charge: smallest k >= 2 with "
        "P(Poisson(lambda) >= k) <= 0.10, pixel > k - 0.5 e-, seed > 9.5 e-.",
        "Origin: site percolation (8-neighbour threshold p_c = 0.407): if the fraction of dark-charge pixels above "
        "threshold stays far below p_c, noise cannot build connected clusters; a per-pixel false-alarm rate as in FDR "
        "source finding (Hopkins et al. 2002, AJ 123, 1086).",
        "Why it works: thresholds follow the physics of the background (Poisson dark charge), not the pedestal fit; "
        "recovers the charge of diffuse clusters. Still merges close X-rays and cannot see < 10 e- events (seed 9.5 e-).",
    ]),
    ("MFD+G - matched-filter detection + disk + growth (exploratory agent, 2026-10-23)", [
        "What: (1) background map: per-pixel offset, Poisson dark charge and read noise, fitted row by row with clusters "
        "masked; (2) matched-filter significance T = sum w (Q - mean) / sqrt(sum w^2 var) with Gaussian kernels of 0.5, 1 "
        "and 1.5 px; (3) seeds = local maxima with T > 5 that also pass an exact Poisson test of the 3x3 / 5x5 box sum; "
        "(4) centre and width from background-subtracted moments in the seed's Voronoi cell; (5) footprint = disk of "
        "radius 3.5 sigma + 0.7 px, grown through pixels above a Poisson-tail threshold (alpha 0.01, morphological "
        "reconstruction); (6) energy = sum of (Q - background) over the footprint.",
        "Origin: radio-astronomy source finders (SoFiA: Serra et al. 2015, Westmeier et al. 2021; PowellSnakes: Carvalho "
        "et al. 2009; wavdetect: Freeman et al. 2002), the Li & Ma (1983, ApJ 272, 317) counting significance, "
        "morphological reconstruction (Vincent 1993, IEEE TIP 2, 176).",
        "Why it works: detection does not depend on a pixel threshold, and the footprint is sized from the cluster's own "
        "width, so diffuse clusters keep their tails and the background is subtracted pixel by pixel: unbiased energies "
        "at every width and close X-rays resolved. Fails on extended deposits (a track becomes several 'X-rays') and "
        "where the background is not dark current (energies come out high).",
    ]),
    ("hybrid v1 / v2 - MFD+G followed by a merge step (2026-10-24)", [
        "What: MFD+G, then pieces inside a B-mask region at least 14 px across are merged (v1: all; v2: if a lenient "
        "saddle test passes); elsewhere two pieces merge only if the charge on the best path between them never drops "
        "below half of the smaller peak (saddle test).",
        "Origin: SExtractor's deblending contrast (Bertin & Arnouts 1996) and the 'dynamics' of watershed / component "
        "trees (Grimaud 1992; Najman & Couprie 2006), used in reverse: a track has no deep valley between its pieces.",
        "Why it partly works: joins smooth tracks, keeps compact pairs apart. Superseded by MHC: it merges diffuse X-ray "
        "pairs (whose union is > 14 px) and fails on crowded CCDs (everything is one large region).",
    ]),
    ("MHC - multi-hypothesis clustering (new, 2026-10-24, clustering_tests/mhc.py)", [
        "What: (0) background map + hot-column correction (column excess of the quiet pixels, subtracted where > 5 "
        "sigma); (1) specialised finders, each for one kind of deposit: points (MFD+G), extended regions (connected "
        "pixels above a Poisson-tail threshold), row / column streaks (>= 85 % of the charge in <= 2 rows or columns "
        "and a run of >= 8 px), few-electron counting groups (pixels >= 1 e-, kept if P(Poisson(local dark charge) >= "
        "charge) < 1e-5); (2) arbitration where finders overlap: the same pixels are fitted with H_pt = the point "
        "Gaussians and H_ext = a chain of Gaussian beads along the region's skeleton (non-negative amplitudes, Poisson-"
        "weighted chi^2, sparse solver); the extended explanation wins if BIC(H_pt) - BIC(H_ext) > 10; (3) a class for "
        "every event (x-ray, x-ray diffuse, low-E, big deposit, alpha, muon, worm, artifact-row / -column) and a "
        "significance.",
        "Origin: model selection with the Bayesian information criterion (Schwarz 1978, Ann. Stat. 6, 461) as used for "
        "source / model choice in astronomy (The Tractor: Lang, Hogg & Mykytyn 2016); skeletonisation (Zhang & Suen 1984); "
        "few-electron clustering with Poisson significance as in skipper-CCD dark-matter searches (SENSEI, DAMIC-M).",
        "Why it works: instead of one rule for everything, each deposit type is found by the tool that suits it, and "
        "where two explanations compete the data decide with a penalty for extra parameters (a free chain could fit "
        "anything; BIC stops it from swallowing close X-ray pairs). Fails where the background is not dark current "
        "(inherits MFD+G's energies) and is the slowest method (up to ~80 s on a crowded 1.75 Mpix CCD).",
    ]),
    ("adaptive - picks the method per region (new, 2026-10-25, clustering_tests/adaptive.py)", [
        "What: (1) background map and read noise sigma (spread of the pixels near 0 e-); (2) if the fitted background "
        "spread is >= 3x the read noise (and >= 0.5 e-), the background is not dark current + noise -> 'unmodelled': "
        "hysteresis on the image itself (10 / 30 sigma, never below the Poisson tail), as WADERs; (3) otherwise the event "
        "occupancy is measured in 31x31 px boxes (fraction of pixels above max(1.5 e-, 4 sigma, Poisson tail of the dark "
        "charge)): boxes with >= 30 % are 'crowded' -> hysteresis at 10 / 30 x the WADERs pedestal sigma (default 80 / "
        "240 e-), the rest 'sparse' -> MHC (engine 'mhc'); with engine 'gmx' GMX clusters the whole image instead.",
        "Origin: regime switching from what was measured in this study (no single rule wins everywhere); locally "
        "adaptive thresholds (Niblack 1986; Sauvola & Pietikainen 2000) were tried for the crowded regime and rejected "
        "(they fragment the charge sheet).",
        "Why it works: the two failure modes of the model-based methods are recognised from the data (a background that "
        "the dark-current model cannot describe; crowding by events, not by dark current) and handed to the method that "
        "is best there. The occupancy counts only pixels above the dark-charge tail, so high-dark-current Fe-55 CCDs stay "
        "with the model-based method (which is much better than WADERs there).",
    ]),
    ("GMX - Gaussian-mixture EM deblending (exploratory agent, 2026-10-25, clustering_tests/gmx.py)", [
        "What: (0) background map + hot columns; (1) seeds: local maxima of the Poisson-calibrated matched filter at "
        "0.5 / 1 / 1.5 / 2.5 px, finest scale first (a coarse maximum is kept only if no finer seed is close); (2) one "
        "2-D Gaussian with a full covariance per seed, all fitted at once by expectation-maximisation with the pixel "
        "charge as weight; residual peaks become new Gaussians; insignificant ones are dropped; (3) every pixel's charge "
        "is SHARED among the Gaussians in proportion to their model values; (4) Gaussians are merged when the two-cloud "
        "model has no dip between them (unless a sharp one sits on a diffuse one); in isolated islands a BIC test 'K round "
        "X-rays' vs 'free shapes' joins tracks; (5) energy = shared charge (+ significant uncovered pixels next to a "
        "cluster), few-electron groups as MHC, classes as MHC, and 'blend' = charge of other clusters under this one / "
        "its own.",
        "Origin: crowded-field photometry in astronomy (DAOPHOT: Stetson 1987, PASP 99, 191; The Tractor; Scarlet: "
        "Melchior et al. 2018, A&C 24, 129); EM for mixtures (Dempster, Laird & Rubin 1977, JRSS B 39, 1); merging "
        "mixture components by modality / ridgeline (Ray & Lindsay 2005, Ann. Stat. 33, 2042; Hennig 2010, ADAC 4, 3); "
        "scale-space detection (Lindeberg 1994).",
        "Why it works: no threshold decides which pixel belongs to which cluster, so a dense field cannot percolate into "
        "one blob and neighbours do not steal each other's charge; 'blend' gives a per-cluster purity that can be cut on. "
        "This is what beats WADERs on the crowded CCD. Fails where the background is not dark current (like MFD+G).",
    ]),
    ("A_GMX = adaptive + GMX (recommended, 2026-10-25)", [
        "Full description with every step, equations and figures: docs/aGMX/aGMX_clustering.tex (Overleaf: docs/aGMX/aGMX_clustering_overleaf.zip).",
        "What: the adaptive switch (fitted background spread >= 3 x read noise -> WADERs-like hysteresis on the image) in "
        "front of GMX on the whole image. Every cluster also gets the event occupancy of its neighbourhood (information).",
        "Why it works: GMX is the best model-based method in every regime where the background model holds (sparse, "
        "high-dark-current, crowded), and the switch catches the one regime where it does not.",
    ]),
    ("Rejected ideas (tested)", [
        "Track-first then MFD+G (merged close pairs, poor on dense CCDs); saddle merge alone (splits lumpy electron "
        "tracks); local-median thresholds and extra energy rings in crowded regions (fragment the sheet / take the "
        "neighbours' charge); occupancy mask (removes almost all of a crowded CCD); persistence watershed (PWS, cannot "
        "split close pairs); round-Gaussian stamp fits (real front-side X-rays are not round, ~50 % charge lost); "
        "mean-shift / DBSCAN (equivalent to thresholds + connectivity).",
    ]),
]


SUMMARY = [
    ("What this report is", [
        "Seven clustering methods developed or tested on 2026-10-23..25 for skipper-CCD images, compared with the WADERs / "
        "panaSKImg ClusterFinder on: (1) every WADERs-calibrated image on this machine (878 image-CCDs, 16 data sets) with "
        "injected objects of known energy and shape; (2) Fe-55 injection at four cluster widths, close pairs and the real "
        "K-alpha peak; (3) injected muons, worms, alphas, hot columns, X-ray pairs in real Am-241 images; (4) few-electron "
        "events on simulated blanks (efficiency and fakes); (5) the crowded Am-241 CCD (29 images) through the constrained "
        "Np L-line fit. Tables are recomputed from the saved results by make_clustering_report.py.",
    ]),
    ("Result in one paragraph", [
        "A_GMX (adaptive switch + Gaussian-mixture EM deblending) is the best general clusterer: unbiased Fe-55 energies at "
        "every cluster width (WADERs: -8 % at 1.8 px, -26 % on the worst CCD), 2.7x better energy resolution than WADERs, "
        "close X-ray pairs resolved (87-93 % vs 0-11 %), tracks kept whole, few-electron events found without fakes, and on "
        "the crowded Am-241 CCD Np L-beta 19.9 sigma vs 15.5 with the right energy scale (k = 1.001 vs 0.984). It supersedes "
        "MHC, the hybrids and the MHC-engine adaptive. MFD+G stays the fastest choice for isolated X-rays (Fe-55 "
        "calibration); WADERs / B auto stay the pipeline reference and the fallback for images whose background is not "
        "dark current.",
    ]),
    ("Recommendation by situation", [
        "general analysis of any image: A_GMX (adaptive_cluster(..., engine='gmx')), cut blend < 0.02 when purity matters",
        "Fe-55 gain / calibration only: MFD+G (same energies as A_GMX, 5x faster, Fe-55 success 0.972 vs 0.951)",
        "standard pipeline, speed: WADERs or B auto (--cluster-threshold-e auto)",
        "few-electron searches: A_GMX's counting finder (0 fakes; 3 e- at 0.001 e-/pix), or >= 1 e- clustering with its Poisson significance as the fake filter",
    ]),
]

WIDE_INTRO = ("Every WADERs-calibrated image found under ~/Documents (copies removed): 878 image-CCDs. Input = the calibrated "
              "image in electrons (linear gain). Per image-CCD: 12 injected Fe-55 clusters (sigma 0.3-1.6 px), 4 muons, 4 worms, "
              "4 pairs of 5.9 keV X-rays 4 px apart, the same for every method. 'ok' = one cluster with E within 5 % (pairs: both "
              "resolved within 10 %); rms = robust spread of E_rec / E_true of the Fe-55 clusters.")
CROWDED_INTRO = ("Am-241 runs_29img, EXT4 (in one corner every pixel holds > 20 e-), 29 images, gain correction per gain period. "
                 "Constrained fit of 10-30 keV: Np L lines with fixed intensity ratios (Lepy et al. 2008), one energy scale k, "
                 "Compton edge, Ag K fluorescence. Z_np = combined significance of La, Lb, Lg and the 26.3 keV gamma.")
CROWDED_NOTES = ("Checks: the GMX / A_GMX result is the same in the even and the odd images (Z 24.8 and 23.6); an isolation cut "
                 "on WADERs clusters makes WADERs worse (Z 27.1 -> 6.2), so the gain is from charge sharing, not from the cut. "
                 "Caveats: dev/ndf 1.98 vs 1.74 (structure at 25-28 keV); sigma_xy of the new methods is a different width "
                 "estimate from panaSKImg's fitted one and the 0.7 cut was not re-optimised.")
TRACKS_INTRO = ("Injected into real AMTEst images (3 images x 3 seeds): muons (15-60 px), electron worms, 59.5 keV diffuse blobs, "
                "17.75 keV and Fe-55 X-rays, alphas, hot-column streaks, few-electron events, pairs of 5.9 keV X-rays 3 / 4 / 6 px "
                "apart. Success = one cluster with E within 5 % (pairs: both resolved). Same objects for every method.")
LOWE_INTRO = ("Simulated blank images (Poisson dark charge + 0.15 e- read noise, 2 x 1.7 Mpix per dark charge), 800 events of 1-20 "
              "e- spread over 1-5 pixels; efficiency by event charge, the fraction with exactly the right charge at 3 e-, and "
              "fake clusters per Mpix on a second blank with nothing injected.")
FE55_INTRO = ("DM09 / DM10 (2025-08-27), EXT1 and EXT4, images 0-1: 220 injected 1580-e- clusters per width (median E_rec / E_true "
              "and the worst CCD), pairs of sigma-1.4-px clusters 5 px apart (fraction resolved with both within 5 %), and the "
              "real front-side K-alpha (mean / 5.899 keV and its spread over the 8 CCDs).")
CLOSING = [
    ("Does A_GMX supersede the others?", [
        "Yes as the general clusterer. MHC is still slightly better on Fe-55 success in the Fe-55 data sets (0.964 vs 0.951) "
        "and on hot columns (0.78 vs 0.74), but A_GMX is better on crowded, noisy and high-dark-current images, worms, alphas, "
        "close and diffuse pairs, resolution, and 2-300x faster. Hybrids and the MHC-engine adaptive are superseded. MFD+G is "
        "kept for Fe-55 calibration, WADERs / B auto for the pipeline. All code and results are kept (see Files).",
    ]),
    ("Corrections made during the study", [
        "AMTEst EXT4 was first called unusable, then baseline-shifted: both wrong - it is a genuinely crowded CCD (the raw .fz "
        "data show the same). The EXT3 'dark charge 0.5-13 e-/pix' of the early reports was the background map fitted on a "
        "crowded CCD. Real EXT1 'big events' are mostly horizontal 1-5-row streaks (median 1.4 keV). The adaptive (MHC "
        "engine) first used the image median dark charge for its occupancy (rows with high dark charge were called crowded; "
        "energies +1.4 % on DM10 img0) and the background offset in its noisy branch (-2.4 e- on PD06): both fixed.",
    ]),
    ("Open items", [
        "inspect A_GMX labels and track merging by eye on real Am-241 tracks; exact pixel integrals for very sharp clusters; "
        "a background model for the 'unmodelled' PD06 images; re-optimise the sigma_xy cut for A_GMX on the crowded CCD and "
        "understand the 25-28 keV structure; decide whether A_GMX becomes an optional run_WADERs flag (needs approval).",
    ]),
    ("Files", [
        "code: comm_scripts/clustering_tests/ (README); A_GMX document: docs/aGMX/aGMX_clustering.tex (Overleaf: docs/aGMX/aGMX_clustering_overleaf.zip); context: "
        "docs/CONTEXT_2026-10-24.md; results: ~/Documents/comm_scripts_data/session_2026-10-23_clustering/clustering_2026-10-25/.",
    ]),
]
