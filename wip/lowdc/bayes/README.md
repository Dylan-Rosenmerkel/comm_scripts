# Bayesian few-electron finder for skipper CCDs at ultra-low dark current (wip/lowdc/bayes)

A principled replacement for heuristic thresholding (A_GMX counting finder, `../proto_box.py`) in the regime
read noise s <= 0.15 e-, dark charge lam <= 1e-3 e-/pix, where charge is sparse. Everything here is local and
self-contained; it imports `../sim.py`, `../metrics.py`, `../bench_agmx.py` (configs and seeds) and
`../proto_box.py` but changes nothing outside this folder. Results: `RESULTS.md`.

| file | what |
|---|---|
| `noise.py` | pedestal nuisances (per row; per column only where significant), scale-mixture noise density fitted on the NEGATIVE side of the zero peak, symmetry test, pixel likelihood tables, soft candidates |
| `finder.py` | event model, sigma / depth prior (DAMIC diffusion law), neighbourhoods, numba evidence kernel, posteriors |
| `background.py` | pooled hierarchical dark-charge model (lam0, row gradient, hot columns / hot pixels as discrete Bayesian components) |
| `pipeline.py` | group driver: noise -> pass 1 -> R_N EM -> pooled background -> pass 2 -> off-image (shuffle) null |
| `bench_bayes.py` | benchmark on the identical simulated images of `../bench_agmx.py` -> `results/<prior>/` |
| `bench_agmx_F.py`, `bench_proto_F.py` | the baselines at other nominal fake rates (ROC points) -> `results/agmx_F*`, `results/proto_F*` |
| `evaluate.py` | tables (`results/tables.md`) and figures (`figs/`) |
| `real_l2.py` | the L2 ACM 103 noise-test images -> `results/real_l2/` (`--noise`, `--point` options, v2) |
| `skips.py` | (v2) skip-resolved transient test for candidate pixels: `SkipModel`, `transient_logbf` (hook for per-skip data) |
| `skip_study.py` | (v2) power of the skip test on simulated skip series -> `results/v2/skip_study.md` |
| `ceiling.py` | (v2) analytic accidental-pair limit, finder vs oracle on the same images, population R_N sensitivity, oracle (s, lam) grid -> `results/v2/` |
| `compare.py` | (v2) before / after table of two benchmark runs -> `results/v2/compare.md` |

## Model

Pixel value (electrons; ADU / gain for real data), after the row pedestal o_i (and a column offset u_j / noise
scale kappa_j only where a column deviates by many standard errors, e.g. the first-column artefact):

    v_ij = n_ij + kappa_j eps_ij,     eps ~ f(eps) = sum_k w_k N(0, s_k^2)

**Noise density from the negative side.** No pixel holds negative charge, so v < 0 is pure noise of n = 0 pixels
(contamination < 1e-6). Assuming f symmetric, |v| for v < 0 is a sample of the folded density; a three-component
half-normal mixture (core s, tail <= 0.6 e-, broad >= 1 e- for excursions) is fitted by weighted EM. Weights are
posterior means under a Dirichlet(1, 1, 1) prior, so a component never seen keeps ~1/n_neg weight (a data-sized,
conservative allowance for excursions instead of an exact zero). In a group of images the fit is pooled (all
images share the noise). Symmetry is tested by comparing counts above +a and below -a for a = 2.5-4 s, a < 0.45 e-
(an n = 1 pixel cannot reach there) and reported. No pixel value is ever rounded; `l_ij(n) = f(v_ij - n) / f(v_ij)`.

**Null.** n_ij = d_ij ~ Poisson(lam_ij). The null factorises over pixels.

**Sparsity / soft candidates.** A pixel matters for any hypothesis only if `max_{n>=1} l(n) > r_min` (1e-2; with a
permissive prior P(n>=1) ~ 1e-3 this is P(n >= 1 | v) > ~1e-5). All other pixels are set to n = 0 in every
hypothesis (relative evidence error < r_min). Candidates within `R_link = 4 px` (~ 4 x sigma_max) of each other form a
neighbourhood (distance, NOT pixel connectivity: non-touching electrons of a diffuse event stay together).

**Event model (Neyman-Scott).** Event centres mu at rate R_N per pixel for N electrons (N >= 2), each electron at
mu + N(0, sigma^2 I), pixel-integrated: `p_ij(mu, sigma) = [Phi((i+1-mu_y)/sigma) - Phi((i-mu_y)/sigma)] x [same in x]`.
1-e- events are not part of the model: a single electron of an event and a dark electron have the same pixel
likelihood and the same spatial distribution, so a 1-e- event rate is only measurable as part of lam (stated in
every lam table).

**Exact evidence of one event in a neighbourhood C.** With `g_i(m) = sum_d Pois(d; lam_i) l_i(m + d)` (dark
electrons in the same pixel marginalised), the multinomial over the candidate pixels gives

    Z1(mu, sigma, N) / Z0 = N! [t^N] prod_{i in C} ( sum_m g_i(m) (p_i t)^m / m! )  /  prod_i g_i(0)

i.e. a polynomial product (numba, `finder.evidence_kernel`) - an exact sum over all integer counts per pixel and all
dark/event attributions of the electrons. Electrons landing outside C have likelihood ~0 (non-candidates are n = 0).
The centre is integrated on a sub-pixel grid (step min(0.25, sigma/2) px, out to 2.5 sigma beyond C), sigma on a
grid 0.1..1.3 px with prior weights w_sigma:

    J(N) = sum_sigma w_sigma  int d^2 mu  Z1(mu, sigma, N) / Z0         (per unit event rate)
    posterior odds(one N-e- event : background) = R_N J(N)
    P(no event | data) = 1 / (1 + sum_N R_N J(N) + R^2 J2)

J2 (two events) is approximated by the spatially separated bipartitions of C (cuts of the 3 longest single-linkage
edges >= 1.5 px), each side's J computed with only its pixels; events sharing pixels are not separable anyway.
Groups above 20 e- (sum of positive residuals) are 'big' deposits left to A_GMX; neighbourhoods within 15 px of one
are flagged 'halo' and not evaluated. A neighbourhood with > 40 candidates keeps its 40 most charge-like pixels
(fallback; not reached in the benchmark).

Outputs per neighbourhood: log J1 = log sum_N p(N) J(N) (Bayes factor per unit rate, the ROC score), P(event),
P(2 events), posterior over N (mean / MAP), posterior over sigma and depth z, posterior-mean centre.

**Priors.**
- p(N): uniform on 2..20 for the Bayes factor; the population EM replaces it by the fitted spectrum R_N.
- sigma, `physics`: depth z uniform in 0..675 um mapped through the DAMIC diffusion law
  `sigma_xy^2 = -A ln|1 - b z|`, A = 285 um^2, b = 8.2e-4 um^-1 (DAMIC at SNOLAB, surface cosmic-muon calibration,
  A. Aguilar-Arevalo et al., PRL 125, 241803 (2020), arXiv:2007.15622), giving sigma_max = 15.2 um = 1.01 px. The
  law (with other A, b) is the one of DAMIC, PRD 94, 082006 (2016), arXiv:1607.07410, which with its 2016 values
  (A = 216 um^2, b = 1.37e-3 um^-1) would give 1.58 px; DAMIC-M CCDs have their own (not public in a form I could
  find) calibration - the parameters are inputs (`finder.DIFF_A`, `DIFF_B`).
- sigma, `bench`: uniform 0.2..1.2 px (the simulation's own distribution) - to measure misspecification: the benchmark
  draws sigma up to 1.2 px, beyond the physics prior's 1.01 px.

**Background (pooled, `background.py`).**

    lam_ij = lam0 (1 + beta u_i) + P_hotcol(j) E[h_j] + P_hotpix(ij) E[h_ij]

lam0, beta: Poisson ML on soft dark counts (candidate pixel: P(H0 | neighbourhood) x E[d | v, H0]; beta only kept if
2 dlogL > 9); hot columns: Poisson(mu_j + h E_j) with h log-uniform 1e-6..1 vs Poisson(mu_j), prior fraction fitted
(Beta(1,1)); hot pixels: occupancy over the T images, Binomial(T, 1 - e^{-(lam+h)}) with h log-uniform 1e-2..10 vs
Binomial(T, lam + q_ev) (q_ev = measured chance that an event electron lands in a given pixel of an image) - defects
persist, events do not. A single image falls back to soft counts + weak Gamma prior (0.5 pseudo-electrons at
1e-5 e-/pix), no hot-pixel component.

**Event-rate spectrum (population EM).** R_N over all neighbourhoods of a group: responsibilities
`r_iN = R_N J_i(N) / (1 + sum_N' R_N' J_i(N') + R^2 J2_i)`, `R_N = (sum_i r_iN + ...) / area` - every neighbourhood
contributes its fractional (sub-threshold) responsibility; R_1 is not separable from lam.

**False-alarm control.**
1. Analytic (posterior): selecting P(event) >= t, the expected number of fakes is `sum_selected P(no event | data)`;
   t is chosen so that this equals F x Mpix.
2. Data-driven null: off-image coincidences. Candidates of accepted events (P > 0.5) are removed, then at every
   pixel location the candidate values are permuted among the images of the group (independently per location):
   stationary defects (hot pixels / columns) stay where they are, spatial correlations inside an image (events)
   are destroyed; the finder is rerun on the shuffled images.
3. Benchmark blanks (measured).

## v2 additions (2026-11-11; all OFF by default, the defaults reproduce v1 exactly; results: RESULTS.md section 13)

API additions (all optional; names, arguments and return structures of v1 unchanged):

| where | addition | what |
|---|---|---|
| `noise.EmpiricalNoise` | class | drop-in for `NoiseModel` (`logpdf(e, kap)`, `core`, and `w` / `s` / `lw` of its parametric base for diagnostics; `info` dict). log f tabulated per side on t = asinh(\|e\| / s0), base model beyond the grid |
| `noise.fit_empirical(v, base, side='neg'\|'asym', c_exc=None\|'fit'\|c, method='spline'\|'dp', ...)` | function | semi-parametric density: base mixture x exp(delta(t)), delta a cubic B-spline (knots 0.25 in t) with a Gaussian smoothness prior whose strength is chosen by the Laplace evidence; `info['rel_err']` = posterior sd of delta. `side='asym'`: the positive side has its own correction below e_safe = min(0.45, 1 - 6 s0) e- (no n = 1 pixel reaches there) and the scales c_tail (e_safe..1 e-) / c_exc (> 1 e-, the excursion band) beyond, fitted by `fit_excursion_scale` |
| `noise.fit_excursion_scale(v, mk, e_safe)` | function | Poisson-process likelihood of the positive pixel values beyond e_safe with free occupation fractions of n = 0..12 e- (charge on the integer comb, excursions anywhere), 17 x 17 grid of (c_tail, c_exc), log-uniform prior; returns posterior means and the log-likelihood grid |
| `pipeline.fit_noise(vv, s, noise)` | function | `'mix'` (v1) \| `'empirical'` (auto: asymmetric only if the core symmetry test has \|z\| > `ASYM_Z` = 4 or the positive-side scales differ from 1 by log BF > `ASYM_LBF` = 3) \| `'empirical_sym'` \| `'empirical_asym'` |
| `pipeline.noise_stage(..., noise='mix', mask_cols=())` | kwargs | noise option; columns always masked (e.g. `(0,)`); returns also `valid_eff` |
| `pipeline.population_em_pt(nbs, area, pN0)` | function | R_N plus a rate spectrum R_pt(N) of point deposits (transient single-pixel defects); returns `(R_N, R_pt)` |
| `pipeline.process_group(..., noise='mix', point_defects=False, mask_cols=(), hot_rows=False, blink=False, nm_fixed=None, lam_fixed=None)` | kwargs | see docstring; output adds `R_pt` (None unless point_defects) and `veto` (bool map) |
| `finder.analyse_image` | field | every evaluated neighbourhood gets `logJpt` (N = 0..KCAP): log sum_i g_i(N) / g_i(0), the evidence of a point deposit of N >= 2 e- in one of its pixels |
| `finder.posterior(nb, R_N, R=None, R_pt=None)` | kwarg + fields | with R_pt the point deposit is a further non-event alternative: `p_ev` = event, `p0` = 1 - p_ev (background or point deposit, what the fake budget sums), new `p_bg`, `p_pt` (also set without R_pt: p_bg = p0, p_pt = 0) |
| `background.fit(..., hot_rows=False, blink=False, pi_b0=1e-5)` | kwargs | hot rows (rate excess of a row: serial-register streak / glow, same test as hot columns); blinking pixels ('one strong pixel repeating': a 2-8 e- burst in a fraction b ~ log-U(1e-2, 0.5) of the images vs null and vs Poisson hot pixel); summary adds `hot_rows`, `blink_pix`, `pi_blink`, `veto` (P(blink) > 0.5 pixels, masked in pass 2) |
| `bench_bayes.py --tag --noise --point --mask-col0 --hot-rows --blink` | CLI | options; local config `persist_s0.10_l1e-05` (persistent hot pixels, blinking pixels, one hot row); objects / events get `p_pt`, `lJ2`, `lJ3` (log J(2), log J(3)), `defect` |

Models (v2):
- **Point deposit** (`point_defects`): charge N >= 2 injected into ONE pixel without diffusion (transient hot pixel,
  integer-like excursion), rate R_pt(N) per pixel learned by the population EM next to R_N. The two are separable
  because an N-e- event puts all charge into one pixel only with probability int p^N d^2mu under the sigma prior;
  single-pixel deposits beyond that are attributed to R_pt. P(event) of a single-pixel deposit drops accordingly;
  multi-pixel events are untouched (J_pt is ~0 when two pixels carry charge).
- **Excursions**: the excursion hypothesis is the noise density itself (a pixel value v = n + eps with eps from the
  broad / empirical tail, continuous, competing with every n): its rate and amplitude come from the negative side,
  a positive-only part from `fit_excursion_scale` (c_exc > 1) when the evidence demands it.
- **Persistence**: hot rows, blinking pixels (veto), always-masked columns; the v1 Poisson hot pixels / columns.
- **Skips** (`skips.py`): constant level vs spike / step (k0 marginalised) with white + optional correlated (1/f)
  noise covariance and polynomial drift; exact Gaussian marginal likelihoods; threshold calibrated on charge pixels.

## How to run

    PY=/opt/homebrew/Caskroom/miniforge/base/envs/damic/bin/python
    $PY bench_bayes.py --prior physics --procs 2 --shuffle 2    # ~ results/physics/
    $PY bench_bayes.py --prior bench --procs 2 --shuffle 0      # ~ results/bench/
    $PY bench_agmx_F.py --F 1 ; $PY bench_agmx_F.py --F 0.01    # A_GMX ROC points (key configs)
    $PY bench_proto_F.py --F 0.01,1,10                          # proto_box ROC points
    $PY evaluate.py                                             # results/tables.md, figs/*.png
    $PY real_l2.py --gain 1.3                                   # results/real_l2/
    $PY real_l2.py --gain 0.8 --exts 4 --tag _ext4_g0.8         # gain variant -> results/real_l2_ext4_g0.8/

Library use on one group of images (Y in ADU, valid masks):

    import finder as FD, pipeline as PL
    res = PL.process_group([(Y1, valid1), (Y2, valid2)], FD.Prior("physics"), gain=1.3, nm_pooled=True)
    for nb in res["nbs"]:            # per image: neighbourhood dicts (y, x, pix, logJ1, p_ev, N_mean, pN, ...)
        ...

v2 runs:

    $PY bench_bayes.py --groups $(cat results/v2/groups.txt) --tag v2/before --procs 2
    $PY bench_bayes.py --groups $(cat results/v2/groups.txt) --tag v2/after --procs 2 --point 1 --noise empirical --hot-rows 1 --blink 1
    $PY compare.py --tags v2/before,v2/after --out results/v2/compare.md
    $PY skip_study.py ; $PY ceiling.py pairs ; $PY ceiling.py oracle --group base_s0.15_l0.0001
    $PY ceiling.py pop --group base_s0.15_l0.0001 ; $PY ceiling.py grid --procs 2   # grid: only a small version was run
    $PY real_l2.py --tag _v2_before ; $PY real_l2.py --tag _v2_after --noise empirical --point 1

Machine load: every script uses single-threaded numpy/numba; `bench_bayes.py` runs one process per group
(`--procs`); keep the total of heavy processes <= 2 for this folder (machine limit 6 with the other agents).
