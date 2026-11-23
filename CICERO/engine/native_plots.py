"""The plots panaSKImg writes with --save-plots, for the native pipeline (native_pipeline.py), same file names:

    <image>_FitDarkCurrentProcess_DCfit_EXT<n>.pdf         pixel-charge histogram and the dark-current fit
    <image>_PedestalSubtractionProcess_Baseline_EXT<n>.pdf  pedestal (baseline) per row
    <image>_PedestalSubtractionProcess_Sigma_EXT<n>.pdf     pedestal sigma per row

So utils/compare_dcfit_vs_skips.py and the usual look at a run work unchanged. The DC-fit plot shows, as panaSKImg's
FitDarkCurrentProcess with x_max_plot: the histogram of the fitted pixels (same window and binning) up to
x_max_plot (autoconfig's set_plot_range: --plot-peaks electrons), the model solid over the fit window and dotted
beyond it, the fit parameters (chi2 / ndf is the Neyman chi2 of the fit) and the readout information.

matplotlib only (Agg), imported when a plot is made; ~0.1 s per plot.
"""
import numpy as np


def plot_range(s, r, n_show=5):
    """x_max_plot [ADU] as autoconfig's set_plot_range for one CCD: with a 1-peak fit the window + 80 %; else up to
    n_show + 0.5 electrons above the zero peak's upper limit (never below x_max). s: fit settings (ext_settings),
    r: autoconfig measurement of the CCD."""
    if s["n_peaks"] == 1:
        return round(s["x_max"] + 0.8 * (s["x_max"] - s["x_min"]), 2)
    if not r.get("resolved") or n_show <= 0:
        return s["x_max"]
    return round(max(s["x_max"], s["mu_max"] + (n_show + 0.5) * r["gain"]), 2)


def _figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.usetex": False, "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    return plt, fig, ax


def dc_fit(path, data, fit, x_max_plot, model, npk, info_lines):
    """The DC-fit plot. data: pedestal-subtracted pixel values used in the fit (mask applied) [ADU]; fit: native fit
    dict (params (value, error), chi2, ndf, settings); model(c, p, npk): the fit function; info_lines: readout text."""
    s = fit["settings"]
    x0, x1 = s["x_min"], s["x_max"]
    bs = (x1 - x0) / max(fit["n_bins"], 1)
    hi = max(x_max_plot, x1)
    nb = max(int(round((hi - x0) / bs)), 1)
    y, edges = np.histogram(data[(data > x0) & (data < x0 + nb * bs)], nb, (x0, x0 + nb * bs))
    c = 0.5 * (edges[1:] + edges[:-1])
    p = [fit[k][0] for k in ("norm", "mu", "sigma", "lam", "gain")]
    plt, fig, ax = _figure()
    ax.step(edges, np.r_[y, y[-1]], where="post", color="black", lw=0.8)
    xs_in = np.linspace(x0, x1, 600)
    ax.plot(xs_in, model(xs_in, p, npk), color="#d62728", lw=2.0)
    if hi > x1:
        npk_plot = max(npk, int(np.ceil((hi - p[1]) / p[4])) + 1)
        xs_out = np.linspace(x1, hi, 600)
        ax.plot(xs_out, model(xs_out, p, npk_plot), color="#d62728", lw=2.0, ls=":")
    ax.set_yscale("log")
    pos = y[y > 0]
    ax.set_ylim(0.6, (pos.max() if pos.size else 10) * 2.5)
    ax.set_xlim(x0, hi)
    ax.set_xlabel("Pixel Charge [ADU]")
    ax.set_ylabel("Counts")
    err = lambda k: fit[k][1] or 0.0
    rows = [("χ² / ndf", f"{fit['chi2']:.1f} / {fit['ndf']}"),
            ("Norm", f"{p[0]:.4g} ± {err('norm'):.3g}"),
            ("μ₀", f"{p[1]:.4g} ± {err('mu'):.3g}"),
            ("σ [ADU]", f"{p[2]:.4f} ± {err('sigma'):.4f}"),
            ("λ [ADU/bin/img]", f"{p[3]:.4g} ± {err('lam'):.3g}"),
            ("gain [ADU/e]", f"{p[4]:.4f} ± {err('gain'):.4f}")]
    txt = "\n".join(f"{a:<16}{b:>24}" for a, b in rows)
    ax.text(0.99, 0.99, txt, transform=ax.transAxes, ha="right", va="top", family="monospace", fontsize=7.5,
            bbox=dict(boxstyle="square,pad=0.4", fc="white", ec="0.6", lw=0.6))
    ax.text(0.99, 0.62, "\n".join(info_lines), transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color="#8c7b6b")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def per_row(path, values, ylabel):
    """Baseline / Sigma plot: one point per row [ADU]."""
    plt, fig, ax = _figure()
    rows = np.arange(values.size)
    ok = np.isfinite(values)
    ax.plot(rows[ok], values[ok], ".", color="black", ms=3.5)
    ax.set_xlabel("row")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, max(values.size, 1) * 1.1)
    ax.ticklabel_format(axis="y", useOffset=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
