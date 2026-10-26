"""Diagrams of the clustering methods for docs/clustering_methods_report.pdf"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _box(ax, x, y, t, c, w=0.2, fs=8):
    ax.text(x, y, t, ha="center", va="center", fontsize=fs, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.45", fc=c, ec="0.3"))


def _arr(ax, x0, y0, x1, y1, t=None):
    ax.annotate("", (x1, y1), (x0, y0), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.2, color="0.25"))
    if t:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.03, t, fontsize=7, ha="center", transform=ax.transAxes, color="0.25")


def fig_methods(fn):
    fig, axs = plt.subplots(4, 1, figsize=(12, 13.5))
    for a in axs: a.axis("off")
    a = axs[0]; a.set_title("WADERs / B auto (hysteresis) and MFD+G (matched filter)", loc="left", fontsize=11, weight="bold")
    _box(a, 0.13, 0.72, "WADERs: pixels > 10 sigma\n(sigma = pedestal-fit width)\nconnected, seed > 30 sigma", "#eee")
    _box(a, 0.13, 0.25, "B auto: pixel >= k e-\n(Poisson tail of the dark charge, 10 %)\nseed > 9.5 e-", "#eee")
    _box(a, 0.4, 0.72, "energy = sum of the cluster pixels", "#f5f5f5")
    _arr(a, 0.25, 0.72, 0.3, 0.72); _arr(a, 0.25, 0.3, 0.3, 0.66)
    _box(a, 0.62, 0.5, "MFD+G: background map (offset, dark charge,\nnoise per pixel) -> matched-filter significance\n(Gaussian kernels 0.5/1/1.5 px) -> peaks > 5 sigma\n+ exact Poisson box test", "#fde")
    _box(a, 0.88, 0.5, "disk R = 3.5 sigma + 0.7 around each\npeak, grown through significant\npixels; E = sum(Q - background)", "#fde")
    _arr(a, 0.76, 0.5, 0.78, 0.5)
    a = axs[1]; a.set_title("MHC - multi-hypothesis clustering (new)", loc="left", fontsize=11, weight="bold")
    _box(a, 0.09, 0.5, "0  background map\n+ hot-column\ncorrection", "#eee", fs=7.5)
    _box(a, 0.3, 0.82, "point finder (MFD+G)", "#fde", fs=7.5)
    _box(a, 0.3, 0.6, "extended regions\n(Poisson tail, alpha 0.01)", "#eed", fs=7.5)
    _box(a, 0.3, 0.38, "row / column streaks", "#ddf", fs=7.5)
    _box(a, 0.3, 0.16, "few-electron counting groups\nP(Poisson) < 1e-5", "#dfd", fs=7.5)
    for y in (0.82, 0.6, 0.38, 0.16): _arr(a, 0.15, 0.5, 0.2, y)
    _box(a, 0.6, 0.6, "where finders overlap: fit the same pixels with\nH_pt = the point Gaussians,  H_ext = a chain of\nGaussian beads along the region's skeleton\n-> extended wins if BIC(H_pt) - BIC(H_ext) > 10", "#ffe", fs=7.5)
    _arr(a, 0.41, 0.82, 0.45, 0.66); _arr(a, 0.42, 0.6, 0.45, 0.6); _arr(a, 0.4, 0.38, 0.45, 0.55)
    _box(a, 0.88, 0.5, "classes: x-ray, x-ray diffuse,\nlow-E, big deposit, alpha,\nmuon, worm, artifact-row/col\n+ significance per event", "#eef", fs=7.5)
    _arr(a, 0.75, 0.6, 0.79, 0.52); _arr(a, 0.43, 0.16, 0.79, 0.45)
    a = axs[2]; a.set_title("adaptive - picks the method per region (new)", loc="left", fontsize=11, weight="bold")
    _box(a, 0.1, 0.5, "background map,\nread noise sigma\n(spread of pixels near 0)", "#eee", fs=7.5)
    _box(a, 0.36, 0.78, "fitted background spread\n>= 3 x read noise?\n(background is not dark current)", "#ffe", fs=7.5)
    _box(a, 0.68, 0.85, "yes -> 'unmodelled': hysteresis on the image\n(10 / 30 sigma, >= Poisson tail), as WADERs", "#eee", fs=7.5)
    _box(a, 0.36, 0.28, "event occupancy per 31x31 box:\npixels above max(1.5 e-, 4 sigma,\nPoisson tail of the dark charge)", "#ffe", fs=7.5)
    _box(a, 0.7, 0.45, ">= 0.3 -> 'crowded': hysteresis\n10 / 30 x the WADERs pedestal sigma\n(default 80 / 240 e-), core energy", "#eee", fs=7.5)
    _box(a, 0.7, 0.12, "< 0.3 -> 'sparse': MHC", "#fde", fs=7.5)
    _arr(a, 0.2, 0.55, 0.26, 0.74); _arr(a, 0.5, 0.8, 0.53, 0.84, "yes"); _arr(a, 0.36, 0.68, 0.36, 0.38, "no")
    _arr(a, 0.48, 0.3, 0.57, 0.43); _arr(a, 0.48, 0.25, 0.6, 0.14)
    a = axs[3]; a.set_title("GMX - Gaussian-mixture EM deblending (from the exploratory agent, verified)", loc="left", fontsize=11, weight="bold")
    _box(a, 0.1, 0.5, "background map\n+ hot columns", "#eee", fs=7.5)
    _box(a, 0.3, 0.5, "seeds: matched-filter\npeaks at 0.5/1/1.5/2.5 px,\nfinest scale first", "#fde", fs=7.5)
    _box(a, 0.53, 0.5, "EM fit of ALL seeds at once:\n2-D Gaussians, full covariance;\nevery pixel's charge SHARED by\nmodel responsibility (no threshold\n-> no percolation in crowded areas)", "#ffe", fs=7.5)
    _box(a, 0.78, 0.72, "merge clouds without a dip\nbetween them (ridge test);\nsharp-on-diffuse kept apart", "#eed", fs=7.5)
    _box(a, 0.78, 0.28, "isolated islands: BIC 'K round\nX-rays' vs 'free shapes' -> tracks;\nfew-e counting groups (as MHC)", "#dfd", fs=7.5)
    _box(a, 0.95, 0.5, "E, shape,\nclass,\nblend", "#eef", fs=7.5)
    _arr(a, 0.16, 0.5, 0.2, 0.5); _arr(a, 0.4, 0.5, 0.42, 0.5); _arr(a, 0.65, 0.56, 0.69, 0.68); _arr(a, 0.65, 0.44, 0.69, 0.32)
    _arr(a, 0.87, 0.7, 0.92, 0.55); _arr(a, 0.87, 0.3, 0.92, 0.45)
    fig.tight_layout(); fig.savefig(fn, dpi=130); return fig


if __name__ == "__main__":
    fig_methods("fig_methods.png")
