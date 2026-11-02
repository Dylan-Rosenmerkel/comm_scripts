# Autor: Nuria Castello Mor
# Fecha: 2-10-2025

from   dataclasses import dataclass, field
from   typing import Dict, Tuple, List, Optional, Sequence, Any
from   glob import glob
from   astropy.io import fits
import numpy as np
import math 
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.io import fits
import os
import re
import math
mpl.rcParams['text.usetex'] = True

def get_norm_from_matrix(M):
    from matplotlib.colors import TwoSlopeNorm
    minv = np.nanmin(M)
    maxv = np.nanmax(M)

    vmin = math.floor(minv)   # tu comportamiento 'min' -> floor
    vmax = math.ceil(maxv)    # tu comportamiento 'max' -> ceil

    return TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# -----------------------------------------
#   Basic utilities
# -----------------------------------------
@dataclass
class SeedCatalog:
    """Cluster seed points for a given CCD
    """
    # position (row, column) for each seed
    # pixel charge in units of electrons 
    # unique cluster identification
    # shape of the image
    ij          : np.ndarray
    S           : np.ndarray
    cluster_id  : np.ndarray
    shape       : Tuple[int,int]
    raw         : bool = False

@dataclass
class PairwiseAlpha:
    """Result alpha for an origin-->destination pair in a single lag
    """
    src                 : str
    dst                 : str
    lag                 : Tuple[int,int]
    n_used              : int
    alpha               : float
    # CI 68 and 95 will be estimated with bootstrap, i.e. (low,high) bootstrap
    alpha_ci68          : Tuple[float,float] 
    alpha_ci95          : Tuple[float,float]
    # Diagnosis of individual ratios (clipped if clipping was applied)
    #   p5, p50, p95, and mad
    ratios_summary      : Dict[str,float]
    # Dependence on energy (k bins): edges (k+1), (k), (k,2), y (k,2)
    energy_bins         : Optional[np.ndarray] = None
    alpha_by_bin        : Optional[np.ndarray] = None
    alpha_by_bin_ci68   : Optional[np.ndarray] = None
    alpha_by_bin_ci95   : Optional[np.ndarray] = None
    counts_by_bin       : Optional[np.ndarray] = None

@dataclass
class CrosstalkResults:
    """Global cross talk results for a given model (i.e. 4 CCDs)
    """
    # labels for each CCD [A,B,C,D] or [CH1,CH2,CH3,CH4]
    order               : List[str]
    lag                 : Tuple[int, int]
    # alpha is a matrix with shape num. of CCDs/module x num. oc CCDs/module
    #   but off-diagonals, as there is no self-crosstalk
    # Be careful:
    #   rows: src
    #   cols: dst
    alpha_matrix        : np.ndarray
    alpha_ci68_matrix   : np.ndarray
    alpha_ci95_matrix   : np.ndarray
    n_matrix            : np.ndarray
    # Detailed results by pair
    per_pair            : Dict[Tuple[str,str], PairwiseAlpha] = field(default_factory=dict)
    # Optional diagnostics
    null_test_alpha_matrix: Optional[np.ndarray] = None
    # Metadata
    meta                : Dict[str,Any] = field(default_factory=dict)

    
    def plot_global_alpha(self, 
            who         : str ='alpha_matrix',
            cmap_name   : str ='RdBu_r', 
            vmin        : Optional[float] =None, 
            vmax        : Optional[float] =None, 
            scale       : float =1e-4, 
            style       : Optional[str] = None,
            fonttextsize: int = 15,
            textcolor   : str = 'white',
            output      : Optional[str] = None,
            ttitle      : Optional[str] = None):

        
        M = getattr(self,who).copy()
        if who=='null_test_alpha_matrix':
            M = getattr(self,'alpha_matrix').copy() - M
        M = M/scale
        np.fill_diagonal(M, np.nan) 

        fig, ax = plt.subplots(figsize=(7.5,6))
        im = ax.imshow(M, cmap=cmap_name, norm=get_norm_from_matrix(M))
        ax.set_xticks(range(len(self.order)))
        ax.set_xticklabels(self.order)
        ax.set_xlabel("source CCD [SRC]", fontsize=fonttextsize)
        ax.set_yticks(range(len(self.order)))
        ax.set_yticklabels(self.order)
        ax.set_ylabel("target CCD [DST]", fontsize=fonttextsize)
        ax.tick_params(axis='x', pad=2)
        ax.tick_params(axis='y', pad=2)
        
        title = '' if ttitle is None else ttitle+": "
        if who.count('ci68')>0:
            title += r"$\alpha$"+ f" matrix CIs 68%, lag={self.lag}"
        elif who.count('ci95')>0:
            title += r"$\alpha$"+ f" matrix CIs 95%, lag={self.lag}"
        elif who.count('null')>0:
            title += f"Corrected Crosstalk Matrix, lag={self.lag}"
        else:
            title += f"Crosstalk Matrix, lag={self.lag}"
        ax.set_title(title)

        for i,src in enumerate(self.order):
            for j,dst in enumerate(self.order):
                if i == j:
                    continue
                a    = M[j,i]
                n    = self.n_matrix[j,i]
                if who in ['alpha_matrix','null_test_alpha_matrix']:
                    lo,hi= self.alpha_ci68_matrix[j,i]/scale
                    _t = ax.text(i,j, f"{a:.2f}\n({lo:.2f},{hi:.2f})\n(n={n})", ha='center', va='center', color=textcolor, fontsize=fonttextsize-2)
                else:
                    _t = ax.text(i,j, f"{a:.2f}\n(n={n})", ha='center', va='center', color=textcolor, fontsize=fonttextsize-2)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if scale!=1:
            label = r"$\times$"+f"{scale:.0e}"
            cbar.ax.set_title(label, loc='right', fontsize=fonttextsize-3)

        fig.subplots_adjust(left=0.02, right=0.90, top=0.90, bottom=0.11)

        if output is not None:
            fig.savefig(output+f"_globabl_alpha.pdf")

        return fig,ax

    def apply_correction_and_save_fits(
        self, images: dict, images_hdr: dict, input_filename: str,
        acm: str, output_dir: str):
        """
        This method builds a single FITS file containing both the crosstalk-corrected and the original (calibrated) images.
        It also merges all individual extension headers into a single global header.

        The output FITS file has the following structure:
            - Primary HDU: first crosstalk-corrected image (`XTALK_1`) with merged header.
            - Extensions:
                * XTALK_2 ... XTALK_4 : remaining crosstalk-corrected images.
                * CALIBRATED_1 ... CALIBRATED_4 : original images.

        The combined header also contains:
            - Crosstalk coefficients from the alpha_matrix (CHiCHj keywords)
            - Crosstalk coefficientes - crosstalk null-test coefficients (CHiCHj_NULL)
            - ACM identifier and lag parameters

        Parameters
        ----------
        images : dict
            Dictionary of image arrays keyed by extension label (e.g. "CH1", "CH2", ...).
        images_hdr : dict
            Dictionary of FITS headers corresponding to each extension.
        input_filename : str
            Path to the original input FITS file. Used to generate the output filename.
        acm : str
            Identifier of the ACM.
        output_dir : str, optional
            Directory for the output file.
        """
        labels = self.order

        alpha_diff_matrix = self.alpha_matrix.copy()
        if self.null_test_alpha_matrix is not None:
            alpha_diff_matrix -= self.null_test_alpha_matrix

        # ---------- Merge headers ----------
        hdr_merged = fits.Header()
        for key_idx, (fname, hdr) in enumerate(images_hdr.items()):
            fileext_value = hdr.get("FILEEXT", None)
            for hkey, hval in hdr.items():
                if hkey in ["NAXIS", "NAXIS1", "NAXIS2", "BITPIX", "FILEEXT"]:
                    continue
                new_key = hkey
                if hkey in hdr_merged:
                    continue
                # ----  Long strings treatment -----
                if isinstance(hval, str) and len(hval) > 68:
                    card = fits.Card(new_key, hval)
                    hdr_merged.append(card)
                    continue
                elif not isinstance(hval, (int, float, str, bool)):
                    hval_str = str(hval)
                    if len(hval_str) > 68:
                        card = fits.Card(new_key, hval_str)
                        hdr_merged.append(card)
                        continue
                    hdr_merged[new_key] = hval_str
                    continue

                hdr_merged[new_key] = hval
        # -------- Add ACM and lag information --------
        hdr_merged["ACM"] = acm
        hdr_merged["LAG0"] = self.lag[0]
        hdr_merged["LAG1"] = self.lag[1]

        # ---------- Add alpha matrix values ----------
        for j, dst in enumerate(labels):
            for i, src in enumerate(labels):
                if i == j:
                    continue
                hdr_merged[f"{dst}{src}"] = float(self.alpha_matrix[j, i])
                if self.null_test_alpha_matrix is not None:
                    hdr_merged[f"{dst}{src}_NULL"] = float(
                self.alpha_matrix[j, i] - self.null_test_alpha_matrix[j, i])

        # ---------- Create HDUs ----------
        hdus = []

        # Crosstalk-corrected
        for idx, dst in enumerate(labels):
            img_corr = images[dst].copy()
            for i, src in enumerate(labels):
                if src == dst:
                    continue
                alpha_null = (
                    self.null_test_alpha_matrix[idx, i]
                    if self.null_test_alpha_matrix is not None
                    else 0.0
                )
                img_corr -= (self.alpha_matrix[idx, i] - alpha_null) * images[src]
            if idx == 0:
                hdr_for_hdu = hdr_merged.copy()
                for key in ["NAXIS", "NAXIS1", "NAXIS2", "BITPIX"]:
                    hdr_for_hdu.pop(key,None)
                primary_hdu = fits.PrimaryHDU(img_corr, header=hdr_for_hdu)
                primary_hdu.header['EXTNAME'] = f"XTALK_{idx+1}"
                primary_hdu.header['EXTEND'] = True  
                hdus.append(primary_hdu)
            else:
                hdus.append(fits.ImageHDU(img_corr, name=f"XTALK_{idx+1}"))

        # Original images
        for idx, dst in enumerate(labels):
            hdus.append(fits.ImageHDU(images[dst].copy(), name=f"CALIBRATED_{idx+1}"))

        # ---------- Save fits file ----------
        hdul = fits.HDUList(hdus)

        base = os.path.basename(input_filename)
        base_no_ext = re.sub(r"_EXT\d+", "", base)
        filename = base_no_ext.replace(".fits", "_xtalkcorrected.fits")
        output_path = os.path.join(output_dir, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        hdul.writeto(output_path, overwrite=True)
        print(f"FITS file saved: {output_path}")

        return
    
    def plot_alpha_vs_energy(self, 
            figsize     : Tuple[float,float] = (12,9),
            marker      : str = "o",
            line_color  : str = "#1f77b4",
            ci_color    : str = "#aec7e8",
            alpha_line_color : str = "#d62728",
            text_alpha_precision: int = 2,
            y_sci_limits: Tuple[int,int] = (-2,2),
            setlogx     : bool = True,
            same_yaxis  : bool = True,
            output      : Optional[str] = None,
            ttitle      : Optional[str] = None):

        from matplotlib.ticker import ScalarFormatter

        def centers_from_edges(edges: np.ndarray) -> np.ndarray:
            return 0.5 * (edges[:-1] + edges[1:])

        n = len(self.order)
        nrows,ncols = n, n-1
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                                    constrained_layout=True)
        
        ymin,ymax = [],[]
        xmin,xmax = [],[]
        for j, dst in enumerate(self.order):
            # dst in a given source, all except self
            srcs = [s for s in self.order if s != dst]
            for i, src in enumerate(srcs):
                # get proper axes in the matrix
                ax = axes[j,i]
                # and results for this specific pair
                pair = self.per_pair.get((dst,src))

                if pair is None:
                    ax.axis('off')
                    ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                    continue

                x   = centers_from_edges(pair.energy_bins)
                y   = pair.alpha_by_bin
                yci = pair.alpha_by_bin_ci68
                a_global = pair.alpha
                
                # Banda CI por bin
                ax.fill_between(x, yci[:, 0], yci[:, 1], color=ci_color, alpha=0.5, linewidth=0)
                # points
                ax.plot(x, y, marker=marker, color=line_color, lw=1.8, ms=4)
                # izontal line for the global alpha
                ax.axhline(a_global, color=alpha_line_color, lw=1.4, ls='--')

                # Add global alpha and number of points per bin
                if np.isfinite(a_global):
                    text = fr"Med.$(\alpha)={a_global:.{text_alpha_precision}e}$"
                    a_null = self.null_test_alpha_matrix[self.order.index(dst),self.order.index(src)]
                    text +="\n"
                    text +=  fr"$\alpha_{{null}}={a_null:.{text_alpha_precision}e}$"
                    ax.text( 0.48, 0.94, text,
                            transform=ax.transAxes, ha="left", va="top", fontsize=9, color=alpha_line_color,
                            bbox=dict(facecolor="white", alpha=0.4, edgecolor="black") )
                    
                    a_a_null = a_global / a_null
                    text = fr"$\alpha/\alpha_{{null}}={a_a_null:.1f}$"
                    ax.text( 0.55, 0.24, text,transform=ax.transAxes, ha="left", va="top", fontsize=12, color="#72467c")
                    
                    for xi,yi,ci in zip(x, y, pair.counts_by_bin):
                        ax.annotate(f"n={int(ci)}", (xi, yi), textcoords="offset points", 
                                xytext=(0, 6), ha='center', fontsize=7, color="#444")
        
                ymin.append( np.min(yci) )
                ymax.append( np.max(yci) )

        # style
        ymin = min(ymin)
        ymax = max(ymax)
        exp = int(np.floor(np.log10(ymax-ymin)))
        step = 10**exp
        ymin_rounded = np.floor(ymin / step) * step
        ymax_rounded = np.ceil(ymax / step) * step
        for j, dst in enumerate(self.order):
            srcs = [s for s in self.order if s != dst]
            for i, src in enumerate(srcs):
                if src==dst:
                    continue
                ax = axes[j,i]
                
                if same_yaxis:
                    ax.set_ylim(ymin_rounded,ymax_rounded)

                if setlogx:
                    ax.set_xscale('log')

                ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.ticklabel_format(axis='y', style='sci', scilimits=y_sci_limits)
                
                if j==3:
                    ax.set_xlabel(r"S ($e^{-}$)", fontsize=12)

                ax.set_ylabel(fr"$\alpha_{{\mathbf{{{dst}}}\leftarrow {src}}}$",fontsize=12)
                ax.tick_params(labelleft=True)

        if ttitle is not None:
            title = fr"{ttitle}: $\alpha_{{\mathrm{{DST}}\leftarrow \mathrm{{SRC}}}}$ vs energy per pair and lag={self.lag}"
        else:
            title = fr"$\alpha_{{\mathrm{{DST}}\leftarrow \mathrm{{SRC}}}}$ vs energy per pair and lag={self.lag}"

        fig.suptitle(title, fontsize=16)
        fig.set_constrained_layout_pads(w_pad=0.10, h_pad=0.05)
        
        if output is not None:
            fig.savefig(output+f"_alpha_vs_energy.pdf")

        return

# -----------------------------------------
#   Extract Seeds from Qmax_map
# -----------------------------------------
def extract_seeds_from_qmax_map(
        qmax_map            : np.ndarray, 
        charge_img_shape    : Tuple[int,int],
        scale               : float = 1e10,
        min_S               : float = 0.0,
        max_S               : Optional[float]=None,
        raw                 : bool = False
        ) -> SeedCatalog:
    """Extract seeds from the encoded map (given by BuildClusterMask process on WADERS). 
    
    The map contain both cluster id and pixel charge, encoded as:

            pixel value = cluster_id * scale + q_pixel

    Args:
        qmax_map        : 2D array, pixels outside seeds = 0; in the seed holds encoded pixel charge
        charge_img_shape: shape (Rows,Cols) of the corresponding charge image (for validation)
        scale           : encoding factor used for cluster_id (default 1e10)
        min_S           : lower threshold of S in e- (exclude S<=min_S)
        max_S           : upper threshold of S in e- (optional)
        raw             : if raw, charge is not encoded, qmax_map contains only pixel charge information
    """
    if qmax_map.shape != charge_img_shape:
        raise ValueError(f"Shape mismatch: qmax_map {qmax_map.shape} vs image {charge_img_shape}")

    # If seeds in map by default should be always positive
    mask = qmax_map > 0
    if not np.any(mask):
        # There is no seeds, return an empty catalog of seeds
        return SeedCatalog( ij=np.empty((0,2), dtype=np.int64),
                            S =np.empty((0,), dtype=np.float64),
                            cluster_id=np.empty((0,), dtype=np.int64),
                            shape=charge_img_shape, raw=raw)
    
    # otherwise extract charge, ids and position for each seed w/ S in (min_S,max_S)
    ij   = np.argwhere(mask)
    vals = qmax_map[mask]
    if raw:
        S = vals.astype(np.float64, copy=False)
    else:
        S = np.remainder(vals, scale).astype(np.float64, copy=False)

    # exclude/filter by S range: S in (min_S, max_S)
    s_mask = (S > min_S)
    if max_S is not None: s_mask &= (S < max_S)
    
    # Extract coordenates only for selected seeds
    if not np.any(s_mask):
        return SeedCatalog( ij=np.empty((0,2), dtype=np.int64),
                            S =np.empty((0,), dtype=np.float64),
                            cluster_id=np.empty((0,), dtype=np.int64),
                            shape=charge_img_shape)

    # apply S region filter 
    ij  = ij[s_mask]
    S   = S[s_mask]
    if raw:
        # cluster id is not encoded on qmax_map, id same for all pixels
        cluster_id = np.ones_like(vals).astype(np.int64, copy=False)
    else:
        # and get cluster id for each 'selected' seed
        cluster_id = np.floor_divide(vals[s_mask], scale).astype(np.int64, copy=False)
    
    return SeedCatalog( ij=ij.astype(np.int64),
                        S=S,
                        cluster_id=cluster_id,
                        shape=charge_img_shape)


# -----------------------------------------
#   Robust estimators for alpha and CIs
# -----------------------------------------
def _median_and_bootstrap(
        ratios  : np.ndarray,
        n_boot  : int = 1000,
        rng     : Optional[np.random.Generator] = None
        ) -> Tuple[float, Tuple[float,float], Tuple[float,float]]:
    """Estimate median and bootstrap CIs (68% and 95%).

        ratios = set(CCD[i,j] / S_{ij}) for all seed
    """
    if ratios.size == 0:
        return np.nan, (np.nan, np.nan), (np.nan, np.nan)

    med = np.median(ratios)
    if n_boot <= 1:
        return med, (np.nan, np.nan), (np.nan, np.nan)

    rng     = np.random.default_rng(rng)
    n       = ratios.size

    try:
        idx     = rng.integers(0, n, size=(n_boot, n), dtype=np.int32, endpoint=False)
        samples = ratios[idx]
        boots   = np.median(samples, axis=1)
    except MemoryError:
        boots = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boots[b] = np.median(ratios[idx])
    
    # Get 0.68 and 0.95 from the boots array, 
    #   which is the alpha values from the resampling
    ci68 = (np.percentile(boots, 16), np.percentile(boots, 84))
    ci95 = (np.percentile(boots, 2.5), np.percentile(boots, 97.5))
    
    return med, ci68, ci95

def _mad(x: np.ndarray) -> float:
    return np.median(np.abs(x - np.median(x)))

def _robust_clip(
        ratios  : np.ndarray,
        k       : float = 5.0,
        scale   : float = 1.4826
        ) -> np.ndarray:
    """Robust clipping of ratios using MAD.

    Keeps only points close to the median: 
        |r - med| <= k * (1.4826 * MAD), 
    filtering out outliers.

    The factor 1.4826 scales the MAD to approximate the STD for normally
    distributed data, making the clipping consistent with Gaussian statistics
    """
    if ratios.size == 0:
        return ratios
    mad = _mad(ratios)
    if mad == 0:
        # nothing to cut
        return ratios
    med     = np.median(ratios)
    keep    = np.abs(ratios - med) <= k * (scale * mad)
    return ratios[keep], keep



# -----------------------------------------
# Estimation of α for an org→dst pair
# -----------------------------------------
def estimate_pair_alpha(
        src           : str,
        dst           : str,
        seeds_src     : SeedCatalog,
        img_dst       : np.ndarray,
        lag           : Tuple[int,int] = (0,0),
        min_S         : float = 0.0,
        max_S         : Optional[float] = None,
        robust_clip_k: Optional[float] = 5.0,
        n_bootstrap   : int = 1000,
        rng           : Optional[np.random.Generator] = None,
        energy_bins   : Optional[Sequence[float]] = None,
        n_energy_bins : Optional[int] = None,
        energy_binning_mode: str = "quantile"
        ) -> PairwiseAlpha:
    """
    Estimates α_{dst←src} using seeds from CCD 'src' and the 'dst' image at (i+di, j+dj).
    
        α = median( img_dst[i+di, j+dj] / S_ij )
    
    Args:
        src             : label for source CCD, e.g. CH1 (or A), ...
        dst             : label for destination CCD, e.g. CH2, ...
        seed_src        : catalog of seeds from the source CCD
        img_dst         : charge image of the destination CCD (same dimensions)
        lag             : optional (di,dj) offset
        min_S           : minimum S, additional filters on charge in source
        max_S           : maximum S, additional filters on charge in source
        robust_clip_k  : if not None, apply MAD-based clipping with k sigmas
        n_bootstrap     : number of bootstrap resamples for CIs
        energy_bins     : explicit edges for S binning (if given, n_energy_bins is ignored)
        n_energy_bins   : number of bins (if edges not given). With 'quantile' 
                                                    uses equal-population quantiles
        energy_binning_mode: 'quantile' or 'uniform' (if n_energy_bins is used)
    """

    R, C   = img_dst.shape
    di, dj = lag

    ij = seeds_src.ij
    S  = seeds_src.S
    
    # mask any seed outside region [min_S,max_S]
    mask_S = (S > min_S)
    if max_S is not None:
        mask_S = np.logical_and(mask_S, S<max_S)

    # Apply lag 
    ii = ij[:, 0] + di
    jj = ij[:, 1] + dj
    mask_in = (ii >= 0) & (ii < R) & (jj >= 0) & (jj < C)
    
    use = np.logical_and(mask_S, mask_in)
    if not np.any(use):
        # no seeds to use
        print(f" -No seeds in Q range ({min_S},{max_S})")
        return PairwiseAlpha(
                src=src, dst=dst, lag=lag, n_used=0,
                alpha=np.nan, 
                alpha_ci68=(np.nan, np.nan), 
                alpha_ci95=(np.nan, np.nan),
                ratios_summary={"p5": np.nan, "p50": np.nan, "p95": np.nan, "mad":np.nan},
                energy_bins=None, alpha_by_bin=None, alpha_by_bin_ci68=None, counts_by_bin=None
                )

    # Only seeds that survive
    ii      = ii[use]
    jj      = jj[use]
    S_use   = S[use]

    # compute ratios (alpha_ij) for each seed
    dst_vals    = img_dst[ii, jj].astype(np.float64)
    ratios_ij   = dst_vals / S_use
    # clipping for robustness
    ratios_keep_ij = None
    if robust_clip_k is not None and ratios_ij.size > 5:
        ratios_ij, ratios_keep_ij = _robust_clip(ratios_ij, k=robust_clip_k)

    # statistics (summary)
    p5 = p50 = p95 = mad = np.nan
    if ratios_ij.size > 0:
        p5, p50, p95 = np.percentile(ratios_ij, [5, 50, 95])
        mad = _mad(ratios_ij)

    # compute median (alpha) and uncertanties by bootstrap
    alpha, ci68, ci95 = _median_and_bootstrap(ratios_ij,n_bootstrap,rng=rng)

    # Compute bin edges for the energy-axis (i.e. S)
    ebins, alpha_bins, alpha_bins_ci68, counts_bins = None, None, None, None
    if (energy_bins is not None) or (n_energy_bins is not None and ratios_ij.size > 0):
        S_sel = S_use[ratios_keep_ij] if ratios_keep_ij is not None else S_sel[:len(ratios)]
        if energy_bins is not None:
            # CASE 1. set format for the given edges vector given by the user
            ebins = np.asarray(energy_bins, dtype=np.float64)
            if ebins.ndim != 1 or ebins.size < 2:
                raise ValueError("energy_bins must be a vector of bin edges")
        else:
            if energy_binning_mode == 'quantile':
                # CASE 2. keep similar number of points per bin
                qs    = np.linspace(0,1, n_energy_bins+1)
                ebins = np.quantile(S_sel, qs)
                eps = 1e-9
                for k in range(1, ebins.size):
                    if ebins[k] <= ebins[k-1]:
                        ebins[k] = ebins[k-1] + eps
            elif energy_binning_mode == 'uniform':
                # CASE 3. Equally spaced bins
                ebins = np.linspace(np.min(S_sel), np.max(S_sel), n_energy_bins + 1)
            else:
                raise ValueError("energy_binning_mode must be 'quantile' or 'uniform'.")
        
        alpha_bins, alpha_bins_ci68, alpha_bins_ci95, counts_bins = [],[],[],[]
        for k in range(ebins.size - 1):
            lo, hi = ebins[k], ebins[k + 1]
            m = (S_sel >= lo) & (S_sel < hi)
            r = ratios_ij[m]
            counts_bins.append(r.size)
            a_bin, ci68_bin, ci95_bin = np.nan, (np.nan, np.nan), (np.nan, np.nan)
            if r.size >= 3:
                a_bin, ci68_bin, ci95_bin = _median_and_bootstrap(r, n_boot=500, rng=rng)
            elif r.size > 0:
                a_bin = np.median(r)

            alpha_bins.append(a_bin)
            alpha_bins_ci68.append([ci68_bin[0], ci68_bin[1]])
            alpha_bins_ci95.append([ci95_bin[0], ci95_bin[1]])
        
        alpha_bins      = np.asarray(alpha_bins, dtype=np.float64)
        alpha_bins_ci68 = np.asarray(alpha_bins_ci68, dtype=np.float64)
        alpha_bins_ci95 = np.asarray(alpha_bins_ci95, dtype=np.float64)
        counts_bins     = np.asarray(counts_bins, dtype=np.int64)
        
    return PairwiseAlpha(
            src=src, dst=dst, lag=lag, n_used=int(ratios_ij.size),
            alpha=float(alpha),
            alpha_ci68=(float(ci68[0]),float(ci68[1])),
            alpha_ci95=(float(ci95[0]), float(ci95[1])),
            ratios_summary={"p5": float(p5), "p50": float(p50),"p95": float(p95), "mad": float(mad)},
            energy_bins=ebins,
            alpha_by_bin=alpha_bins,
            alpha_by_bin_ci68=alpha_bins_ci68,
            counts_by_bin=counts_bins)



# -----------------------------------------
#   Global evaluation: 4x4 matrix for a lag
# -----------------------------------------
def evaluate_crosstalk_matrix(
        images              : Dict[str,np.ndarray],
        seed_catalogs       : Dict[str,np.ndarray],
        labels              : Sequence[str] = ('A','B','C','D'),
        lag                 : Tuple[int,int] = (0,0),
        min_S               : float = 0,
        max_S               : Optional[float] = None,
        robust_clip_k       : Optional[float] = 5.0,
        n_bootstrap         : int = 1000,
        rng                 : Optional[np.random.Generator] = None,
        energy_bins         : Optional[Sequence[float]] = None,
        n_energy_bins       : Optional[int] = 6,
        energy_binning_mode : str = 'quantile',
        do_null_test        : bool = True,
        ) -> CrosstalkResults:
    """
    Evaluates alpha for all (src -> dst) pairs and builds the 4x4 matrix
    """

    labels  = list(labels)
    n       = len(labels)
    
    # checking all iamges has the same shape
    shapes = {lab: images[lab].shape for lab in labels}
    if len(set(shapes.values())) != 1:
            raise ValueError(f"The images do not have the same shape: {shapes}")
    
    alpha_mat = np.full((n,n),   np.nan, dtype=np.float64)
    ci68_mat  = np.full((n,n,2), np.nan, dtype=np.float64)
    ci95_mat  = np.full((n,n,2), np.nan, dtype=np.float64)
    n_mat     = np.zeros((n, n), dtype=np.int64)

    per_pair: Dict[Tuple[str, str], PairwiseAlpha] = {}
    for i_src, src in enumerate(labels):
        # seeds in source
        seeds = seed_catalogs[src]
        for j_dst, dst in enumerate(labels):
            if src == dst:
                # do not compute self-crosstalk has no sense
                continue
            res = estimate_pair_alpha(
                            src,dst,seeds,images[dst],
                            lag,min_S,max_S,robust_clip_k,
                            n_bootstrap,rng,
                            energy_bins,n_energy_bins,energy_binning_mode)

            # fill results at the different matrix
            alpha_mat[j_dst,i_src]  = res.alpha
            ci68_mat[j_dst,i_src,:] = res.alpha_ci68
            ci95_mat[j_dst,i_src,:] = res.alpha_ci95
            n_mat[j_dst,i_src]      = res.n_used
            per_pair[(dst,src)]     = res

    results = CrosstalkResults(
            order=labels, lag=lag,
            alpha_matrix=alpha_mat,
            alpha_ci68_matrix=ci68_mat,
            alpha_ci95_matrix=ci95_mat,
            n_matrix=n_mat,
            per_pair=per_pair,
            meta={"note": f"alpha=median(dst/S) using seeds at same (i,j)+{lag}"})

    
    if do_null_test:
    # random coordenates for seeds, keeping same number of seeds
        null_mat = np.full_like(alpha_mat, np.nan, dtype=np.float64)
        R,C      = images[labels[0]].shape
        rng_local= np.random.default_rng(rng)

        for i_src, src in enumerate(labels):
            Nsrc = seed_catalogs[src].ij.shape[0]
            if Nsrc == 0:
                continue
            # random positions but same lag
            di,dj = lag
            ii = rng_local.integers(max(0, -di), min(R, R - di), size=Nsrc)
            jj = rng_local.integers(max(0, -dj), min(C, C - dj), size=Nsrc)
            # use S from real seeds to build random ratios
            S_src = seed_catalogs[src].S
            if S_src.size > Nsrc:
                S_src = S_src[:Nsrc]
            for j_dst, dst in enumerate(labels):
                if src == dst:
                    continue
                vals   = images[dst][ii + di, jj + dj].astype(np.float64)
                ratios = vals / S_src
                # Clipping
                if robust_clip_k is not None and ratios.size > 5:
                    ratios,keep = _robust_clip(ratios, k=robust_clip_k)

                null_mat[j_dst,i_src] = np.median(ratios) if ratios.size > 0 else np.nan

        results.null_test_alpha_matrix = null_mat

    return results



# -----------------------------------------
#   main function for the cross-talk
# -----------------------------------------
def evaluate_crosstalk(
        images              : Dict[str,np.ndarray],
        qmax_maps           : Dict[str,np.ndarray],
        labels              : Sequence[str] = ("A", "B", "C", "D"),
        qmax_scale          : float = 1e10,
        seed_min_S          : float = 0.0,
        seed_max_S          : Optional[float] = None,
        lags                : Sequence[Tuple[int,int]] = ((0,0),),
        min_S               : float = 0.0,
        max_S               : Optional[float] = None,
        robust_clip_k       : Optional[float] = 5.0,
        n_bootstrap         : int = 1000,
        rng                 : Optional[np.random.Generator] = None,
        energy_bins         : Optional[Sequence[float]] = None,
        n_energy_bins       : Optional[int] = 6,
        energy_binning_mode : str = 'quantile',
        do_null_test        : bool = True,
        ) -> Dict[Tuple[int,int], CrosstalkResults]:
    """
    Full pipeline:
    1) Extract seeds per CCD from qmax_map
    2) For each lag, estimate the alpha matrix and return results
    """

    labels = list(labels)
    # validations
    shapes_img = {lab: images[lab].shape for lab in labels}
    if len(set(shapes_img.values())) != 1:
        raise ValueError(f"Images has different shapes: {shapes_img}")
    shapes_qm = {lab: qmax_maps[lab].shape for lab in labels}
    if set(shapes_img.values()) != set(shapes_qm.values()):
        raise ValueError("qmax_maps and images must have same shape per CCD.")


    # 1) Extract seed per CCD
    seeds = {}
    for lab in labels:
        seeds[lab] = extract_seeds_from_qmax_map(
                                qmax_maps[lab], images[lab].shape,
                                scale=qmax_scale, min_S=seed_min_S, max_S=seed_max_S)

    # 2) Per lag, compute alpha matrix
    results_by_lag: Dict[Tuple[int,int], CrosstalkResults] = {}
    for lag in lags:
        res = evaluate_crosstalk_matrix(
                    images, seeds, labels=labels, lag=lag,
                    min_S=min_S, max_S=max_S, robust_clip_k=robust_clip_k,
                    n_bootstrap=n_bootstrap, rng=rng,
                    energy_bins=energy_bins, n_energy_bins=n_energy_bins,
                    energy_binning_mode=energy_binning_mode,
                    do_null_test=do_null_test)
        results_by_lag[lag] = res

    return results_by_lag


# -----------------------------------------
#   Helpers
# -----------------------------------------
def visual_charge_correlation_plot(
        images  : Dict[str,np.ndarray], 
        labels  : Sequence[str] = ("A", "B", "C", "D"),
        ylim    : Tuple[float,float] = (-2,2),
        xlim    : Tuple[float,float] = (0,3000),
        alpha_null_matrix : Optional[np.ndarray] = None,
        alpha_matrix : Optional[np.ndarray] = None,
        null_matrix  : Optional[np.ndarray] = None,
        title   : Optional[str] = "Crosstalk Characterization across channels: ",
        output  : Optional[str] = None,
        qmaps   : Optional[Dict[str,np.ndarray]] = None
        ):
    """
    Assuming j: rows, dst
             i: cols, src
    """

    fig = plt.figure(figsize=(10,8.5))
    gs  = fig.add_gridspec(4,4, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    
    title += fr" Black line: $\alpha$; Red line: $\alpha - \alpha_{{null}}$"
    fig.suptitle(title, fontsize=16, y=0.94)
    for i,src in enumerate(labels):
        for j,dst in enumerate(labels):
            if j==3:
                axs[j,i].set_xlabel(f"src={src}", fontsize=14)
            if i==0:
                axs[j,i].set_ylabel(f"dst={dst}", fontsize=14)
            if src==dst:
                # no me gusta sin los ejes :(
                #axs[j,i].set_axis_off()
                continue
            
            axs[j,i].scatter(images[src].ravel(),images[dst].ravel(), s=2, color="#4682B4", rasterized=True, marker=",", linewidths=0)            
            axs[j,i].set_ylim(*ylim)
            axs[j,i].set_xlim(*xlim)

            if alpha_null_matrix is not None:
                if alpha_matrix is not None:
                    text = fr"$\alpha={alpha_matrix[j,i]:.2e} \cdot x + n_{{e}}$"
                axs[j,i].text(0.15,0.24,text, transform=axs[j,i].transAxes, ha="left", va="top", fontsize=10, color="k") 
            text = "\n"
            text += fr"$\alpha/\alpha_{{null}}={alpha_null_matrix[j,i]:.1f}$"
            axs[j,i].text(0.15,0.24,text, transform=axs[j,i].transAxes, ha="left", va="top", fontsize=10, color="#72467c")
            
            ### plotting a line on the first peak
            if null_matrix is not None:
                xtalk_corr = lambda x,ne: (alpha_matrix[j,i]-null_matrix[j,i]) * x + ne
            xtalk = lambda x,ne: alpha_matrix[j,i] * x + ne

            for ne in [0,1,2]:
                yvals = [xtalk(xlim[0],ne), xtalk(xlim[1],ne)]
                axs[j,i].plot(xlim, yvals, linestyle='solid', color='black')
                if null_matrix is not None:
                    yvals_corr = [xtalk_corr(xlim[0],ne), xtalk_corr(xlim[1],ne)]
                    axs[j,i].plot(xlim, yvals_corr, linestyle='dashed', color='red')
            
    
    fig.subplots_adjust(top=0.90,bottom=0.08,left=0.08,right=0.95)
    if output is not None:
        fig.savefig(output+f"_visual_xtalk_summary.pdf")

    return fig,gs,axs



# --------------------------------------------------------------------------------
#    Orchestrator function
# --------------------------------------------------------------------------------
def load_data(
        datadir : str,
        infile  : str,
        extname_image : Optional[str] = 'CALIBRATED'
        ):
    """
    """
    slof = glob(f"{datadir}/{infile}")
    if not len(slof)==4:
        raise IOError(f"Expected 4 images from the same module (ACM), i.e. on for each chanel/ext. Found {len(slof)}")
    labels = []
    images,qmax_maps,lof={},{},{}
    for infile in slof:
        # Get the EXT number
        ext_num = int(infile.split('EXT')[-1].split('_')[0])
        # get data and mask from cluster mas
        with fits.open(infile) as hdul:
            ccd_letter = hdul[0].header.get(f'CH{ext_num-1}').strip()[-1]  # 'CCD_D' -> 'D'
            label = f"{ccd_letter}-CH{ext_num-1}"
            images[label] = hdul[extname_image].data
            qmax_maps[label] = hdul['QMAX'].data
        lof[label] = (infile, ext_num)

    return images, qmax_maps, lof


def plot_correction_vs_energy(alpha_matrix, labels, null_alpha_matrix=None, xmin=-2, xmax=5000, output=None):
    
    x = np.linspace(xmin,xmax,100)
    y = np.linspace(xmin,xmax,100)

    y_corr = {}
    for j,dst in enumerate(labels):
        y_corr[dst] = y.copy()
        for i,src in enumerate(labels):
            if dst==src:
                continue
            
            alpha_null = null_alpha_matrix[j,i] if null_alpha_matrix is not None else 0.0
            y_corr[dst] -= (alpha_matrix[j,i] - alpha_null) * x
    
    plt.figure(figsize=(8,5))
    plt.title("Global crosstalk correction vs charge")
    colors = ['#1F77B4','#D62728','#2CA02C','#FF7F0E']
    for dst in labels:
        plt.plot(x, y-y_corr[dst], linestyle='dashed', lw=2, label=dst)

    plt.legend(loc='best')
    plt.xlim(xmin,xmax)
    plt.xlabel(r"charge ($e^{-}$)")
    plt.ylabel(r"$\Delta q^{corr} (e^{-})$")
    plt.tight_layout()
    if output:
        plt.savefig(output)
    return

def plot_pcd(images, alpha_matrix, null_alpha_matrix=None, qmin=-2, qmax=2.5, dq=0.02, output=None):

    n_bins = int(abs(qmax-qmin)/dq)
    img_corr = {}

    fig = plt.figure(figsize=(10,5.4))
    gs  = fig.add_gridspec(2,2)
    axes = gs.subplots()
    ax = axes.flatten()
    
    labels = sorted(images.keys(), key=lambda x: int(x.split('CH')[1]))
    for j,dst in enumerate(labels):
        img_corr[dst] = images[dst].copy()
        _ = ax[j].hist(img_corr[dst].ravel(), n_bins, range=(qmin,qmax), color='black', alpha=0.3, label=f'{dst}: raw')
        for i,src in enumerate(labels):
            if dst==src:
                continue

            alpha_null = null_alpha_matrix[j,i] if null_alpha_matrix is not None else 0.0
            img_corr[dst] -= (alpha_matrix[j,i]-alpha_null) * images[src]

        _ = ax[j].hist(img_corr[dst].ravel(), n_bins, range=(qmin,qmax), histtype='step', linewidth=1, linestyle='solid', color='red', label=f'{dst}: corrected')

        ax[j].set_yscale('log')
        ax[j].legend(loc='best')
        ax[j].set_xlim(qmin,qmax)
        ax[j].set_xlabel(r"charge ($e^{-}$)")
        ax[j].set_ylabel(fr"counts/{dq:.2f}$e^{{-}}$")
    plt.tight_layout()
    if output:
        plt.savefig(output)

    return




