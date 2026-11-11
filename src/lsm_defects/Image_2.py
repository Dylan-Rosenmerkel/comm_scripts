# copied from LSM_Module_Testing_Analysis/src/Image_2.py (commit af3787b) by utils/sync_from_lsm_analysis.py: edit it there, not here
# Diego Venegas-Vargas
# DAMIC-M Collaboration
# Pedestal-subtracts a 4-channel CCD image, detects column defects/traps,
# estimates CTI (X/Y), and plots per-channel views with a summary log.

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
from pathlib import Path
from scipy import stats

# ---------------- Config ----------------
overscan_start = 6152
prescan = 8
e_transient_end = 100
running_window = 40
col_threshold = 6
# For CTI estimates (adjust if needed)
y_overscan_start = 1536
cti_window = 100

# -------------- Robust helpers --------------
def robust_noise_from_overscan(sub, overscan_x_start, clip_sigma=5.0, max_iter=2):
    os_sub = sub[:, overscan_x_start:]
    res = os_sub - np.nanmedian(os_sub, axis=1)[:, None]
    mask = np.isfinite(res)
    for _ in range(max_iter):
        mad = 1.4826 * np.nanmedian(np.abs(np.where(mask, res, np.nan)))
        mad = max(float(mad), 1e-8)
        new_mask = np.abs(res) <= (clip_sigma * mad)
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask
    return float(np.nanstd(np.where(mask, res, np.nan)))

def subtract_pedestal(data, clip_sigma=5.0, max_iter=2):
    os = data[:, overscan_start:]
    row_med = np.nanmedian(os, axis=1)
    for _ in range(max_iter):
        res = os - row_med[:, None]
        mad = 1.4826 * np.nanmedian(np.abs(res))
        mad = max(float(mad), 1e-8)
        keep = np.abs(res) <= (clip_sigma * mad)
        os_kept = np.where(keep, os, np.nan)
        new_row_med = np.nanmedian(os_kept, axis=1)
        row_med = np.where(np.isfinite(new_row_med), new_row_med, row_med)
    subtracted = data - row_med[:, None]
    noise = robust_noise_from_overscan(subtracted, overscan_start,
                                       clip_sigma=clip_sigma, max_iter=max_iter)
    return subtracted, float(round(noise, 2))

# -------- running median/MAD & traps --------
def running_median_absolute_deviation(arr, window_size):
    medians, mads = [], []
    arr = np.where(np.ma.getmask(arr), np.nan, arr)
    for i in range(len(arr)):
        start = max(0, i - window_size // 2)
        end = min(len(arr), i + window_size // 2 + 1)
        medians.append(np.nanmedian(arr[start:end]))
        mads.append(stats.median_abs_deviation(arr[start:end], nan_policy='omit'))
    return np.array(medians), np.array(mads)

def find_col_traps(image, threshold=col_threshold):
    masked_image = np.ma.masked_array(image)
    masked_image[:, :prescan] = np.ma.masked
    masked_image[:, overscan_start - prescan:] = np.ma.masked

    col_medians = np.ma.median(masked_image, axis=0)

    combined_medians = np.empty_like(col_medians)
    combined_mads = np.empty_like(col_medians)

    run_medians, run_mads = running_median_absolute_deviation(
        col_medians[:e_transient_end], running_window)
    combined_medians[:e_transient_end] = run_medians
    combined_mads[:e_transient_end] = run_mads

    combined_medians[e_transient_end:overscan_start] = np.ma.median(col_medians[e_transient_end:])
    combined_mads[e_transient_end:overscan_start] = stats.median_abs_deviation(col_medians[e_transient_end:])

    condition_high = combined_medians + threshold * combined_mads
    condition_low  = combined_medians - threshold * combined_mads

    traps = ((col_medians[prescan:overscan_start - prescan] > condition_high[prescan:overscan_start - prescan]) |
             (col_medians[prescan:overscan_start - prescan] < condition_low[prescan:overscan_start - prescan]))
    trap_indices = np.where(traps)[0] + prescan
    return condition_high, condition_low, trap_indices, col_medians

# -------- CTI (serial X, parallel Y) --------
def calculate_cti_x(col_medians):
    pixels = col_medians[overscan_start - cti_window:overscan_start]
    overscan_val = col_medians[overscan_start]
    overscan = col_medians[overscan_start + cti_window:]
    px_mean = np.mean(pixels); px_std = np.std(pixels)
    os_mean = np.mean(overscan); os_std = np.std(overscan)
    val = px_mean - os_mean
    os_step = overscan_val - os_mean
    if val == 0 or os_step == 0:
        return np.nan, np.nan
    unc_val = np.sqrt(px_std**2 + os_std**2)
    os_unc = np.sqrt(2) * os_std
    cti = os_step / val
    cti_unc = cti * np.sqrt((unc_val / val)**2 + (os_unc / os_step)**2)
    return cti * 100, cti_unc * 100

def calculate_cti_y(row_medians):
    pixels = row_medians[y_overscan_start - cti_window:y_overscan_start]
    overscan_val = row_medians[y_overscan_start + 1]
    overscan = row_medians[y_overscan_start + 1:]
    px_mean = np.mean(pixels); px_std = np.std(pixels)
    os_mean = np.mean(overscan); os_std = np.std(overscan)
    val = px_mean - os_mean
    os_step = overscan_val - os_mean
    if val == 0 or os_step == 0:
        return np.nan, np.nan
    unc_val = np.sqrt(px_std**2 + os_std**2)
    os_unc = np.sqrt(2) * os_std
    cti = os_step / val
    cti_unc = cti * np.sqrt((unc_val / val)**2 + (os_unc / os_step)**2)
    return cti * 100, cti_unc * 100

# -------- display ranges --------
def compute_display_vrange(images, prescan_, overscan_x_start, pct_low=0.5, pct_high=99.5):
    data_pixels = []
    for img in images:
        roi = img[:, prescan_:overscan_x_start]
        data_pixels.append(roi.ravel())
    all_pixels = np.concatenate(data_pixels) if data_pixels else np.array([0.0, 1.0])
    vmin = np.nanpercentile(all_pixels, pct_low)
    vmax = np.nanpercentile(all_pixels, pct_high)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = float(np.nanmin(all_pixels)), float(np.nanmax(all_pixels))
    # Clamp floors
    vmin = max(0.0, float(vmin))
    vmax = max(10.0, float(vmax))

    # Ensure non-degenerate range
    if vmax <= vmin:
        vmax = max(vmin + 1.0, 10.0)
    return float(vmin), float(vmax)

def _finite_vals(x):
    if np.ma.isMaskedArray(x):
        return x.compressed()
    v = np.asarray(x).ravel()
    return v[np.isfinite(v)]

def compute_colpanel_yrange(images, pct_low=0.5, pct_high=99.5, pad_frac=0.05):
    vals = []
    for img in images:
        high, low, _, col_medians = find_col_traps(img)
        vals.append(_finite_vals(col_medians))
        vals.append(_finite_vals(high))
        vals.append(_finite_vals(low))
    all_vals = np.concatenate([v for v in vals if v.size > 0]) if vals else np.array([0.0, 1.0])
    ymin = np.nanpercentile(all_vals, pct_low)
    ymax = np.nanpercentile(all_vals, pct_high)
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin >= ymax:
        ymin, ymax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    pad = pad_frac * (ymax - ymin if (ymax - ymin) > 0 else 1.0)
    return float(ymin - pad), float(ymax + pad)

# ---------------- plotting ----------------
def process_fz_and_plot_all_channels(fz_file, ccd_name, pct_low=0.5, pct_high=99.5,output_dir='.'):
    # First pass: load, subtract, collect images + per-channel noise
    images, noises = [], []
    with fits.open(fz_file) as hdul:
        for i in range(4):
            data = np.array(hdul[i + 1].data)
            image, noise = subtract_pedestal(data)
            images.append(image)
            noises.append(noise)

    # Dynamic ranges across 4 channels
    vmin, vmax = compute_display_vrange(images, prescan, overscan_start, pct_low, pct_high)
    ymin_cm, ymax_cm = compute_colpanel_yrange(images, pct_low, pct_high)

    # Prepare log
    log_path = output_dir / f"{ccd_name}.log"
    log_file = open(log_path, "w")

    # Plot with shared scales
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    axs = axs.flatten()

    for i, image in enumerate(images):
        ax = axs[i]
        im = ax.imshow(image, cmap='cividis', origin='lower',
                       vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f'{ccd_name} – Channel {i}', fontsize=14, fontweight="bold")
        ax.set_xlabel("Column Number", fontsize=12)
        ax.set_ylabel("Row Number", fontsize=12)
        ax.set_ylim(1, 18)  # keep or remove as you prefer

        divider = make_axes_locatable(ax)
        axbottom = divider.append_axes("bottom", size="30%", pad=0.8, sharex=ax)
        axright  = divider.append_axes("right",  size="15%", pad=0.8, sharey=ax)
        cax      = divider.append_axes("right",  size="5%",  pad=0.3)

        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Signal [ADU]", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        # Column & row medians
        condition_high, condition_low, trap_indices, col_medians = find_col_traps(image)
        row_medians = np.ma.median(image, axis=1)

        # Bottom panel (shared y-limits)
        x = np.arange(image.shape[1])
        axbottom.step(x, col_medians,      where='mid', label='Median')
        axbottom.step(x, condition_high,   color='red', linestyle='dashed', label='Median + 6 MAD')
        axbottom.step(x, condition_low,    color='red', linestyle='dotted', label='Median - 6 MAD')
        axbottom.set_ylabel("ADU", fontsize=10)
        axbottom.set_xlabel("Column Number", fontsize=10)
        axbottom.set_title("Column Median", fontsize=10)
        # axbottom.set_ylim(ymin_cm, ymax_cm)
        axbottom.set_ylim(-10, 10)  # keep or remove as you prefer, I think it is not necessary to compute dinamic y-limits for this image
        axbottom.tick_params(labelsize=9)
        axbottom.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5),
                        borderaxespad=0., fontsize=8, frameon=True)
        axbottom.grid(True, linestyle='--', alpha=0.3)

        # Right panel
        axright.step(row_medians, np.arange(image.shape[0]), where='mid')
        axright.set_xlabel("ADU", fontsize=10)
        axright.set_title("Row Median", fontsize=10)
        axright.tick_params(labelsize=9)



        log_file.write("-" * 72 + "\n")

        log_file.write(f"---------------------     {ccd_name} - Channel : {i}   -------------------------\n")
        log_file.write(f"Median Noise : {noises[i]} ADU\n")
        log_file.write(f"Column Defects Location: {trap_indices.tolist()}\n")
        log_file.write(f"Number of Column Defects: {len(trap_indices)}\n")

    log_file.close()

    fig.suptitle(f'Serial Register Defect Maps – {ccd_name}', fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.96, 0.95])
    outname = output_dir / f"{ccd_name}.png"
    plt.savefig(outname, dpi=300)
    plt.close()
    print(f"[SAVED] {outname}")
    print(f"[LOG SAVED] {log_path}")

# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Plot 4-channel CCD views with robust scales")
    parser.add_argument("--files", required=True, help="Input .fz FITS file")
    parser.add_argument("--output_dir", required=True, help="Path for output directory.")
    parser.add_argument("--module", required=True, help="Module/CCD label for outputs (<module>.png/.log)")
    parser.add_argument("--pct-low", type=float, default=0.5, help="Lower percentile for display ranges")
    parser.add_argument("--pct-high", type=float, default=99.5, help="Upper percentile for display ranges")
    args = parser.parse_args()

    fz_file = Path(args.files)
    if not fz_file.exists():
        print(f"[ERROR] File not found: {fz_file}")
        raise SystemExit(1)

    output_dir = Path(args.output_dir)
    process_fz_and_plot_all_channels(str(fz_file), args.module,
                                     pct_low=args.pct_low, pct_high=args.pct_high, output_dir=output_dir)
    print("[DONE] Combined plot and log generated.")

if __name__ == "__main__":
    main()
