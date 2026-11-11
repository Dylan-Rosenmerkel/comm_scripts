# copied from LSM_Module_Testing_Analysis/src/Image_4.py (commit af3787b) by utils/sync_from_lsm_analysis.py: edit it there, not here
# Diego Venegas-Vargas
# DAMIC-M Collaboration
# Builds a composite median CCD image from multiple FZ files per channel,
# estimates per-pixel event rate, and plots the result.

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
from pathlib import Path
from scipy import stats
import sys
from natsort import natsorted
import glob
from matplotlib.lines import Line2D

# Configuration constants
overscan_start = 6152 # 300
y_overscan_start = 1200
prescan = 8
running_window = 40
col_threshold = 6
cti_window = 100

def compute_event_rate_per_pix_per_image(image, noise_adu, n_images,
                                         prescan=prescan, overscan_start=overscan_start,
                                         roi_center_frac=0.5, threshold_sigma=5.0):
    """
    Estimate an event rate from the composite median image as:
      (# pixels above threshold) / (N_pix_in_ROI * N_images)

    - Uses active area columns [prescan:overscan_start]
    - Uses a central ROI in the active area with linear size = roi_center_frac of the full active extent
      (roi_center_frac=0.5 => central 50% in x and y).
    - Threshold is (threshold_sigma * noise_adu) in ADU, counting only positive excursions.
    """
    if image is None or not np.isfinite(noise_adu) or noise_adu <= 0 or n_images <= 0:
        return np.nan, 0, 0, np.nan

    h, w = image.shape
    x0_act = int(max(0, prescan))
    x1_act = int(min(w, overscan_start))
    if x1_act <= x0_act:
        return np.nan, 0, 0, np.nan

    # active area height is full image
    y0_act, y1_act = 0, h

    # central ROI within active area
    frac = float(roi_center_frac)
    frac = min(max(frac, 0.05), 1.0)
    roi_w = int(round((x1_act - x0_act) * frac))
    roi_h = int(round((y1_act - y0_act) * frac))
    roi_w = max(1, roi_w)
    roi_h = max(1, roi_h)

    cx = (x0_act + x1_act) // 2
    cy = (y0_act + y1_act) // 2
    x0 = max(x0_act, cx - roi_w // 2)
    x1 = min(x1_act, x0 + roi_w)
    y0 = max(y0_act, cy - roi_h // 2)
    y1 = min(y1_act, y0 + roi_h)

    roi = image[y0:y1, x0:x1]
    roi = roi[np.isfinite(roi)]
    n_pix = int(roi.size)
    if n_pix == 0:
        return np.nan, 0, 0, np.nan

    thr_adu = float(threshold_sigma) * float(noise_adu)
    n_events = int(np.sum(roi > thr_adu))
    rate = n_events / (n_pix * n_images)
    return float(rate), n_events, n_pix, float(thr_adu)

def robust_noise_from_overscan(sub, overscan_start, clip_sigma=5.0, max_iter=2):
    """
    Robust RMS from the X-overscan of a pedestal-subtracted image.
    Centers per row, then iteratively MAD-clips outliers (tracks, spikes).
    """
    os_sub = sub[:, overscan_start:]                           # overscan region
    res = os_sub - np.nanmedian(os_sub, axis=1)[:, None]       # row-centered

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
    """
    Robust per-row pedestal using X-overscan with iterative MAD clipping,
    plus robust noise estimate (printed) from the overscan after subtraction.
    """
    os = data[:, overscan_start:]
    row_med = np.nanmedian(os, axis=1)

    for _ in range(max_iter):
        res = os - row_med[:, None]
        # use a global MAD for stability against sparse tracks
        mad = 1.4826 * np.nanmedian(np.abs(res))
        mad = max(float(mad), 1e-8)
        keep = np.abs(res) <= (clip_sigma * mad)
        os_kept = np.where(keep, os, np.nan)
        new_row_med = np.nanmedian(os_kept, axis=1)
        row_med = np.where(np.isfinite(new_row_med), new_row_med, row_med)

    subtracted = data - row_med[:, None]

    noise = robust_noise_from_overscan(subtracted, overscan_start,
                                       clip_sigma=clip_sigma, max_iter=max_iter)
    print("Noise [ADU]:", round(noise, 2))
    return subtracted

# def subtract_pedestal(data):
#     row_medians = np.median(data[:, overscan_start:], axis=1)
#     subtracted = data - row_medians[:, np.newaxis]
#     noise_os = np.std(subtracted[:, overscan_start:])
#     mean_os = np.mean(subtracted[:, overscan_start:])
#     threshold = mean_os + 5 * noise_os
#     valid_pixels = subtracted[:, overscan_start:][subtracted[:, overscan_start:] < threshold]
#     print("Noise [ADU] : ", round(np.std(valid_pixels), 1))
#     return subtracted

def compute_median_images(fz_files):
    all_images = [[] for _ in range(4)]
    all_noises = [[] for _ in range(4)]

    for f in fz_files:
        with fits.open(f) as file:
            for channel in range(1, 5):
                raw = np.array(file[channel].data)
                print(f"Loaded {Path(f).name} channel {channel-1}")
                sub = subtract_pedestal(raw)
                all_images[channel - 1].append(sub)
                # noise_os = np.std(sub[:, overscan_start:])
                # mean_os = np.mean(sub[:, overscan_start:])
                # threshold = mean_os + 5 * noise_os
                # valid = sub[:, overscan_start:][sub[:, overscan_start:] < threshold]
                # all_noises[channel - 1].append(np.std(valid))
                all_noises[channel - 1].append(robust_noise_from_overscan(sub, overscan_start))


    median_images = [np.median(np.stack(imgs), axis=0) for imgs in all_images]
    median_noises = [round(np.median(noises), 1) for noises in all_noises]
    return median_images, median_noises

def running_median_absolute_deviation(arr, window_size):
    medians, mads = [], []
    arr = np.where(np.ma.getmask(arr), np.nan, arr)
    for i in range(len(arr)):
        start = max(0, i - window_size // 2)
        end = min(len(arr), i + window_size // 2 + 1)
        medians.append(np.nanmedian(arr[start:end]))
        mads.append(stats.median_abs_deviation(arr[start:end], nan_policy='omit'))
    return np.ma.masked_array(medians), np.ma.masked_array(mads)

def find_col_traps(image, threshold=col_threshold):
    masked = np.ma.masked_array(image)
    masked[:, :prescan] = np.ma.masked
    masked[:, overscan_start - prescan:] = np.ma.masked
    col_medians = np.ma.median(masked, axis=0)
    mask = np.ma.getmaskarray(col_medians)
    running_medians, running_mads = running_median_absolute_deviation(col_medians, window_size=running_window)
    high = running_medians + threshold * running_mads
    low = running_medians - threshold * running_mads
    traps = ~mask & ((col_medians > high) | (col_medians < low))
    return high, low, traps, np.where(traps)[0]

def calculate_cti_x(col_medians):
    pixels = col_medians[overscan_start - cti_window:overscan_start]
    overscan_val = col_medians[overscan_start]
    overscan = col_medians[overscan_start + cti_window:]
    px_mean = np.mean(pixels)
    px_std = np.std(pixels)
    os_mean = np.mean(overscan)
    os_std = np.std(overscan)
    val = px_mean - os_mean
    unc_val = np.sqrt(px_std**2 + os_std**2)
    os_step = overscan_val - os_mean
    os_unc = np.sqrt(2) * os_std
    cti = os_step / val
    cti_unc = cti * np.sqrt((unc_val / val)**2 + (os_unc / os_step)**2)
    return cti * 100, cti_unc * 100

def calculate_cti_y(row_medians):
    pixels = row_medians[y_overscan_start - cti_window:y_overscan_start]
    overscan_val = row_medians[y_overscan_start + 1]
    overscan = row_medians[y_overscan_start + 1:]
    px_mean = np.mean(pixels)
    px_std = np.std(pixels)
    os_mean = np.mean(overscan)
    os_std = np.std(overscan)
    val = px_mean - os_mean
    unc_val = np.sqrt(px_std**2 + os_std**2)
    os_step = overscan_val - os_mean
    os_unc = np.sqrt(2) * os_std
    cti = os_step / val
    cti_unc = cti * np.sqrt((unc_val / val)**2 + (os_unc / os_step)**2)
    return cti * 100, cti_unc * 100

def compute_display_vrange(median_images, prescan, overscan_start,
                           pct_low=0.5, pct_high=99.5):
    """
    Return (vmin, vmax) as robust percentiles over the data region of the
    four composite images (columns [prescan:overscan_start]).
    Floors: vmin >= 0, vmax >= 10.
    """
    data_pixels = []
    for img in median_images:
        roi = img[:, prescan:overscan_start]   # data region only
        data_pixels.append(roi.ravel())

    if not data_pixels:
        return 0.0, 10.0

    all_pixels = np.concatenate(data_pixels)
    # Keep only finite values
    all_pixels = all_pixels[np.isfinite(all_pixels)]
    if all_pixels.size == 0:
        return 0.0, 10.0

    vmin = np.nanpercentile(all_pixels, pct_low)
    vmax = np.nanpercentile(all_pixels, pct_high)

    # Fallbacks in degenerate cases
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(np.nanmin(all_pixels))
        vmax = float(np.nanmax(all_pixels))

    # Clamp floors
    vmin = max(0.0, float(vmin))
    vmax = max(10.0, float(vmax))

    # Ensure non-degenerate range
    if vmax <= vmin:
        vmax = max(vmin + 1.0, 10.0)

    return float(vmin), float(vmax)


def _finite_vals(x):
    # Extract finite, unmasked values from ndarray or masked array
    if np.ma.isMaskedArray(x):
        return x.compressed()
    v = np.asarray(x).ravel()
    return v[np.isfinite(v)]

# def compute_colpanel_yrange(median_images, prescan, overscan_start,
#                             pct_low=0.5, pct_high=99.5, pad_frac=0.05):
#     """
#     Compute common (ymin, ymax) for the 'Column Median' panel by scanning
#     all 4 composite images. Uses robust percentiles over:
#       - column medians in data region (excludes prescan & overscan)
#       - running thresholds (Median ± N*MAD) so curves aren't clipped.

#     Final bounds are clamped so ymin >= 0 and ymax >= 10.
#     """
#     vals = []
#     for img in median_images:
#         # Same masking as find_col_traps to stay consistent
#         masked = np.ma.masked_array(img)
#         masked[:, :prescan] = np.ma.masked
#         masked[:, overscan_start - prescan:] = np.ma.masked

#         col_medians = np.ma.median(masked, axis=0)
#         # Reuse your existing logic to get high/low curves
#         high, low, _, _ = find_col_traps(img)

#         vals.append(_finite_vals(col_medians))
#         vals.append(_finite_vals(high))
#         vals.append(_finite_vals(low))

#     all_vals = np.concatenate([v for v in vals if v.size > 0]) if vals else np.array([0.0, 1.0])

#     vmin = np.nanpercentile(all_vals, pct_low)
#     vmax = np.nanpercentile(all_vals, pct_high)
#     if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
#         vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

#     # Add a small padding so curves don’t touch the frame
#     span = (vmax - vmin) if np.isfinite(vmax - vmin) and (vmax - vmin) > 0 else 1.0
#     pad = pad_frac * span
#     y0 = vmin - pad
#     y1 = vmax + pad

#     # Clamp floors: ymin >= 0, ymax >= 10
#     y0 = max(0.0, y0)
#     y1 = max(10.0, y1)

#     # Ensure non-degenerate range
#     if y1 <= y0:
#         y1 = y0 + max(1.0, 10.0)

#     return float(y0), float(y1)

def compute_colpanel_yrange(median_images, prescan, overscan_start,
                            pct_low=0.5, pct_high=99.5, pad_frac=0.05):
    """
    Compute common (ymin, ymax) for the 'Column Median' panel by scanning
    all 4 composite images. Uses robust percentiles over:
      - column medians in data region (excludes prescan & overscan)
      - running thresholds (Median ± N*MAD) so curves aren't clipped.
    """
    vals = []
    for img in median_images:
        # Same masking as find_col_traps to stay consistent
        masked = np.ma.masked_array(img)
        masked[:, :prescan] = np.ma.masked
        masked[:, overscan_start - prescan:] = np.ma.masked

        col_medians = np.ma.median(masked, axis=0)
        # Reuse your existing logic to get high/low curves
        high, low, _, _ = find_col_traps(img)

        vals.append(_finite_vals(col_medians))
        vals.append(_finite_vals(high))
        vals.append(_finite_vals(low))

    all_vals = np.concatenate([v for v in vals if v.size > 0]) if vals else np.array([0.0, 1.0])

    vmin = np.nanpercentile(all_vals, pct_low)
    vmax = np.nanpercentile(all_vals, pct_high)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

    # Add a small padding so curves don’t touch the frame
    pad = pad_frac * (vmax - vmin if np.isfinite(vmax - vmin) and (vmax - vmin) > 0 else 1.0)
    return float(vmin - pad), float(vmax + pad)




def save_image_detailed(images, ccd_image_name, log_file, vmin, vmax, colpanel_ylim,
                        output_dir='.', panel_vranges=None):
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    axs = axs.flatten()

    for i, (image, channel, noise) in enumerate(images):
        ax = axs[i]
        if panel_vranges is not None:
            pvmin, pvmax = panel_vranges[i]
        else:
            pvmin, pvmax = vmin, vmax
        im = ax.imshow(image, cmap='cividis', origin='lower', vmin=pvmin, vmax=pvmax, aspect='auto')
        ax.set_title(f'{ccd_image_name} – Channel {channel - 1}', fontsize=14, fontweight="bold")
        ax.set_xlabel("Column Number", fontsize=12)
        ax.set_ylabel("Row Number", fontsize=12)

        divider = make_axes_locatable(ax)
        axbottom = divider.append_axes("bottom", size="30%", pad=0.8, sharex=ax)
        axright  = divider.append_axes("right",  size="15%", pad=0.8, sharey=ax)
        cax      = divider.append_axes("right",  size="5%",  pad=0.3)

        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Signal [ADU]", fontsize=12)

        col_medians = np.ma.median(image, axis=0)
        row_medians = np.ma.median(image, axis=1)
        col_medians[:prescan] = np.nan  # assumes `prescan` exists in your scope

        condition_high, condition_low, _, trap_locations = find_col_traps(image)

        # --- bottom panel (shared y-limits across channels) ---
        x_idx = np.arange(len(col_medians))
        med_line, = axbottom.step(x_idx, col_medians, where='mid', label='Median')
        axbottom.step(np.arange(len(condition_high)), condition_high, color='red',
                      linestyle='dashed', label='Median + 6 MAD')
        axbottom.step(np.arange(len(condition_low)),  condition_low,  color='red',
                      linestyle='dotted', label='Median - 6 MAD')
        axbottom.set_ylabel("ADU", fontsize=10)
        axbottom.set_xlabel("Column Number", fontsize=10)
        axbottom.set_title("Column Median", fontsize=10)

        axbottom.set_ylim(colpanel_ylim)

        # ---- NEW: highlight trap columns in bottom panel ----
        if trap_locations is not None and len(trap_locations) > 0: # to generate files with the mark 
        # if False: # to generate the files to upload
            trap_x = np.asarray(trap_locations, dtype=int)
            trap_y = np.array(col_medians)[trap_x]

            # Get the vertical midpoint of the bottom panel
            ylo, yhi = axbottom.get_ylim()
            y_mid = 0.5 * (ylo + yhi)

            # Plot dots at fixed y position
            axbottom.plot(trap_x, [y_mid] * len(trap_x),linestyle='', marker='o', markersize=3,
                          color='lime',mec='darkgreen', markeredgewidth=0.3,
                          label='defect columns')

            # Update legend
            handles, labels = axbottom.get_legend_handles_labels()
            axbottom.legend(handles, labels, loc='lower left',
                            bbox_to_anchor=(1.0, 0.5),
                            borderaxespad=0., fontsize=8, frameon=True)
        else:
            axbottom.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5),
                            borderaxespad=0., fontsize=8, frameon=True)
 
        # Right panel (row medians)
        axright.step(row_medians, np.arange(image.shape[0]), where='mid')
        axright.set_xlabel("ADU", fontsize=10)
        axright.set_title("Row Median", fontsize=10)

        cti_x, cti_x_err = calculate_cti_x(col_medians)
        cti_y, cti_y_err = calculate_cti_y(row_medians)

        log_file.write("-" * 72 + "\n")
        log_file.write(f"---------------------   {ccd_image_name} - Channel : {i}   -------------------------\n")
        log_file.write(f"Median Noise : {noise} ADU\n")
        log_file.write(f"Column Defects Location: {np.asarray(trap_locations).tolist()}\n")
        log_file.write(f"Number of Column Defects: {len(trap_locations)}\n")
        log_file.write(f"CTI X [%]: {cti_x:.2f} ± {cti_x_err:.2f}\n")
        log_file.write(f"CTI Y [%]: {cti_y:.2f} ± {cti_y_err:.2f}\n")

    fig.suptitle(f'Column Defect Maps – {ccd_image_name}', fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / f"{ccd_image_name}.png", dpi=300)
    plt.close()

def main():
    # # argparse interface
    parser = argparse.ArgumentParser(description="Compose CCD median images from multiple FZ files")
    parser.add_argument(
        "--files",
        nargs="+",                 # <-- accept one or more
        required=True,
        help="One or more .fz files or glob patterns (e.g. avg_Image_4_Low_Temp_106_*_*_*.fz)",
    )
    parser.add_argument("--output_dir", required=True, help="Path for output directory.")
    parser.add_argument("--module", required=True, help="Module name (used for outputs: <module>.png and <module>.log)")
    parser.add_argument("--event-threshold-sigma", type=float, default=5.0,
                        help="Threshold in units of noise (ADU) for counting event pixels in the composite median image. Default 5.")
    parser.add_argument("--event-roi-center-frac", type=float, default=0.5,
                        help="Central ROI linear fraction of the active area used to count event pixels. 0.5 means central 50%% in x and y. Default 0.5.")
    parser.add_argument("--minutes-per-image", type=float, default=12.0,
                        help="Acquisition time per image in minutes, used to convert rates to events/pix/day. Default 12.")
    parser.add_argument("--binning-x", type=int, default=1,
                        help="Hardware/software binning factor in X (columns). For 20x20 binning use 20. Default 1.")
    parser.add_argument("--binning-y", type=int, default=1,
                        help="Hardware/software binning factor in Y (rows). For 20x20 binning use 20. Default 1.")
    parser.add_argument("--independent-colorbars", action="store_true",
                        help="Use independent colorbar ranges for each panel (default: shared range).")
    args = parser.parse_args()

    # Gather files from patterns or explicit paths
    fz_files = []
    for pat in args.files:
        # expand globs
        matches = glob.glob(pat)
        if matches:
            fz_files.extend(matches)
        else:
            # allow explicit single paths (no glob match needed)
            if Path(pat).exists():
                fz_files.append(pat)

    # sort naturally and de-duplicate
    fz_files = natsorted(list(dict.fromkeys(fz_files)))

    if not fz_files:
        print("[ERROR] No .fz files provided or matched.")
        sys.exit(1)


    ccd_image_name = args.module

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    median_images, median_noises = compute_median_images(fz_files)
    images = [(median_images[i], i + 1, median_noises[i]) for i in range(4)]
    # NEW: dynamic display range over the 4 composites
    vmin, vmax = compute_display_vrange(median_images, prescan, overscan_start,
                                        pct_low=0.5, pct_high=99.5)
    panel_vranges = None
    if args.independent_colorbars:
        panel_vranges = []
        for ch in range(4):
            ch_vmin, ch_vmax = compute_display_vrange([median_images[ch]], prescan, overscan_start,
                                                      pct_low=0.5, pct_high=99.5)
            panel_vranges.append((ch_vmin, ch_vmax))
    
    # NEW: common y-limits for the bottom 'Column Median' panels
    ymin_cm, ymax_cm = compute_colpanel_yrange(median_images, prescan, overscan_start,
                                           pct_low=0.5, pct_high=99.5)


    log_file_path = Path(output_dir / f"{ccd_image_name}.log")
    with log_file_path.open('w') as log_file:
        # Event rate estimate in active area central ROI
        n_images = len(fz_files)
        minutes_per_image = float(args.minutes_per_image)
        images_per_day = (1440.0 / minutes_per_image) if minutes_per_image > 0 else np.nan
        bx = max(int(args.binning_x), 1)
        by = max(int(args.binning_y), 1)
        bin_area = float(bx * by)
        log_file.write("=" * 72 + "\n")
        log_file.write("Event rate estimate (composite median image)\n")
        log_file.write(f"Definition: N(pixels > {args.event_threshold_sigma:.2f}*noise) / (N_pix_ROI * N_images)\n")
        log_file.write(f"ROI: central {args.event_roi_center_frac:.2f} of active area; active cols [{prescan}:{overscan_start})\n")
        log_file.write(f"N_images: {n_images}\n")
        log_file.write(f"Minutes per image: {minutes_per_image:.2f}  => images/day: {images_per_day:.2f}\n")
        log_file.write(f"Binning: {bx}x{by}  => bin area: {bin_area:.0f} native pix / binned pix\n")
        for ch in range(4):
            rate, n_evt, n_pix, thr_adu = compute_event_rate_per_pix_per_image(
                median_images[ch],
                noise_adu=float(median_noises[ch]),
                n_images=n_images,
                prescan=prescan,
                overscan_start=overscan_start,
                roi_center_frac=args.event_roi_center_frac,
                threshold_sigma=args.event_threshold_sigma,
            )
            # 'rate' is per *binned* pixel per image (since it's computed on the binned image array)
            rate_bin_per_day = rate * images_per_day if np.isfinite(images_per_day) else np.nan
            # Convert to native pixel rate per day by dividing by the bin area
            rate_pix_per_day = (rate_bin_per_day / bin_area) if (np.isfinite(rate_bin_per_day) and bin_area > 0) else np.nan
            log_file.write(
                f"Channel {ch}: threshold={thr_adu:.3f} ADU; N_evt={n_evt}; N_pix={n_pix}; "
                f"rate={rate:.6e} events/bin/image; {rate_bin_per_day:.6e} events/bin/day; {rate_pix_per_day:.6e} events/pix/day\n"
            )
        log_file.write("=" * 72 + "\n\n")
        save_image_detailed(images, ccd_image_name, log_file, vmin, vmax, (ymin_cm, ymax_cm),
                            output_dir=output_dir, panel_vranges=panel_vranges)

   

    print(f"[DONE] Saved: {ccd_image_name}.png and {ccd_image_name}.log")

if __name__ == "__main__":
    main()
