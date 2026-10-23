#!/usr/bin/env python3
"""Decode a level-1 skipper-CCD image (every skip kept, *.RJ45ADCDATA<ch>) and plot noise vs skips.

A level-1 file holds, for every pixel, the ADC sums of the pedestal and signal windows of
EVERY skip (not only their average), so one image gives the noise for any number of skips N.

File layout (big-endian 32-bit words; checked on img_200x1500x4000_..._L1_101_..._15)
--------------------------------------------------------------------------------------
    START 0xD3544152, BUSY 0xC2555359, one header word, then per row:
    "LINE" 0x4C494E45, then per pixel (4*NSKIP + 15 words):
        NSKIP x [pedestal header, pedestal sum, signal header, signal sum]
            header: top nibble 5, bits 19..0 = samples summed (n, e.g. 150),
                    flag bit 0x80 (bits 27..20) = signal window, low flag nibble = pixel number mod 16
            sum   : 31-bit signed sum of the n ADC samples
        one more pedestal [header, sum]
        pixel trailer, 13 words: [0x7.. header with n*NSKIP], sum over skips of (signal - pedestal)
        as a 62-bit number (2 x 31 bits), further firmware sums, and "EPIX" 0x45504958
    Every row is checked: LINE/EPIX markers, skip headers, and the per-skip sums must add up
    exactly to the trailer's (signal - pedestal) sum.

Pixel value of skip k: (pedestal_k - signal_k) / n  (ADU; charge is positive).
The rows are read in order; a file cut short (copy or readout interrupted) is decoded up
to the last complete row.

Noise vs N (in e-, gain from the electron comb of the full average)
-------------------------------------------------------------------
* first --skip-first skips are dropped (default 20): right after the charge arrives the
  first ~15 skips carry a large transient (skip 0: ~+170 e- on the 101 image), the same in
  every pixel, so it is an offset, not noise, but it would spoil the curves.
* "within pixel" (Allan deviation): the skips of each pixel are cut into blocks of N and
  consecutive block means are compared, sigma_N = robust std of (b[j+1] - b[j]) / sqrt(2)
  over all pixels of a row (median over rows). Charge in the pixel does not matter, and the
  rare pixel whose charge changes during its skips does not pull it up; this is the
  amplifier's own averaging curve. Up to N = skips/2.
* "image": the N-skip average of each pixel (first N skips) minus its electron content
  (known from the full average, where the noise is ~0.1 e-), with one pedestal per row
  (row median), as in an image of N skips. Includes noise slower than one pixel (drifts
  along the row). Only pixels with <= --max-e electrons are used.
* white-noise reference: sigma_1 / sqrt(N), and the spectrum of the skip sequence
  (frequency in 1/skip).

Composite images (--composites N)
---------------------------------
The used skips (after --skip-first) are cut into N consecutive blocks of L = (used skips) // N
skips, and every block is averaged into its own image: N images of L skips each (left-over
skips at the end are not used). E.g. 3980 used skips, --composites 4: 4 images of 995 skips
(skips 20-1014, 1015-2009, ...). Written to OUTDIR/<name>_composites_N<N>/ as
<name>_skips<first>-<last>.fits (skip numbers count from 0 in the raw file), with a list
of the files (composites.txt) and composites.csv (skip range and noise of every image,
same 'image' method as above: shows whether the noise changes during the pixel's skips).

Usage
    python skip_noise_L1.py FILE.RJ45ADCDATA0 -o OUTDIR [--skip-first 20] [--gain 1.41] [--max-rows N]
    python skip_noise_L1.py FILE.RJ45ADCDATA0 -o OUTDIR --composites 4      # + 4 images of 1/4 of the skips
    (-o defaults to the current folder)
Output: OUTDIR/<name>_noise_vs_skips.png, .csv, <name>_avg.fits (average of the used skips)
        [OUTDIR/<name>_composites_N<N>/ with --composites]
Needs numpy, astropy, matplotlib, and comm_scripts/src/autoconfig_panaSKImg.py (gain finder):
unlike the other utils/ scripts this one needs src/ next to utils/.
"""
import argparse
import csv
import os
import re
import sys
import time

import numpy as np

# the gain finder comes from autoconfig_panaSKImg.py in comm_scripts/src (keep utils/ next to src/)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

LINE, EPIX = 0x4C494E45, 0x45504958
TRAILER = 13


def s31(x):
    """31-bit two's complement (bit 31 of the data words is never set)."""
    x = x & 0x7FFFFFFF
    return x - ((x >> 30) << 31)


def geometry_from_name(path):
    m = re.search(r"_(\d+)x(\d+)x(\d+)_\d+x\d+_L1_", os.path.basename(path))
    return tuple(int(v) for v in m.groups()) if m else None


def read_rows(path, ncols, nskip, max_rows=None):
    """Yield (row, samples per window, ped, sig) with ped/sig int64 [ncols, nskip] (window sums)."""
    w = np.memmap(path, dtype=">u4", mode="r")
    pix = 4 * nskip + 2 + TRAILER
    row_len = 1 + ncols * pix
    first = np.where(np.asarray(w[:256]) == LINE)[0]
    if not first.size:
        raise ValueError("no LINE marker at the start: not a level-1 image?")
    start = int(first[0])
    complete = (len(w) - start) // row_len
    left = (len(w) - start) - complete * row_len
    if left:
        print(f"note: file ends inside row {complete} ({left / row_len:.0%} of it); "
              f"decoding the {complete} complete rows")
    for r in range(min(complete, max_rows or complete)):
        o = start + r * row_len
        blk = np.asarray(w[o:o + row_len])
        if blk[0] != LINE:
            raise ValueError(f"row {r}: no LINE marker (word {o}); layout differs from the expected one")
        p = blk[1:].reshape(ncols, pix).astype(np.int64)
        hdr = p[:, 0:4 * nskip:2]
        n = int(hdr[0, 0] & 0xFFFFF)
        if np.any(p[:, -1] != EPIX) or np.any((hdr >> 28) != 5) or np.any((hdr & 0xFFFFF) != n):
            raise ValueError(f"row {r}: unexpected pixel layout")
        ped, sig = s31(p[:, 1:4 * nskip:4]), s31(p[:, 3:4 * nskip:4])
        tot = s31(p[:, 4 * nskip + 3]) * (1 << 31) + (p[:, 4 * nskip + 4] & 0x7FFFFFFF)
        if np.any(tot != (sig - ped).sum(axis=1)):
            raise ValueError(f"row {r}: per-skip sums do not add up to the pixel trailer")
        yield r, n, ped, sig


def robust_std(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def n_values(nmax):
    return sorted({int(round(v)) for v in np.logspace(0, np.log10(nmax), 45)} | {nmax})


def main():
    ap = argparse.ArgumentParser(description="Decode a level-1 (per-skip) skipper image and plot noise vs skips.")
    ap.add_argument("infile")
    ap.add_argument("-o", "--outdir", default=".")
    ap.add_argument("--geometry", metavar="ROWSxCOLSxSKIPS", help="default: from the file name (img_200x1500x4000_...)")
    ap.add_argument("--skip-first", type=int, default=20, help="skips dropped at the start of every pixel (default 20)")
    ap.add_argument("--gain", type=float, help="ADU/e- (default: from the electron comb of the full average)")
    ap.add_argument("--max-e", type=float, default=20, help="'image' noise: pixels with at most this many e- (default 20)")
    ap.add_argument("--max-rows", type=int, help="decode only the first rows (quick look)")
    ap.add_argument("--composites", type=int, metavar="N",
                    help="also write N images, each the average of a consecutive 1/N of the used skips")
    args = ap.parse_args()

    geo = (tuple(int(v) for v in args.geometry.lower().split("x")) if args.geometry
           else geometry_from_name(args.infile))
    if not geo:
        sys.exit("cannot read ROWSxCOLSxSKIPS from the file name; give --geometry")
    nrows, ncols, nskip = geo
    K = args.skip_first
    M = nskip - K
    ns = n_values(M)
    nb = [n for n in ns if M // n >= 2]                 # block sizes for the within-pixel method
    name = os.path.basename(args.infile).split(".")[0]
    NC = args.composites
    if NC is not None:
        if not 1 <= NC <= M:
            sys.exit(f"--composites must be between 1 and {M} (the used skips)")
        LC = M // NC                                    # skips per composite image
        gb = NC * min(nrows, args.max_rows or nrows) * ncols * 4 / 1e9
        print(f"composites: {NC} images of {LC} skips"
              + (f" ({M - NC * LC} left-over skips at the end not used)" if M % NC else "")
              + (f"; needs ~{gb:.1f} GB of memory" if gb > 1 else ""))
    os.makedirs(args.outdir, exist_ok=True)

    t0 = time.time()
    img_n = []                        # [row] -> [len(ns), ncols] first-N-skip averages (ADU)
    s_rows = []                       # [row] -> within-pixel (Allan) sigma per block size (ADU)
    prof = np.zeros(nskip)            # median over pixels per skip index (transient), summed over rows
    psd = None
    comp = []                         # [row] -> [NC, ncols] composite images (ADU), with --composites
    for r, n, ped, sig in read_rows(args.infile, ncols, nskip, args.max_rows):
        q = (ped - sig) / n                                         # ADU per skip, charge positive
        prof += np.median(q - np.median(q[:, K:], axis=1, keepdims=True), axis=0)
        d = q[:, K:]
        c = np.cumsum(d, axis=1)
        img_n.append(np.stack([c[:, k - 1] / k for k in ns]))
        if NC is not None:
            comp.append(d[:, :NC * LC].reshape(ncols, NC, LC).mean(axis=2).T.astype(np.float32))
        y = d - d.mean(axis=1, keepdims=True)
        y = y - np.median(y, axis=0)                                # pattern common to all pixels: not noise
        s_r = []
        for k in nb:
            m = M // k
            b = y[:, :m * k].reshape(ncols, m, k).mean(axis=2)
            s_r.append(robust_std(np.diff(b, axis=1).ravel()) / np.sqrt(2))
        s_rows.append(s_r)
        if r % 4 == 0:                                              # spectrum from every 4th row
            P = np.abs(np.fft.rfft(y, axis=1)) ** 2 / M
            psd = P.sum(axis=0) if psd is None else psd + P.sum(axis=0)
        if r % 20 == 0:
            print(f"  row {r}  ({time.time() - t0:.0f} s)", flush=True)
    rows = len(img_n)
    if not rows:
        sys.exit("no complete row in the file")
    img = np.stack(img_n, axis=1)                                   # [len(ns), rows, ncols]
    prof /= rows
    psd /= sum(1 for r in range(rows) if r % 4 == 0) * ncols
    print(f"decoded {rows} rows x {ncols} cols x {nskip} skips ({n} samples per window) in {time.time() - t0:.0f} s")

    # ---- gain from the comb of the full average
    full = img[-1] - np.median(img[-1], axis=1, keepdims=True)
    import autoconfig_panaSKImg as A
    g = args.gain
    if g is None:
        g0 = A.find_gain(A.clip_tails(full.ravel()), 0.6, 3.0)
        if g0 is None:
            sys.exit("no electron comb in the full average; give --gain")
        g = A.refine_gain(A.clip_tails(A.align_rows(full, g0).ravel()), g0)[0]
    print(f"gain {g:.4f} ADU/e-")

    # ---- electron content of every pixel (full average), then the 'image' noise per N
    xa = A.align_rows(full, g)
    off = A.comb_offset(A.clip_tails(xa.ravel()), g)
    ne = np.round((xa - off) / g)
    use = (ne >= 0) & (ne <= args.max_e)
    s_img = []
    for i, k in enumerate(ns):
        res = img[i] - ne * g                                       # noise + pedestal of the row
        res = res - np.nanmedian(np.where(use, res, np.nan), axis=1, keepdims=True)
        s_img.append(robust_std(res[use]) / g)
    s_img = np.array(s_img)
    s_blk = np.median(np.array(s_rows), axis=0) / g
    s1 = s_blk[0]
    f = np.fft.rfftfreq(M)
    white = np.sqrt(np.median(psd[(f > 0.02) & (f < 0.05)])) / g    # white-noise floor per skip (e-)

    # ---- composite images (--composites)
    if NC is not None:
        from astropy.io import fits
        cdir = os.path.join(args.outdir, f"{name}_composites_N{NC}")
        os.makedirs(cdir, exist_ok=True)
        comp = np.stack(comp, axis=1)                               # [NC, rows, ncols]
        paths, ctab = [], []
        for j in range(NC):
            a, b = K + j * LC, K + (j + 1) * LC - 1
            res = comp[j] - ne * g
            res = res - np.nanmedian(np.where(use, res, np.nan), axis=1, keepdims=True)
            sj = robust_std(res[use]) / g
            hdu = fits.PrimaryHDU(comp[j])
            hdu.header.update(NSKIPS=(LC, "skips averaged"), SKIPSTRT=(a, "first skip (0 = first in file)"),
                              SKIPEND=(b, "last skip"), COMPIDX=(j, f"composite {j + 1} of {NC}"), NSAMP=n,
                              GAIN=(round(g, 5), "ADU/e- (full average)"), NOISE_E=(round(float(sj), 5), "e-"),
                              NROWSF=(rows, "complete rows decoded"), SRCFILE=os.path.basename(args.infile)[:68])
            pth = os.path.join(cdir, f"{name}_skips{a}-{b}.fits")
            hdu.writeto(pth, overwrite=True)
            paths.append(os.path.abspath(pth))
            ctab.append((j, a, b, LC, sj))
        with open(os.path.join(cdir, "composites.txt"), "w") as fh:
            fh.write("".join(p + "\n" for p in paths))
        with open(os.path.join(cdir, "composites.csv"), "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["index", "first_skip", "last_skip", "n_skips", "noise_image_e"])
            wr.writerows([(j, a, b, l, f"{sj:.5f}") for j, a, b, l, sj in ctab])
        nz = np.array([t[4] for t in ctab])
        print(f"composites: {NC} images of {LC} skips in {cdir}; noise {nz.mean():.4f} e- "
              f"(min {nz.min():.4f}, max {nz.max():.4f})")
        del comp

    # ---- table
    csv_path = os.path.join(args.outdir, f"{name}_noise_vs_skips.csv")
    with open(csv_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["N_skips", "noise_image_e", "noise_within_pixel_e", "white_sigma1_over_sqrtN_e"])
        for i, k in enumerate(ns):
            wb = s_blk[nb.index(k)] if k in nb else ""
            wr.writerow([k, f"{s_img[i]:.5f}", f"{wb:.5f}" if wb != "" else "", f"{s1 / np.sqrt(k):.5f}"])

    # ---- average image (used skips) as FITS
    from astropy.io import fits
    hdu = fits.PrimaryHDU(img[-1].astype(np.float32))
    hdu.header.update(NSKIPS=M, SKIPFRST=(K, "skips dropped per pixel"), NSAMP=n, GAIN=(round(g, 5), "ADU/e-"),
                      NROWSF=(rows, "complete rows decoded"), SRCFILE=os.path.basename(args.infile)[:68])
    hdu.writeto(os.path.join(args.outdir, f"{name}_avg.fits"), overwrite=True)

    # ---- plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
    a = ax[0]
    a.loglog(ns, s_img, "o-", ms=3.5, lw=1.2, color="#1f5fbf", label="image (N-skip average, one pedestal per row)")
    a.loglog(nb, s_blk, "s-", ms=3, lw=1.0, color="#d1495b", label="within pixel (Allan deviation, blocks of N skips)")
    nn = np.array([1, M])
    a.loglog(nn, s1 / np.sqrt(nn), "--", color="0.5", lw=1, label=rf"$\sigma_1/\sqrt{{N}}$, $\sigma_1$ = {s1:.2f} e$^-$")
    a.loglog(nn, white / np.sqrt(nn), ":", color="0.3", lw=1, label=f"white floor/$\\sqrt{{N}}$ ({white:.2f} e$^-$ per skip)")
    a.annotate(f"{s_img[-1]:.3f} e$^-$ at {ns[-1]} skips", (ns[-1], s_img[-1]), textcoords="offset points",
               xytext=(-120, -22), fontsize=9, arrowprops=dict(arrowstyle="-", color="0.4"))
    a.set_xlabel("number of skips averaged N")
    a.set_ylabel("noise [e$^-$]")
    a.set_title(f"{name}\n{rows} rows x {ncols} cols, first {K} skips dropped, gain {g:.3f} ADU/e$^-$", fontsize=9)
    a.grid(True, which="both", alpha=0.3)
    a.legend(fontsize=8)
    b = ax[1]
    b.loglog(f[1:], psd[1:] / g ** 2, lw=0.5, color="#1f5fbf")
    b.set_xlabel("frequency [1 / skip]")
    b.set_ylabel("power of the skip sequence [e$^{-2}$ per bin]")
    b.set_title("noise spectrum over the skips of a pixel", fontsize=9)
    b.grid(True, which="both", alpha=0.3)
    c = ax[2]
    k_show = min(nskip, 200)
    c.plot(np.arange(k_show), prof[:k_show] / g, lw=1, color="#d1495b")
    c.axvline(K - 0.5, color="0.4", ls="--", lw=1)
    c.set_yscale("symlog", linthresh=1)
    c.set_xlabel("skip index in the pixel")
    c.set_ylabel("median offset from the pixel mean [e$^-$]")
    c.set_title(f"start-of-pixel transient (left of dashed line: dropped)", fontsize=9)
    c.grid(True, alpha=0.3)
    plt.tight_layout()
    png = os.path.join(args.outdir, f"{name}_noise_vs_skips.png")
    plt.savefig(png, dpi=130)
    print(f"single skip {s1:.2f} e- (white floor {white:.2f} e-); {ns[-1]} skips: {s_img[-1]:.3f} e- (image)")
    print(f"wrote {png}\n      {csv_path}\n      {os.path.join(args.outdir, name + '_avg.fits')}")


if __name__ == "__main__":
    main()
