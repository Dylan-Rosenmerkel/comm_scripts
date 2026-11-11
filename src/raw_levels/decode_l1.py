# copied from ccd_opt_ml/src/decode_l1.py (commit 06ad4cd) by utils/sync_from_ccd_opt_ml.py: edit it there, not here
"""
decode_l1.py -- decode a level-1 skipper image (``*.RJ45ADCDATA<ch>`` written with acq_level 1): the pedestal and
signal window sums of EVERY skip of every pixel.  numpy only.

Level 1 (acmpy filter level 1 on every channel, FILTER 0x55) keeps the inline per-skip sums that a trace also holds,
but no raw ADC samples, and records all four channels at once.  It is read out by the image's own sequencer routine
(``AcquireImage``), so a small level-1 image (a "strip") sees exactly the image's clocking: its prescan, its register
clear, its skips.

File layout (big-endian 32-bit words).  The same layout as ``comm_scripts/utils/skip_noise_L1.py`` (which was checked
on img_200x1500x4000_..._L1_101_..._15); checked here on the 1x1 level-1 image of 2026-10-06 (100x1000x3000, a file
cut after 80 pixels: every layout check passes)::

    START 0xD3544152, BUSY 0xC2555359, one header word, then per row:
    "LINE" 0x4C494E45, then per pixel 4 * NSKIP + 2 + 13 words:
        NSKIP x [pedestal header, pedestal sum, signal header, signal sum]
            header: top nibble 5, bits 19..0 = samples summed n (150 = 10 us), flag 0x80 = signal window
            sum   : 31-bit signed sum of the n ADC samples
        one more [pedestal header, pedestal sum]
        a 13-word pixel trailer: [0x7.. header], the sum over the skips of (signal - pedestal) as a 62-bit number
        (2 x 31 bits), further firmware sums, and "EPIX" 0x45504958 as the last word

Pixel value of skip k: (pedestal_k - signal_k) / n, in ADU (charge positive), as in skip_noise_L1.py.  Every complete
pixel is checked (skip headers, EPIX, per-skip sums against the trailer's sum); a file cut short is decoded up to its
last complete pixel.  Binned images (e.g. 20x20 of 2026-10-06): the same pixel records (ncol = binned columns), but
each row is preceded by one LINE per vertical shift (NPBIN of them); rows are found after each run of LINE markers.

    from decode_l1 import read_l1
    strip = read_l1("l1new_2x1000x500_1x1_L1_101_....RJ45ADCDATA0", ncol=1000, nskip=500)
    strip.values          # (pixels, skips) ADU, pixels in readout order (row-major)
    strip.rows            # row of every pixel
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LINE, EPIX, START = 0x4C494E45, 0x45504958, 0xD3544152
TRAILER = 13


def s31(x: np.ndarray) -> np.ndarray:
    """31-bit two's complement (bit 31 of the data words is never set)."""
    x = np.asarray(x, np.int64) & 0x7FFFFFFF
    return x - ((x >> 30) << 31)


@dataclass
class L1Image:
    values: np.ndarray       # (pixels, skips) per-skip pixel value (pedestal - signal) / n, ADU
    rows: np.ndarray         # (pixels,) row of each pixel, readout order
    cols: np.ndarray         # (pixels,) column of each pixel within its row
    n_samples: int           # samples per window (n)
    ncol: int
    nskip: int
    complete_rows: int       # rows decoded completely
    problems: list[str]      # layout problems found (empty = all checks passed)

    @property
    def n_pixels(self) -> int:
        return len(self.values)


def geometry_from_name(path: str | Path) -> tuple[int, int, int] | None:
    """(rows, cols, skips) from a level-1 file name such as ``img_2x1000x500_1x1_L1_101_...`` (any binning; cols =
    binned columns, as NCOL)."""
    m = re.search(r"_(\d+)x(\d+)x(\d+)_\d+x\d+_L1_", Path(path).name)
    return tuple(int(v) for v in m.groups()) if m else None


def read_l1(path: str | Path, ncol: int | None = None, nskip: int | None = None) -> L1Image:
    """Decode a 1x1 level-1 image (see the module docstring); geometry from the arguments or the file name."""
    if ncol is None or nskip is None:
        geo = geometry_from_name(path)
        if geo is None:
            raise ValueError(f"no geometry for {path}: give ncol and nskip")
        ncol, nskip = geo[1], geo[2]
    w = np.fromfile(path, dtype=">u4").astype(np.int64)
    pix = 4 * nskip + 2 + TRAILER
    first = np.flatnonzero(w[:256] == LINE)
    if not len(first):
        raise ValueError(f"{path}: no LINE marker at the start: not a level-1 image")
    problems: list[str] = []
    values, rows, cols = [], [], []
    n_samples, complete_rows, r = 0, 0, 0
    # A row starts after its LINE markers: one per vertical shift, so NPBIN of them in a vertically binned image
    # (the first NPBIN - 1 without pixels), then ncol pixel records of the layout above (binned or not).
    o = int(first[0])
    while True:
        if o >= len(w) or w[o] != LINE:
            if o < len(w) - 1:
                problems.append(f"row {r}: no LINE marker (word {o}); stopped")
            break
        while o < len(w) and w[o] == LINE:
            o += 1
        k = min(ncol, (len(w) - o) // pix)
        if k <= 0:
            break
        p = w[o:o + k * pix].reshape(k, pix)
        o += k * pix
        hdr = p[:, 0:4 * nskip:2]
        counts = hdr & 0xFFFFF                   # samples in each window: pedestal (even), signal (odd columns)
        n = int(np.median(counts))
        # Windows may alternate between N and N+1 samples when delayInt is not a whole number of ADC samples
        # (decode_trace.window_samples); allow that, flag anything further off.
        good = (p[:, -1] == EPIX) & ((hdr >> 28) == 5).all(1) & (np.abs(counts - n) <= 1).all(1)
        ped, sig = s31(p[:, 1:4 * nskip:4]), s31(p[:, 3:4 * nskip:4])
        tot = s31(p[:, 4 * nskip + 3]) * (1 << 31) + (p[:, 4 * nskip + 4] & 0x7FFFFFFF)
        good &= tot == (sig - ped).sum(axis=1)
        if not good.all():
            problems.append(f"row {r}: {int((~good).sum())} of {k} pixels fail the layout checks")
        # window MEANS (each sum over its own sample count): with equal counts this is (ped - sig) / n as before;
        # with N / N+1 windows it avoids the one-baseline-sample offset that a shared n would add
        n_ped, n_sig = np.maximum(counts[:, 0::2], 1), np.maximum(counts[:, 1::2], 1)
        values.append((ped[good] / n_ped[good] - sig[good] / n_sig[good]))
        rows.append(np.full(int(good.sum()), r))
        cols.append(np.flatnonzero(good))
        n_samples = n
        if k < ncol:
            break                                # the file ends inside this row
        complete_rows += 1
        r += 1
    if not values:
        raise ValueError(f"{path}: no complete pixel")
    return L1Image(values=np.vstack(values), rows=np.concatenate(rows), cols=np.concatenate(cols), n_samples=n_samples,
                   ncol=ncol, nskip=nskip, complete_rows=complete_rows, problems=problems)


def write_l1(path: str | Path, values: np.ndarray, ncol: int, n: int = 150, baseline: float = 1.0e6) -> None:
    """Write per-skip pixel values (pixels x skips, ADU, rows of ``ncol`` pixels) as a level-1 file of the layout
    above (for tests and rehearsals; the trailer words that read_l1 does not check are zero)."""
    values = np.asarray(values, float)
    npx, nskip = values.shape
    words = [START, 0xC2555359, 0xE0B503E8]
    for r in range(npx // ncol):
        words.append(LINE)
        for c in range(ncol):
            v = values[r * ncol + c]
            ped = np.full(nskip, baseline * n, dtype=np.int64)
            sig = ped - np.round(v * n).astype(np.int64)
            px = np.zeros(4 * nskip + 2 + TRAILER, dtype=np.int64)
            px[0:4 * nskip:4] = 0x50000000 | n
            px[1:4 * nskip:4] = ped & 0x7FFFFFFF
            px[2:4 * nskip:4] = 0x58000000 | n
            px[3:4 * nskip:4] = sig & 0x7FFFFFFF
            px[4 * nskip] = 0x50000000 | n
            px[4 * nskip + 1] = int(baseline * n) & 0x7FFFFFFF
            tot = int((sig - ped).sum())
            px[4 * nskip + 2] = 0x70000000 | (n * nskip & 0xFFFFF)
            px[4 * nskip + 3] = (tot >> 31) & 0x7FFFFFFF
            px[4 * nskip + 4] = tot & 0x7FFFFFFF
            px[-1] = EPIX
            words.extend(px.tolist())
    np.asarray(words, dtype=np.uint64).astype(">u4").tofile(path)
