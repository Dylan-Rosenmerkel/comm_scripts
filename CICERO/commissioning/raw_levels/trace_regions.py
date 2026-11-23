# copied from ccd_opt_ml/src/trace_regions.py (commit 3d7855e) by utils/sync_from_ccd_opt_ml.py: edit it there, not here
"""
trace_regions.py -- split a decoded trace into its regions and stack the repeats.

A trace of a skipper-CCD readout consists of (see ``decode_trace.py`` for the file format)

    vertical transfers   NPBIN repeats of one vertical transfer, once per row
    horizontal transfers NSBIN repeats of one horizontal transfer, before every pixel
    skip region          NDCM skip cycles (pedestal window, signal window) and one extra
                         pedestal window, per pixel

Everything is found from the data, for any geometry (rows, columns, skips, binning) and any
clock timing:

* **windows**: the pedestal (RD, flag bit 0) and signal (RU, flag bit 1) window flags;
* **pixels**: from the ORDER of the windows. A pixel is N (pedestal, signal) pairs followed by
  one extra pedestal, so a pixel ends at a pedestal window followed by another pedestal window;
* **skip cycle**: the spacing of the pedestal windows inside a pixel (its mean is the exact,
  non-integer period; the cycles are cut with its rounded value);
* **horizontal transfers**: rising edges of flag bit 6 in the gap before each pixel;
* **vertical transfers**: rising edges of flag bit 7 (bit 5 on module 101 too, but not on
  module 102, whose bcf uses Vtransfer21);
* **row starts**: pixel ``p`` starts a row if ``p % NCOL == 0`` (from the ``.meta``); without the
  ``.meta``, if vertical-transfer edges precede it (pixel 0 always starts a row).

The ``.meta`` geometry (``trace_geometry.py``) is used to cross-check the measured values
(mismatches go to ``warnings``) and as a fallback when the data cannot show a value (e.g. the
length of a horizontal transfer when there is only one per pixel).  Flag bits 6 and 7 are the
sequencer's clock outputs for ``reference_v5(_pumping).seq``; their meaning was inferred from
real traces.  Nothing is averaged: every repeat is kept as its own row.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from decode_trace import DecodedTrace, decode_trace
from trace_geometry import TraceGeometry

_H_BIT, _V_BIT, _V_BIT_ALT = 6, 7, 5


@dataclass
class TraceRegions:
    """The regions of one trace.  All waveforms are float ADU, one row per repeat."""

    n_ped: int                       # samples in one pedestal window
    sig_start: int                   # sample (within a skip cycle) where the signal window starts
    period: int                      # samples cut for one skip cycle (rounded period)
    skip: list[np.ndarray]           # per pixel: (n_skip, period), cycle starts at its pedestal window
    cds: list[np.ndarray]            # per pixel: (n_skip,) pedestal - signal from the inline L1 data
    h: list[np.ndarray]              # per pixel: (n_transfer, h_len) horizontal transfers
    v: np.ndarray | None             # (n_transfer, v_len) vertical transfers of one block
    h_period: float                  # samples between horizontal transfers (not an integer)
    v_period: float | None           # samples between vertical transfers
    # positions in the decoded sample stream (for plots): (start, end) sample numbers
    skip_bounds: list = field(default_factory=list)   # skip region of every pixel
    h_block: int = 0                                  # typical gap (horizontal transfers only) before a pixel
    v_bounds: tuple | None = None                     # the vertical block whose transfers are in ``v``
    # added for arbitrary geometries
    period_exact: float = float("nan")                # mean spacing of the pedestal windows (samples)
    ped_starts: list = field(default_factory=list)    # per pixel: absolute starts of ALL its pedestal windows
    row_starts: list = field(default_factory=list)    # pixels that start a row
    geometry: TraceGeometry = field(default_factory=TraceGeometry)
    warnings: list = field(default_factory=list)

    @property
    def n_pixels(self) -> int:
        return len(self.skip)

    @property
    def n_skips(self) -> int:
        return int(min((len(c) for c in self.skip), default=0))


def _rising(flags: np.ndarray, bit: int, min_high: int = 10) -> np.ndarray:
    """
    Sample numbers where flag ``bit`` goes from 0 to 1 and then stays 1 for at least ``min_high``
    samples.  Real traces contain isolated short flag glitches (e.g. a bit-7 pulse about 158
    samples after a pixel's last window), which must not count as transfers.
    """
    x = ((flags >> bit) & 1).astype(np.int8)
    change = np.diff(np.r_[0, x, 0])
    up, down = np.flatnonzero(change == 1), np.flatnonzero(change == -1)
    return up[(down - up) >= min_high]


def _stack(values: np.ndarray, edges: np.ndarray, length: int, last_allowed_end: int) -> np.ndarray:
    """Stack ``values[e:e+length]`` for every start ``e`` whose segment ends before ``last_allowed_end``."""
    if length <= 0:
        return np.empty((0, 0))
    rows = [values[e:e + length] for e in edges if e + length <= last_allowed_end]
    return np.array(rows, dtype=float) if rows else np.empty((0, length))


def _pixel_windows(trace: DecodedTrace) -> list[dict]:
    """
    Group the pedestal and signal windows into pixels by their order.

    Returns one dict per pixel: ``ped`` and ``sig`` = arrays of (start, end) sample numbers.
    """
    ped_s, ped_e = trace.window_slices()
    sig_s, sig_e = trace.signal_window_slices()
    events = sorted([(s, e, False) for s, e in zip(ped_s, ped_e)] + [(s, e, True) for s, e in zip(sig_s, sig_e)])
    pixels, current = [], {"ped": [], "sig": []}
    for i, (s, e, is_sig) in enumerate(events):
        current["sig" if is_sig else "ped"].append((s, e))
        next_is_ped = i + 1 == len(events) or not events[i + 1][2]
        if not is_sig and next_is_ped:                    # extra pedestal: the pixel ends here
            pixels.append({k: np.array(v, dtype=np.int64).reshape(-1, 2) for k, v in current.items()})
            current = {"ped": [], "sig": []}
    return pixels


def _median_diff(groups: list[np.ndarray], max_ratio: float = 1.5) -> float:
    """Median spacing of consecutive edges inside each group (gaps above max_ratio x median dropped)."""
    diffs = np.concatenate([np.diff(g) for g in groups if len(g) > 1] or [np.empty(0)])
    if not len(diffs):
        return float("nan")
    med = np.median(diffs)
    return float(np.median(diffs[diffs <= max_ratio * med]))


def split_regions(trace: DecodedTrace, meta: dict | None = None) -> TraceRegions:
    """Split ``trace`` into skip cycles, horizontal transfers and vertical transfers."""
    values = trace.values.astype(float)
    flags = trace.flags.astype(np.int64)
    geo = TraceGeometry.from_meta(meta)
    warnings: list[str] = []

    # ---- pixels and skip cycles ----------------------------------------------------------------
    pixels = _pixel_windows(trace)
    if not pixels:
        raise ValueError("no pedestal windows found")
    ped_starts = [px["ped"][:, 0] for px in pixels]
    spacing = [np.diff(s) for s in ped_starts if len(s) > 1]
    all_spacing = np.concatenate(spacing)
    period = int(np.median(all_spacing))
    period_exact = float(np.mean(all_spacing[np.abs(all_spacing - period) <= 1])) if len(all_spacing) else float(period)
    n_ped = int(np.median(np.concatenate([px["ped"][:, 1] - px["ped"][:, 0] for px in pixels])))
    offsets = [px["sig"][:, 0] - px["ped"][:len(px["sig"]), 0] for px in pixels if len(px["sig"])]
    sig_start = int(np.median(np.concatenate(offsets))) if offsets else n_ped

    skip, bounds = [], []
    for starts in ped_starts:
        cycle_starts = starts[:-1]                         # the last pedestal is the extra one
        cycle_starts = cycle_starts[cycle_starts + period <= len(values)]
        skip.append(np.stack([values[s:s + period] for s in cycle_starts]) if len(cycle_starts)
                    else np.empty((0, period)))
        bounds.append((int(starts[0]), min(int(starts[-1]) + period, len(values))))

    l1 = trace.skips()
    cds = [l1["cds"][l1["pixel"] == p] for p in range(len(pixels))]
    if len(np.unique(l1["pixel"])) != len(pixels):
        warnings.append(f"{len(pixels)} pixels in the window flags but {len(np.unique(l1['pixel']))} in the inline records")

    # ---- gaps before each pixel: transfers -------------------------------------------------------
    gap_start = [0] + [int(px["ped"][-1, 1]) for px in pixels[:-1]]   # end of the previous extra pedestal
    gap_end = [int(px["ped"][0, 0]) for px in pixels]
    e_h = _rising(flags, _H_BIT)
    e_v = _rising(flags, _V_BIT)
    if len(e_v) < 2:
        e_v = _rising(flags, _V_BIT_ALT)
    h_edges = [e_h[(e_h >= a) & (e_h < b)] for a, b in zip(gap_start, gap_end)]
    v_edges = [e_v[(e_v >= a) & (e_v < b)] for a, b in zip(gap_start, gap_end)]

    # row starts: from NCOL if known, else from vertical-transfer edges in the gap
    if geo.ncol:
        row_starts = [p for p in range(len(pixels)) if p % geo.ncol == 0]
    else:
        row_starts = [0] + [p for p in range(1, len(pixels)) if len(v_edges[p])]

    h_period = _median_diff(h_edges)
    if not np.isfinite(h_period) and geo.h_transfer:
        h_period = geo.h_transfer                          # one transfer per pixel: use the .meta
    h = []
    for p in range(len(pixels)):
        if np.isfinite(h_period):
            h.append(_stack(values, h_edges[p], int(round(h_period)), gap_end[p]))
        else:
            h.append(np.empty((0, 0)))
    inner = [gap_end[p] - gap_start[p] for p in range(1, len(pixels)) if p not in row_starts]
    h_block = int(np.median(inner)) if inner else 0

    # vertical transfers: the block before the first row start after pixel 0, else the leading block
    v, v_period, v_bounds = None, None, None
    v_period_measured = _median_diff([v_edges[p] for p in row_starts])
    candidates = [p for p in row_starts if p > 0] + [0]
    for p in candidates:
        edges = v_edges[p]
        period_v = v_period_measured if np.isfinite(v_period_measured) else (geo.v_transfer or float("nan"))
        if len(edges) and np.isfinite(period_v):
            hi = int(h_edges[p][0]) if len(h_edges[p]) else gap_end[p]
            stacked = _stack(values, edges, int(round(period_v)), hi)
            if len(stacked) >= 3:
                stacked = stacked[1:]                      # the first transfer follows the skip region
            if len(stacked):
                v, v_period, v_bounds = stacked, float(period_v), (int(gap_start[p]), hi)
                break

    regions = TraceRegions(n_ped, sig_start, period, skip, cds, h, v, h_period, v_period,
                           bounds, h_block, v_bounds, period_exact, ped_starts, row_starts, geo, warnings)
    _cross_check(regions, h_edges, v_edges)
    return regions


def _cross_check(r: TraceRegions, h_edges: list, v_edges: list) -> None:
    """Compare what was found with what the .meta says; add a warning for every mismatch."""
    g, w = r.geometry, r.warnings
    if g.n_pixels is not None and g.n_pixels != r.n_pixels:
        w.append(f"{r.n_pixels} pixels found, .meta says {g.nrow} x {g.ncol} = {g.n_pixels}")
    if g.ndcm is not None and any(len(c) != g.ndcm for c in r.skip):
        w.append(f"skips per pixel {sorted({len(c) for c in r.skip})}, .meta says NDCM = {g.ndcm}")
    if g.nsbin is not None:
        counts = sorted({len(e) for e in h_edges})
        # every pixel: NSBIN transfers; the first pixel of a row also gets the NSBIN transfers of each
        # skipped prescan pixel (NCOLPRE), and from the second row on those of the previous row's
        # skipped post-scan pixels (NCOLPOST)
        allowed = {g.nsbin, g.nsbin * (g.ncolpre + 1), g.nsbin * (g.ncolpre + g.ncolpost + 1)}
        if not set(counts) <= allowed:
            w.append(f"horizontal-transfer edges per pixel {counts}, .meta says NSBIN = {g.nsbin}"
                     + (f", NCOLPRE = {g.ncolpre}" if g.ncolpre else "")
                     + (f", NCOLPOST = {g.ncolpost}" if g.ncolpost else ""))
    if g.timing_known:
        checks = (("skip cycle", r.period_exact, g.cycle, 1.0), ("window length", r.n_ped, g.n_window, 2.0),
                  ("signal-window offset", r.sig_start, g.sig_offset, 2.0),
                  ("horizontal transfer", r.h_period, g.h_transfer, 2.0),
                  ("vertical transfer", r.v_period, g.v_transfer, 2.0))
        if g.nsbin == 1 and g.dump and r.h_period is not None and np.isfinite(r.h_period):
            # without binning, consecutive transfers (prescan pixels) are separated by the dump pulse
            options = (g.h_transfer, g.h_transfer + g.dump)
            checks = tuple((n, f, min(options, key=lambda o: abs(o - f)) if n == "horizontal transfer" else e, t)
                           for n, f, e, t in checks)
        for name, found, expected, tolerance in checks:
            if found is not None and np.isfinite(found) and abs(found - expected) > tolerance:
                w.append(f"{name}: {found:.1f} samples found, {expected:.1f} expected from the .meta")


def load_regions(path: str, meta: dict | None = None) -> TraceRegions:
    """Decode ``path`` and split it into regions (reads the ``.meta`` next to it if ``meta`` is None)."""
    import json
    from pathlib import Path
    if meta is None and Path(path).with_suffix(".meta").exists():
        meta = json.loads(Path(path).with_suffix(".meta").read_text())
    return split_regions(decode_trace(path), meta)
