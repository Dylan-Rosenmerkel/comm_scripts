# copied from ccd_opt_ml/src/plot_trace.py (commit 3d7855e) by utils/sync_from_ccd_opt_ml.py: edit it there, not here
"""
plot_trace.py -- make an overview figure of one decoded trace.

Four panels, all from :func:`decode_trace.decode_trace`:

    A  the raw video signal for three consecutive skip cycles, with the pedestal (RD) and
       signal (RU) integration windows shaded
    B  pedestal and signal window means for every skip of one pixel (inline L1 data), and
       their difference (CDS): shows the baseline droop and the first-skip transient
    C  all skip cycles of one pixel folded on top of each other (mean +- 1 sigma), after
       removing each cycle's own baseline (the pedestal-window mean) so that the slow
       baseline walk of panel B does not hide the cycle-to-cycle variation
    D  noise spectrum of what is left after subtracting each cycle's baseline and the mean
       cycle shape (the deterministic, repeating part gives only a comb at 1/32.6 us)

Usage:  python plot_trace.py TRACE_FILE OUT.png [--pixel 0]
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from decode_trace import SAMPLE_RATE_HZ, decode_trace


def averaged_psd(values: np.ndarray, segment: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Welch-style PSD in ADU/sqrt(Hz): Hann-windowed segments, mean removed, averaged."""
    n_seg = len(values) // segment
    segs = values[:n_seg * segment].reshape(n_seg, segment).astype(float)
    segs -= segs.mean(axis=1, keepdims=True)
    window = np.hanning(segment)
    power = np.abs(np.fft.rfft(segs * window, axis=1)) ** 2
    psd = power.mean(axis=0) / (SAMPLE_RATE_HZ * (window ** 2).sum())
    psd[1:-1] *= 2
    return np.fft.rfftfreq(segment, 1 / SAMPLE_RATE_HZ), np.sqrt(psd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace")
    parser.add_argument("out")
    parser.add_argument("--pixel", type=int, default=0, help="pixel number shown in panels B and C")
    args = parser.parse_args()

    trace = decode_trace(args.trace)
    cycles, period, cycle_pixel = trace.skip_cycles()
    skips = trace.skips()
    name = args.trace.split("/")[-1]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(f"{name}\n{len(trace.values)} samples at {SAMPLE_RATE_HZ / 1e6:.0f} MHz, "
                 f"{len(cycles)} skip cycles of {period} samples ({period / 15:.1f} us)", fontsize=11)

    # ---- A: three raw skip cycles ------------------------------------------------------
    ax = axes[0, 0]
    first = np.flatnonzero(cycle_pixel == args.pixel)[10]            # a cycle in the middle of pixel
    sl = slice(trace.window_slices()[0][first], trace.window_slices()[0][first] + 3 * period)
    t = trace.time_us[sl] - trace.time_us[sl][0]
    ax.plot(t, trace.values[sl], lw=0.7, color="0.25")
    ax.fill_between(t, *ax.get_ylim(), where=trace.rd[sl], color="tab:red", alpha=0.18, label="pedestal window (RD)")
    ax.fill_between(t, *ax.get_ylim(), where=trace.ru[sl], color="tab:green", alpha=0.18, label="signal window (RU)")
    ax.set(title="A  raw video signal, 3 skip cycles", xlabel="time (us)", ylabel="ADC value (ADU)")
    ax.legend(loc="best", fontsize=8)

    # ---- B: per-skip window means of one pixel (L1) ----------------------------------------
    ax = axes[0, 1]
    sel = skips["pixel"] == args.pixel
    k = skips["skip"][sel]
    ax.plot(k, skips["pedestal"][sel], ".", ms=3, color="tab:red", label="pedestal mean")
    ax.plot(k, skips["signal"][sel], ".", ms=3, color="tab:green", label="signal mean")
    ax.set(title=f"B  pixel {args.pixel}: window means per skip (inline L1)", xlabel="skip number",
           ylabel="window mean (ADU)")
    ax2 = ax.twinx()
    ax2.plot(k, skips["cds"][sel], lw=0.7, color="tab:blue", label="CDS = pedestal - signal")
    ax2.set_ylabel("CDS (ADU)", color="tab:blue")
    ax.legend(loc="lower left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    # ---- C: folded skip cycle -----------------------------------------------------------------
    ax = axes[1, 0]
    keep = (cycle_pixel == args.pixel)
    folded = cycles[keep][:-1].astype(float)                         # drop the extra pedestal-only cycle
    folded -= folded[:, :150].mean(axis=1, keepdims=True)            # remove each cycle's baseline
    tt = np.arange(period) / 15.0
    mean, std = folded.mean(axis=0), folded.std(axis=0)
    ax.fill_between(tt, mean - std, mean + std, color="tab:blue", alpha=0.25, label="+-1 sigma over skips")
    ax.plot(tt, mean, color="tab:blue", lw=0.9, label=f"mean of {len(folded)} skips")
    lo, hi = ax.get_ylim()
    rd, ru = trace.rd, trace.ru
    start = trace.window_slices()[0][np.flatnonzero(cycle_pixel == args.pixel)[10]]
    ax.fill_between(tt, lo, hi, where=rd[start:start + period], color="tab:red", alpha=0.12)
    ax.fill_between(tt, lo, hi, where=ru[start:start + period], color="tab:green", alpha=0.12)
    ax.set_ylim(lo, hi)
    ax.set(title=f"C  pixel {args.pixel}: skip cycles folded (baseline removed)", xlabel="time in cycle (us)",
           ylabel="ADC value - cycle baseline (ADU)")
    ax.legend(loc="best", fontsize=8)

    # ---- D: noise spectrum --------------------------------------------------------------------
    ax = axes[1, 1]
    residual = (folded - folded.mean(axis=0)).ravel()                # noise left after removing the shape
    freq, amp = averaged_psd(residual, segment=period * 8)
    ax.loglog(freq[1:], amp[1:], lw=0.8, color="tab:purple")
    ax.set(title="D  noise spectrum (baseline and mean cycle shape removed)", xlabel="frequency (Hz)",
           ylabel="ADU / sqrt(Hz)")
    ax.grid(True, which="both", alpha=0.3)
    ax.text(0.03, 0.04, f"windows verified: inline vs raw sums differ by {trace.check_l1()}",
            transform=ax.transAxes, fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
