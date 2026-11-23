# copied from ccd_opt_ml/src/decode_trace.py (commit 3d7855e) by utils/sync_from_ccd_opt_ml.py: edit it there, not here
"""
decode_trace.py -- decode a raw ACM trace file (``*.RJ45ADCDATA<ch>``) with numpy only.

``acmpy.Decoder`` is a compiled Linux library, so this module re-implements the part of
the format that the ML pipeline needs.  The layout below was worked out from real traces
and CHECKED against the data itself: the inline per-skip sums written by the firmware
equal the sums of the raw samples inside the flagged windows (exact match on the sample
file), and the windows are exactly ``delayInt`` long (150 samples = 10 us at 15 MHz).

File layout (big-endian 32-bit words)
-------------------------------------
    keyword words    bit 31 set:   START 0xD3544152, BUSY 0xC2555359, a header word
                                   0xE0B4_03E8, ... END 0xA0454E44
    raw samples      top nibble 2: bits 19..0  = signed 20-bit ADC value
                                   bit 20 (flag bit 0) = RD, pedestal window
                                   bit 21 (flag bit 1) = RU, signal window
                                   other flag bits: sequencer states, not decoded here
    L1 "skip" words top nibble 5: header word, bits 19..0 = number of samples n (150),
                                   flag bit 7 (0x80) = signal window (else pedestal),
                                   flag low nibble = pixel number in the trace (4 bits
                                   only, so it repeats after 16 pixels and is NOT used
                                   to group the records; see ``pixel_segments``).
                                   The NEXT word is the sum of the n samples
                                   (31-bit signed; value / n = the window mean).
    other words      everything else (end-of-pixel records starting 0x70k..., a few
                     markers).  Kept, undecoded, in ``DecodedTrace.other``.

The raw samples are interleaved with the L1 words in the file; after decoding they are
separated, and the "sample index" counts raw samples only (15 MHz -> 66.7 ns each).

Usage
-----
    from decode_trace import decode_trace
    trace = decode_trace("trace_...ch0_...RJ45ADCDATA0")
    trace.values, trace.rd, trace.ru        # raw samples and window flags
    trace.skips()                           # per-skip pedestal / signal / CDS (from L1)
    trace.skip_cycles()                     # raw samples folded into one row per skip cycle

    python decode_trace.py trace_...RJ45ADCDATA0 [--csv out.csv]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE_HZ = 15e6          # ADC sample rate (also assumed by commant)
SEQUENCER_CLOCK_HZ = 100e6     # sequencer clock: delays such as delayInt are in these cycles
SAMPLES_PER_CYCLE = SAMPLE_RATE_HZ / SEQUENCER_CLOCK_HZ   # 0.15: delayInt 1000 -> 150 samples


def window_samples(delay_int_cycles: float) -> float:
    """
    ADC samples in one CDS window of ``delay_int_cycles`` sequencer cycles (1000 -> 150).

    When this is not a whole number (delayInt not a multiple of 20 cycles), the windows
    alternate between N and N+1 samples.  The traces handle that (window means), but the
    on-board image CDS is built from window SUMS, so a pixel's pedestal and signal windows
    of different lengths shift its value by a whole baseline sample: such images can show
    discrete offset levels ~10 ADU apart and no resolved electron peaks (v2 night,
    2026-11-05: 23 of 25 rejected WADERs fits).
    """
    return float(delay_int_cycles) * SAMPLES_PER_CYCLE


def fractional_window(delay_int_cycles: float) -> bool:
    """True if a delayInt of this many cycles gives a non-integer number of samples per window."""
    n = window_samples(delay_int_cycles)
    return abs(n - round(n)) > 1e-6
_KEYWORD_BIT = 1 << 31
_MAX_WINDOW = 5000             # a skip-window header carries n <= this many samples


@dataclass
class DecodedTrace:
    """Result of :func:`decode_trace`.  All arrays are numpy arrays."""

    path: str
    keywords: np.ndarray        # keyword words (START, BUSY, header, END)
    values: np.ndarray          # raw ADC samples, int32 (ADU)
    rd: np.ndarray              # bool, sample inside a pedestal window
    ru: np.ndarray              # bool, sample inside a signal window
    flags: np.ndarray           # uint16, all 12 flag bits of each raw sample
    l1_pixel: np.ndarray        # pixel number of every inline L1 record
    l1_is_signal: np.ndarray    # bool, True = signal window, False = pedestal
    l1_n: np.ndarray            # samples summed in each record
    l1_sum: np.ndarray          # int64 sum of those samples
    other_index: np.ndarray     # word positions of undecoded words
    other: np.ndarray           # the undecoded words (uint32)

    # -- basic properties ---------------------------------------------------
    @property
    def time_us(self) -> np.ndarray:
        """Time of every raw sample in microseconds (sample number / 15 MHz)."""
        return np.arange(len(self.values)) / (SAMPLE_RATE_HZ / 1e6)

    # -- L1 stream -------------------------------------------------------------
    def pixel_segments(self) -> list[tuple[int, int]]:
        """
        ``(first, end)`` index ranges of the inline L1 records of every pixel, in order.

        Found from the ORDER of the records, which works for any number of pixels and skips:
        a pixel is N (pedestal, signal) pairs followed by ONE extra pedestal (``ReadoutPedestal``),
        and the next pixel starts again with a pedestal.  So a pixel ends at a pedestal record
        that is followed by another pedestal record (or by the end of the data).
        """
        sig = self.l1_is_signal
        ends = [i + 1 for i in range(len(sig)) if not sig[i] and (i + 1 == len(sig) or not sig[i + 1])]
        return list(zip([0] + ends[:-1], ends))

    def skips(self) -> dict[str, np.ndarray]:
        """
        Per-skip pedestal / signal means and CDS, from the inline L1 records.

        Every pixel is read as ``N`` pedestal/signal pairs followed by ONE extra pedestal
        window (``ReadoutPedestal`` in the sequencer).  Pairs are therefore formed pixel by
        pixel.  Returns arrays with one entry per skip, in acquisition order:
        ``pixel`` (the pixel's position in the readout: 0, 1, 2, ...), ``skip`` (index within the
        pixel), ``pedestal`` and ``signal`` (window means in ADU) and ``cds = pedestal - signal``
        (the sign convention of ``commant``); plus ``final_pedestal``, the extra pedestal of each
        pixel, indexed by ``pixel_ids``.
        """
        mean = self.l1_sum / np.maximum(self.l1_n, 1)
        out = {k: [] for k in ("pixel", "skip", "pedestal", "signal")}
        final, pixel_ids = [], []
        for index, (a, b) in enumerate(self.pixel_segments()):     # pixel = position in the readout
            is_sig = self.l1_is_signal[a:b]
            peds, sigs = mean[a:b][~is_sig], mean[a:b][is_sig]
            n = len(sigs)
            out["pixel"].append(np.full(n, index))
            out["skip"].append(np.arange(n))
            out["pedestal"].append(peds[:n])
            out["signal"].append(sigs)
            if len(peds) > n:
                final.append(peds[n])
                pixel_ids.append(index)
        result = {k: np.concatenate(v) if v else np.empty(0) for k, v in out.items()}
        result["cds"] = result["pedestal"] - result["signal"]
        result["final_pedestal"] = np.array(final)
        result["pixel_ids"] = np.array(pixel_ids)
        return result

    # -- raw samples ---------------------------------------------------------------
    def window_slices(self) -> tuple[np.ndarray, np.ndarray]:
        """Start and end sample numbers (end exclusive) of every pedestal window."""
        return _window_runs(self.rd)

    def signal_window_slices(self) -> tuple[np.ndarray, np.ndarray]:
        """Start and end sample numbers (end exclusive) of every signal window."""
        return _window_runs(self.ru)

    def skip_cycles(self) -> tuple[np.ndarray, int, np.ndarray]:
        """
        Fold the raw samples into one row per skip cycle.

        Every cycle starts at the beginning of a pedestal window and has the median
        cycle length.  Returns ``(cycles, period, pixel)``: ``cycles.shape == (n, period)``
        and ``pixel[i]`` is the pixel the cycle belongs to.  The LAST cycle of every pixel
        is the extra pedestal-only window; its row runs into the gap before the next pixel,
        so drop it (``skip == N``) when only normal cycles are wanted.
        """
        starts, _ = self.window_slices()
        period = int(np.median(np.diff(starts)))
        n_cycles = int(np.sum(starts + period <= len(self.values)))
        cycles = np.stack([self.values[s:s + period] for s in starts[:n_cycles]])
        pedestal_pixels = self.l1_pixel[~self.l1_is_signal]
        return cycles, period, pedestal_pixels[:n_cycles]

    # -- self-check -------------------------------------------------------------------
    def check_l1(self) -> int:
        """
        Largest |inline sum - sum of raw samples in the flagged window| over all skips.

        0 means the raw samples, the window flags and the inline sums agree exactly; -1 means there is no complete
        pedestal and signal window to compare (a trace cut short).
        """
        starts_rd, ends_rd = self.window_slices()
        starts_ru, ends_ru = self.signal_window_slices()
        csum = np.r_[0, np.cumsum(self.values, dtype=np.int64)]
        raw_ped = csum[ends_rd] - csum[starts_rd]
        raw_sig = csum[ends_ru] - csum[starts_ru]
        inline_ped = self.l1_sum[~self.l1_is_signal]
        inline_sig = self.l1_sum[self.l1_is_signal]
        n = min(len(raw_ped), len(inline_ped), len(raw_sig), len(inline_sig))
        if n == 0:
            return -1
        return int(max(np.abs(raw_ped[:n] - inline_ped[:n]).max(),
                       np.abs(raw_sig[:n] - inline_sig[:n]).max()))

    def quality_problems(self) -> list[str]:
        """
        Signs that part of the trace was lost or corrupted (empty list = looks healthy).

        Checked: every pedestal window has the same length, up to one sample (see below); the
        inline sums equal the sums of the raw samples; every pixel has the same number of inline
        records.  A damaged trace (e.g. UDP packets dropped during the acquisition) fails these
        checks and should not be used as a repeat or for training.

        Window lengths may alternate between N and N+1 samples: the ADC samples every 20/3
        clock cycles, so a delayInt that is not a multiple of 20 cycles (e.g. 470 -> 70.5
        samples) gives windows of 70 and 71 samples.  That is the sequencer's timing, not
        damage (the inline sums still match the raw samples exactly), and skips() uses window
        MEANS, so the CDS is unaffected.  Only lengths more than one sample from the median
        are flagged.  (The on-board image CDS is affected; see SAMPLES_PER_CYCLE.)
        """
        problems = []
        starts, ends = self.window_slices()
        lengths = ends - starts
        odd = int(np.sum(np.abs(lengths - np.median(lengths)) > 1))
        if odd:
            problems.append(f"{odd} pedestal windows have an unusual length")
        mismatch = self.check_l1()
        if mismatch < 0:
            problems.append("no complete pedestal and signal window")
        elif mismatch:
            problems.append(f"inline sums differ from the raw samples by up to {mismatch}")
        counts = np.array([b - a for a, b in self.pixel_segments()])
        if len(counts) and counts.min() != counts.max():
            problems.append(f"pixels have different numbers of inline records ({counts.min()} to {counts.max()})")
        if len(starts) != int(np.sum(~self.l1_is_signal)):
            problems.append(f"{len(starts)} pedestal windows in the samples but {int(np.sum(~self.l1_is_signal))} inline pedestal records")
        return problems

    def to_csv(self, path: str | Path) -> None:
        """Write ``sample, val, RD, RU`` (the layout ``commant`` expects from ``df0_view``)."""
        table = np.column_stack([np.arange(len(self.values)), self.values,
                                 self.rd.astype(int), self.ru.astype(int)])
        np.savetxt(path, table, fmt="%d", delimiter=",", header="sample,val,RD,RU", comments="")


def _window_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Start and end (exclusive) sample numbers of the runs of True in ``mask``.

    Runs shorter than half the typical run are dropped: real traces contain isolated
    one-sample flag glitches that would otherwise be counted as extra windows.
    """
    change = np.diff(np.r_[0, mask.view(np.int8), 0])
    starts, ends = np.flatnonzero(change == 1), np.flatnonzero(change == -1)
    keep = (ends - starts) >= 0.5 * np.median(ends - starts)
    return starts[keep], ends[keep]


def _signed(value: np.ndarray, bits: int) -> np.ndarray:
    """Interpret the low ``bits`` bits of ``value`` as a two's-complement integer."""
    value = (value & ((1 << bits) - 1)).astype(np.int64)
    return np.where(value >= (1 << (bits - 1)), value - (1 << bits), value)


def decode_trace(path: str | Path) -> DecodedTrace:
    """Decode one ``.RJ45ADCDATA<ch>`` file (see the module docstring for the format)."""
    words = np.fromfile(path, dtype=">u4")
    flag = (words >> 20).astype(np.int64)            # 12 flag bits
    top = words >> 28                                # top nibble: record type

    keyword = (words & _KEYWORD_BIT) != 0
    raw = top == 2

    # L1 skip records: header word (top nibble 5, pixel 0..15, n samples) + sum word.
    low20 = (words & 0xFFFFF).astype(np.int64)
    header = (top == 5) & ((flag & 0x70) == 0) & (low20 > 0) & (low20 <= _MAX_WINDOW)
    header[-1:] = False
    header_pos = np.flatnonzero(header)
    sum_pos = header_pos + 1
    usable = ~keyword[sum_pos]                        # the sum word must be plain data
    header_pos, sum_pos = header_pos[usable], sum_pos[usable]

    taken = np.zeros(len(words), dtype=bool)
    taken[header_pos] = True
    taken[sum_pos] = True
    raw &= ~taken                                     # a sum word is never a sample, whatever its top nibble
    other = ~(keyword | raw | taken)

    return DecodedTrace(
        path=str(path),
        keywords=words[keyword],
        values=_signed(words[raw], 20).astype(np.int32),
        rd=(flag[raw] & 0x1) != 0,
        ru=(flag[raw] & 0x2) != 0,
        flags=flag[raw].astype(np.uint16),
        l1_pixel=(flag[header_pos] & 0xF).astype(np.int16),
        l1_is_signal=(flag[header_pos] & 0x80) != 0,
        l1_n=low20[header_pos],
        l1_sum=_signed(words[sum_pos], 31),
        other_index=np.flatnonzero(other),
        other=words[other],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode an ACM trace file and print a summary.")
    parser.add_argument("trace", help="path to a .RJ45ADCDATA<ch> file")
    parser.add_argument("--csv", help="also write sample,val,RD,RU to this CSV file")
    args = parser.parse_args()

    trace = decode_trace(args.trace)
    cycles, period, _ = trace.skip_cycles()
    skips = trace.skips()
    print(f"file              : {args.trace}")
    print(f"keyword words     : {[hex(int(k)) for k in trace.keywords]}")
    print(f"raw samples       : {len(trace.values)}  ({len(trace.values) / SAMPLE_RATE_HZ * 1e3:.1f} ms)")
    print(f"skip cycles       : {len(cycles)} of {period} samples ({period / SAMPLE_RATE_HZ * 1e6:.1f} us)")
    print(f"L1 records        : {len(trace.l1_sum)} (pixels {np.unique(trace.l1_pixel).tolist()})")
    print(f"undecoded words   : {len(trace.other)}")
    print(f"inline vs raw sums: max difference {trace.check_l1()} (0 = exact agreement)")
    problems = trace.quality_problems()
    print("quality           : " + ("OK" if not problems else "PROBLEMS: " + "; ".join(problems)))
    print(f"skips per pixel   : {len(skips['cds']) // len(np.unique(skips['pixel']))}  "
          f"(+1 extra pedestal window per pixel)")
    print(f"CDS rms per skip  : {skips['cds'].std():.1f} ADU (uncalibrated, includes transients)")
    if args.csv:
        trace.to_csv(args.csv)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
