# copied from ccd_opt_ml/src/trace_geometry.py (commit 3d7855e) by utils/sync_from_ccd_opt_ml.py: edit it there, not here
"""
trace_geometry.py -- what a trace should look like, from its ``.meta`` file.

A trace's layout follows from the acquisition parameters recorded in its ``.meta``:

    NROW x NCOL pixels; per row NPBIN vertical transfers; per pixel NSBIN horizontal
    transfers, then NDCM skip cycles (pedestal window + signal window) and one extra
    pedestal window.

and the durations follow from the sequencer delays (10 ns ticks; the ADC samples at 15 MHz,
so one tick = 0.15 samples).  The formulas below are those of ``reference_v5.seq`` /
``reference_v5_pumping.seq`` (checked against real traces: cycle 488.7, window 150, signal
offset 207, horizontal transfer 1894, vertical transfer 5404 samples):

    skip cycle   = OGH + OGL + RGL + RGH + Int + SWH + SWL + Int
    windows      = Int (both)            signal window starts Int + SWH + SWL after the pedestal
    reset pulse  : OGH + OGL + RGL after the end of the signal window, then RGH of settling
    H transfer   = RGL + 3 Ha + 3 Hb + Hc
    V transfer   = RGL + 3 Va + 3 Vb

``trace_regions.split_regions`` measures everything from the data itself; this module gives
the EXPECTED values, used (1) as a cross-check (mismatches are reported as warnings) and (2)
where the data cannot show a value (e.g. the length of a horizontal transfer when there is only
one transfer per pixel).  With another sequencer file the formulas may not hold: then only the
counts (NROW, NCOL, NDCM, ...) are used.
"""

from __future__ import annotations

from dataclasses import dataclass

SAMPLES_PER_TICK = 0.15          # 10 ns ticks, 15 MHz ADC
KNOWN_SEQUENCERS = ("reference_v5.seq", "reference_v5_pumping.seq")


@dataclass
class TraceGeometry:
    """Expected layout and durations of a trace (None = unknown)."""
    nrow: int | None = None
    ncol: int | None = None
    ndcm: int | None = None
    npbin: int | None = None
    nsbin: int | None = None
    ncolpre: int = 0                  # skipped (prescan) pixels before the first read pixel of a row
    ncolpost: int = 0                 # skipped pixels after the last read pixel of a row
    timing_known: bool = False        # the sequencer formulas apply
    cycle: float | None = None        # samples per skip cycle (not an integer)
    n_window: float | None = None     # samples per pedestal/signal window
    sig_offset: float | None = None   # signal-window start after the pedestal-window start
    pulse_offset: float | None = None # reset edge after the end of the signal window
    h_transfer: float | None = None   # samples per horizontal transfer
    v_transfer: float | None = None   # samples per vertical transfer
    dump: float | None = None         # samples of the dump-gate pulse after a pixel (DGL + DGH)

    @property
    def n_pixels(self) -> int | None:
        return self.nrow * self.ncol if self.nrow and self.ncol else None

    @classmethod
    def from_meta(cls, meta: dict | None) -> "TraceGeometry":
        """Read the geometry from a ``.meta`` dict (an empty geometry if ``meta`` is None)."""
        if not meta:
            return cls()

        def integer(key):
            return int(meta[key]) if key in meta and meta[key] is not None else None

        g = cls(nrow=integer("NROW"), ncol=integer("NCOL"), ndcm=integer("NDCM"),
                npbin=integer("NPBIN"), nsbin=integer("NSBIN"),
                ncolpre=integer("NCOLPRE") or 0, ncolpost=integer("NCOLPOST") or 0)
        delays = ("delayOGH", "delayOGL", "delayRGL", "delayRGH", "delayInt", "delaySWH", "delaySWL",
                  "delay_Ha", "delay_Hb", "delay_Hc", "delay_Va", "delay_Vb")
        if str(meta.get("SEQNAME", "")) in KNOWN_SEQUENCERS and all(k in meta for k in delays):
            d = {k: float(meta[k]) * SAMPLES_PER_TICK for k in delays}
            g.timing_known = True
            g.cycle = (d["delayOGH"] + d["delayOGL"] + d["delayRGL"] + d["delayRGH"]
                       + 2 * d["delayInt"] + d["delaySWH"] + d["delaySWL"])
            g.n_window = d["delayInt"]
            g.sig_offset = d["delayInt"] + d["delaySWH"] + d["delaySWL"]
            g.pulse_offset = d["delayOGH"] + d["delayOGL"] + d["delayRGL"]
            g.h_transfer = d["delayRGL"] + 3 * d["delay_Ha"] + 3 * d["delay_Hb"] + d["delay_Hc"]
            g.v_transfer = d["delayRGL"] + 3 * d["delay_Va"] + 3 * d["delay_Vb"]
            if "delayDGL" in meta and "delayDGH" in meta:
                g.dump = (float(meta["delayDGL"]) + float(meta["delayDGH"])) * SAMPLES_PER_TICK
        return g
