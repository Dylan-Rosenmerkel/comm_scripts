# copied from ccd_opt_ml/src/view_trace.py (commit 3d7855e) by utils/sync_from_ccd_opt_ml.py: edit it there, not here
"""
view_trace.py -- an interactive viewer for a whole decoded trace (zoom and pan in a browser).

Writes ONE self-contained HTML file that embeds every raw sample of the trace (compressed), the
pedestal/signal window flags and the region boundaries found by ``trace_regions.py``.  Open it
in any modern browser; no server and no internet connection are needed.

In the viewer
-------------
    mouse wheel        zoom in/out around the pointer (hold Shift: zoom the vertical axis only)
    drag               pan
    double click       back to the full trace
    arrow keys, + -    pan / zoom
    drop-down          jump to the vertical block, or the horizontal block / skip region of a pixel
    buttons            toggle the region shading, the pedestal/signal window shading, the robust
                       y scale (ignores rare outlier samples when choosing the range), lock the y axis

How it stays fast: the browser builds a min/max pyramid of the samples when the file is opened,
so every pixel column of the plot is the minimum and maximum of the samples it covers, at any zoom.
When zoomed far enough that single samples are visible they are drawn as points.

Usage:  python view_trace.py TRACE_FILE [OUT.html]
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path

import numpy as np

from decode_trace import SAMPLE_RATE_HZ, decode_trace
from trace_regions import split_regions

_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 10px 14px; background: #fafafa; color: #222; }
  h1 { font-size: 15px; margin: 0 0 6px 0; font-weight: 600; }
  #bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 6px; font-size: 13px; }
  button, select { font-size: 13px; padding: 3px 8px; }
  button.on { background: #cfe3ff; }
  #readout { font-variant-numeric: tabular-nums; min-width: 420px; }
  canvas { background: white; border: 1px solid #bbb; width: 100%; height: 540px; display: block; cursor: crosshair; }
  #legend span { margin-right: 14px; font-size: 12px; }
  .sw { display: inline-block; width: 12px; height: 12px; vertical-align: -2px; margin-right: 4px; opacity: .6; }
  #msg { color: #a00; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<div id="bar">
  <button id="full">Full trace</button>
  <select id="jump"></select>
  <button id="tglRegions" class="on">Regions</button>
  <button id="tglWindows" class="on">Pedestal / signal windows</button>
  <button id="tglRobust" class="on" title="ignore rare outlier samples when choosing the y range">Robust y scale</button>
  <button id="tglLock">Lock y axis</button>
  <span id="readout"></span>
  <span id="msg"></span>
</div>
<canvas id="cv"></canvas>
<div id="legend">
  <span><i class="sw" style="background:#1f77b4"></i>vertical transfers</span>
  <span><i class="sw" style="background:#ff7f0e"></i>horizontal transfers</span>
  <span><i class="sw" style="background:#2ca02c"></i>skip region</span>
  <span><i class="sw" style="background:#d62728"></i>pedestal window (RD)</span>
  <span><i class="sw" style="background:#17becf"></i>signal window (RU)</span>
  <span id="info"></span>
</div>
<script>
const META = __META__;
const B64 = "__DATA__";

(async function () {
  const msg = document.getElementById("msg");
  // ---- decode the embedded data ----------------------------------------------------------
  msg.textContent = "decoding ...";
  const bin = Uint8Array.from(atob(B64), c => c.charCodeAt(0));
  const raw = await new Response(new Blob([bin]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
  const N = META.n;
  const vals = new Int32Array(raw, 0, N);
  const flags = new Uint8Array(raw, N * 4, N);
  const RATE = META.rate_mhz;                 // samples per microsecond

  // ---- min/max pyramid (level l: blocks of 2^l samples) ----------------------------------------
  const mins = [vals], maxs = [vals];
  for (let l = 1; (1 << l) <= N && l <= 18; l++) {
    const pm = mins[l - 1], pM = maxs[l - 1], n = Math.ceil(pm.length / 2);
    const mn = new Int32Array(n), mx = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const a = 2 * i, b = Math.min(a + 1, pm.length - 1);
      mn[i] = Math.min(pm[a], pm[b]); mx[i] = Math.max(pM[a], pM[b]);
    }
    mins.push(mn); maxs.push(mx);
  }
  msg.textContent = "";
  function rangeMinMax(a, b) {                // min and max of samples [a, b)
    a = Math.max(0, Math.floor(a)); b = Math.min(N, Math.max(a + 1, Math.ceil(b)));
    const len = b - a;
    const l = Math.min(mins.length - 1, Math.max(0, Math.floor(Math.log2(len)) - 1));
    const s = 1 << l;
    let lo = Infinity, hi = -Infinity;
    for (let i = Math.floor(a / s); i <= Math.floor((b - 1) / s); i++) {
      if (mins[l][i] < lo) lo = mins[l][i];
      if (maxs[l][i] > hi) hi = maxs[l][i];
    }
    return [lo, hi];
  }

  // ---- view state ------------------------------------------------------------------------------------------
  const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
  const M = { l: 78, r: 14, t: 12, b: 38 };
  let x0 = 0, x1 = N;                          // visible sample range
  let y0 = 0, y1 = 1, lockY = false, showRegions = true, showWindows = true;
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    cv.width = cv.clientWidth * dpr; cv.height = cv.clientHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); draw();
  }
  window.addEventListener("resize", resize);
  const W = () => cv.clientWidth - M.l - M.r, H = () => cv.clientHeight - M.t - M.b;
  const sx = s => M.l + (s - x0) / (x1 - x0) * W();
  const sy = v => M.t + (y1 - v) / (y1 - y0) * H();
  const xs = px => x0 + (px - M.l) / W() * (x1 - x0);

  let robustY = true, clipped = false;
  function autoY() {
    let lo, hi;
    clipped = false;
    if (robustY && (x1 - x0) > 200) {            // ignore the most extreme 0.5% of the pixel columns
      const cols = 400, step = (x1 - x0) / cols, los = [], his = [];
      for (let c = 0; c < cols; c++) { const [a, b] = rangeMinMax(x0 + c * step, x0 + (c + 1) * step); los.push(a); his.push(b); }
      los.sort((a, b) => a - b); his.sort((a, b) => a - b);
      lo = los[Math.floor(cols * 0.005)]; hi = his[Math.floor(cols * 0.995)];
      const [flo, fhi] = rangeMinMax(x0, x1), tol = 0.3 * (hi - lo);
      clipped = flo < lo - tol || fhi > hi + tol;      // only report clearly distant outliers
    } else [lo, hi] = rangeMinMax(x0, x1);
    const pad = 0.05 * (hi - lo || 1);
    y0 = lo - pad; y1 = hi + pad;
  }
  function clampX() {
    const span = Math.max(8, Math.min(N, x1 - x0));
    const c = (x0 + x1) / 2; x0 = c - span / 2; x1 = c + span / 2;
    if (x0 < 0) { x1 -= x0; x0 = 0; } if (x1 > N) { x0 -= x1 - N; x1 = N; }
    x0 = Math.max(0, x0);
  }
  function niceStep(span, target) {
    const raw = span / target, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
    return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
  }
  function fmtTime(us) { return Math.abs(us) >= 1000 ? (us / 1000).toPrecision(5).replace(/\.?0+$/, "") + " ms" : us.toPrecision(5).replace(/\.?0+$/, "") + " us"; }

  // ---- drawing ------------------------------------------------------------------------------------------------------
  function draw() {
    clampX(); if (!lockY) autoY();
    const w = cv.clientWidth, h = cv.clientHeight;
    ctx.clearRect(0, 0, w, h);
    ctx.font = "11px system-ui"; ctx.textBaseline = "middle";
    const spanUs = (x1 - x0) / RATE;

    // regions
    if (showRegions) {
      const regs = [["V", META.v, "#1f77b4"], ["H", META.h, "#ff7f0e"], ["S", META.skip, "#2ca02c"]];
      for (const [, list, color] of regs) {
        ctx.fillStyle = color; ctx.globalAlpha = 0.13;
        for (const [a, b] of list) {
          if (b < x0 || a > x1) continue;
          const pa = Math.max(M.l, sx(a)), pb = Math.min(M.l + W(), sx(b));
          ctx.fillRect(pa, M.t, Math.max(1, pb - pa), H());
        }
      }
      ctx.globalAlpha = 1;
    }
    // windows
    if (showWindows && (x1 - x0) < 60000) {
      const i0 = Math.max(0, Math.floor(x0)), i1 = Math.min(N, Math.ceil(x1));
      for (const [bit, color] of [[1, "#d62728"], [2, "#17becf"]]) {
        ctx.fillStyle = color; ctx.globalAlpha = 0.18;
        let start = -1;
        for (let i = i0; i <= i1; i++) {
          const on = i < i1 && (flags[i] & bit) !== 0;
          if (on && start < 0) start = i;
          if (!on && start >= 0) { ctx.fillRect(sx(start), M.t, Math.max(1, sx(i) - sx(start)), H()); start = -1; }
        }
      }
      ctx.globalAlpha = 1;
    }
    // grid and axes
    ctx.strokeStyle = "#e3e3e3"; ctx.fillStyle = "#444"; ctx.lineWidth = 1;
    const xStep = niceStep(spanUs, 9) * RATE;
    ctx.textAlign = "center";
    for (let s = Math.ceil(x0 / xStep) * xStep; s <= x1; s += xStep) {
      const px = Math.round(sx(s)) + 0.5;
      ctx.beginPath(); ctx.moveTo(px, M.t); ctx.lineTo(px, M.t + H()); ctx.stroke();
      ctx.fillText(fmtTime(s / RATE), px, M.t + H() + 14);
    }
    const yStep = niceStep(y1 - y0, 7);
    ctx.textAlign = "right";
    for (let v = Math.ceil(y0 / yStep) * yStep; v <= y1; v += yStep) {
      const py = Math.round(sy(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(M.l, py); ctx.lineTo(M.l + W(), py); ctx.stroke();
      ctx.fillText(String(Math.round(v * 100) / 100), M.l - 6, py);
    }
    ctx.strokeStyle = "#888"; ctx.strokeRect(M.l + 0.5, M.t + 0.5, W(), H());
    ctx.save(); ctx.translate(14, M.t + H() / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center";
    ctx.fillText("ADC value (ADU)", 0, 0); ctx.restore();
    ctx.textAlign = "center"; ctx.fillText("time (sample number / 15 MHz)", M.l + W() / 2, M.t + H() + 30);

    // the trace
    const spp = (x1 - x0) / W();                       // samples per screen pixel
    ctx.save(); ctx.beginPath(); ctx.rect(M.l, M.t, W(), H()); ctx.clip();
    if (spp > 0.6) {                                    // envelope: min..max per pixel column
      ctx.strokeStyle = "#222"; ctx.lineWidth = 1; ctx.beginPath();
      for (let px = 0; px < W(); px++) {
        const [lo, hi] = rangeMinMax(x0 + px * spp, x0 + (px + 1) * spp);
        const X = M.l + px + 0.5; ctx.moveTo(X, sy(lo)); ctx.lineTo(X, sy(hi) - 0.01);
      }
      ctx.stroke();
    } else {                                            // single samples: line and points
      const i0 = Math.max(0, Math.floor(x0) - 1), i1 = Math.min(N - 1, Math.ceil(x1) + 1);
      ctx.strokeStyle = "#222"; ctx.lineWidth = 1; ctx.beginPath();
      for (let i = i0; i <= i1; i++) { const X = sx(i), Y = sy(vals[i]); i === i0 ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y); }
      ctx.stroke();
      if (spp < 0.12) { ctx.fillStyle = "#222"; for (let i = i0; i <= i1; i++) ctx.fillRect(sx(i) - 1.5, sy(vals[i]) - 1.5, 3, 3); }
    }
    ctx.restore();
    document.getElementById("info").textContent =
      "showing " + fmtTime(spanUs) + " = " + Math.round(x1 - x0).toLocaleString() + " of " + N.toLocaleString() + " samples" +
      (clipped && !lockY ? "  (rare outliers beyond the y range: switch off 'Robust y scale' to see them)" : "");
  }

  // ---- interaction ------------------------------------------------------------------------------------------------
  function zoom(factor, atPx, yOnly) {
    if (yOnly) { const c = (y0 + y1) / 2, half = (y1 - y0) / 2 * factor; y0 = c - half; y1 = c + half; lockY = true; sync(); draw(); return; }
    const anchor = xs(atPx), left = (anchor - x0) * factor, right = (x1 - anchor) * factor;
    x0 = anchor - left; x1 = anchor + right; draw();
  }
  function sync() { document.getElementById("tglLock").classList.toggle("on", lockY); }
  cv.addEventListener("wheel", e => {
    e.preventDefault();
    const r = cv.getBoundingClientRect(); zoom(e.deltaY < 0 ? 0.8 : 1.25, e.clientX - r.left, e.shiftKey);
  }, { passive: false });
  let drag = null;
  cv.addEventListener("mousedown", e => { drag = { x: e.clientX, y: e.clientY, x0, x1, y0, y1 }; });
  window.addEventListener("mouseup", () => { drag = null; });
  window.addEventListener("mousemove", e => {
    const r = cv.getBoundingClientRect(), px = e.clientX - r.left;
    if (drag) {
      const ds = (e.clientX - drag.x) / W() * (drag.x1 - drag.x0);
      x0 = drag.x0 - ds; x1 = drag.x1 - ds;
      if (lockY) { const dy = (e.clientY - drag.y) / H() * (drag.y1 - drag.y0); y0 = drag.y0 + dy; y1 = drag.y1 + dy; }
      draw();
    }
    const ro = document.getElementById("readout");
    if (px >= M.l && px <= M.l + W() && e.clientY - r.top >= M.t && e.clientY - r.top <= M.t + H()) {
      const s = xs(px), i = Math.round(s);
      if (i >= 0 && i < N) {
        const f = flags[i], win = (f & 1 ? " [pedestal window]" : "") + (f & 2 ? " [signal window]" : "");
        const spp = (x1 - x0) / W();
        const [lo, hi] = rangeMinMax(s - spp / 2, s + spp / 2);
        ro.textContent = "sample " + i.toLocaleString() + "   t = " + fmtTime(i / RATE) + "   " +
          (spp > 0.6 ? "min " + lo + "  max " + hi : "ADC " + vals[i]) + win;
      }
    } else ro.textContent = "";
  });
  cv.addEventListener("dblclick", () => { x0 = 0; x1 = N; lockY = false; sync(); draw(); });
  window.addEventListener("keydown", e => {
    const span = x1 - x0;
    if (e.key === "ArrowLeft") { x0 -= span * 0.2; x1 -= span * 0.2; }
    else if (e.key === "ArrowRight") { x0 += span * 0.2; x1 += span * 0.2; }
    else if (e.key === "+" || e.key === "=") zoom(0.8, M.l + W() / 2, false);
    else if (e.key === "-") zoom(1.25, M.l + W() / 2, false);
    else return; draw();
  });
  document.getElementById("full").onclick = () => { x0 = 0; x1 = N; lockY = false; sync(); draw(); };
  document.getElementById("tglRegions").onclick = e => { showRegions = !showRegions; e.target.classList.toggle("on", showRegions); draw(); };
  document.getElementById("tglWindows").onclick = e => { showWindows = !showWindows; e.target.classList.toggle("on", showWindows); draw(); };
  document.getElementById("tglRobust").onclick = e => { robustY = !robustY; e.target.classList.toggle("on", robustY); draw(); };
  document.getElementById("tglLock").onclick = () => { lockY = !lockY; sync(); draw(); };

  const jump = document.getElementById("jump");
  const targets = [["Jump to ...", null]];
  if (META.v.length) targets.push(["vertical transfers", META.v[0]]);
  META.skip.forEach((b, k) => { targets.push(["pixel " + k + ": horizontal transfers", META.h[k]]); targets.push(["pixel " + k + ": skip region", b]); targets.push(["pixel " + k + ": first 5 skips", [b[0], b[0] + 5 * META.period]]); });
  targets.forEach(([label, range], i) => { const o = document.createElement("option"); o.value = i; o.textContent = label; jump.appendChild(o); });
  jump.onchange = () => {
    const r = targets[jump.value][1]; if (!r) return;
    const pad = (r[1] - r[0]) * 0.03; x0 = r[0] - pad; x1 = r[1] + pad; lockY = false; sync(); draw(); jump.value = 0;
  };
  resize();
})();
</script>
</body>
</html>
"""


def build_html(trace_path: str) -> str:
    """Decode ``trace_path`` and return the self-contained viewer page as a string."""
    trace = decode_trace(trace_path)
    regions = split_regions(trace)

    values = trace.values.astype("<i4")
    flags = (trace.rd.astype(np.uint8) | (trace.ru.astype(np.uint8) << 1))
    payload = base64.b64encode(gzip.compress(values.tobytes() + flags.tobytes(), compresslevel=6)).decode("ascii")

    h_blocks = [[s - regions.h_block, s] for s, _ in regions.skip_bounds]
    meta = {
        "n": int(len(values)),
        "rate_mhz": SAMPLE_RATE_HZ / 1e6,
        "period": regions.period,
        "skip": [list(map(int, b)) for b in regions.skip_bounds],
        "h": [list(map(int, b)) for b in h_blocks],
        "v": [list(map(int, regions.v_bounds))] if regions.v_bounds else [],
    }
    title = f"{Path(trace_path).name} ({len(values):,} samples, {len(values) / SAMPLE_RATE_HZ * 1e3:.0f} ms)"
    return (_TEMPLATE.replace("__TITLE__", title)
            .replace("__META__", json.dumps(meta))
            .replace("__DATA__", payload))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace")
    parser.add_argument("out", nargs="?", help="output HTML file (default: plots/viewer_<trace name>.html)")
    args = parser.parse_args()

    out = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "plots" / f"viewer_{Path(args.trace).stem}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(args.trace))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
