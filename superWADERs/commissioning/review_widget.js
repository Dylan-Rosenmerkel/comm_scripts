/* superWADERs reviews on the pages (inlined into the overviews and the module reports): shows the reviews, which are
written in the campaign's spreadsheet reviews.csv and brought to the pages by run_superWADERs (reviews.js,
window.SW_REVIEWS = {format, campaign, made, sheet, lines: {"<module> <ext>": line of the sheet}, reviews: [...]}).

A review: {id, reviewer, status: approved | concern | rejected | note, comment, created, updated, data: the stamp of
the data it was written on, values: the numbers then (text), newer_data, target: {tool, module, ext, ccd, acm, dataset}}.
The pages do not write anything. */
const SWR = (() => {
  const STATUS = { approved: ["✔", "pass", "approved"], concern: ["▲", "warn", "concern"], rejected: ["✖", "fail", "rejected"],
                   note: ["✎", "na", "note"] };
  let cfg = { same: () => true };
  const esc = s => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const store = () => window.SW_REVIEWS || { reviews: [], lines: {} };
  const all = () => (store().reviews || []);
  const latest = rs => rs.length ? rs[rs.length - 1] : null;
  // newer data: the CCD's data changed after the review was written (run_superWADERs compares, reviews.py)
  const stale = r => !!r.newer_data;
  function badge(rs, stamp) {
    const r = latest(rs); if (!r) return "";
    const s = STATUS[r.status] || STATUS.note, st = stale(r, stamp);
    return `<span class="swr-badge swr-${s[1]}" title="${rs.length} review(s); latest: ${esc(r.reviewer)}, ${s[2]}${st ? " (newer data since)" : ""}">${s[0]}${rs.length > 1 ? " " + rs.length : ""}${st ? " *" : ""}</span>`;
  }
  function statusText(rs, stamp) { const r = latest(rs); if (!r) return ""; return `${(STATUS[r.status] || STATUS.note)[2]} (${r.reviewer}${stale(r, stamp) ? ", newer data since" : ""})`; }
  function where(t) { return [t.tool && t.tool !== "campaign" ? t.tool : "", t.dataset ? "data set " + t.dataset : "", t.ext || (t.module ? "whole module" : "")].filter(Boolean).join(" · "); }
  function listHTML(rs, stamp) {
    if (!rs.length) return `<div class="swr-none">No review yet.</div>`;
    return rs.slice().reverse().map(r => { const s = STATUS[r.status] || STATUS.note, st = stale(r, stamp);
      return `<div class="swr-item swr-${s[1]}"><div class="swr-head"><b>${s[0]} ${esc(s[2])}</b> · ${esc(r.reviewer)} · ${esc(r.updated || "")}`
        + `${st ? ' · <span class="swr-stale">newer data since</span>' : ""}</div>`
        + `${r.comment ? `<div class="swr-text">${esc(r.comment).replace(/\n/g, "<br>")}</div>` : ""}`
        + `<div class="swr-where">${esc(where(r.target || {}))}${r.values && st ? `<br>numbers then: ${esc(r.values)}` : ""}</div></div>`; }).join("");
  }
  // where to write a review of this CCD / module: its line in reviews.csv
  function howTo(t) {
    const L = store().lines || {}, k = `${t.module || ""} ${t.ext || ""}`.trim();
    const line = L[k] || L[k + " "] || Object.entries(L).find(([x]) => x.startsWith((t.module || "") + " "))?.[1];
    return `<div class="swr-how">To review: open <b>reviews.csv</b> in the campaign folder (Excel, Numbers, LibreOffice)`
      + `${line ? `, line <b>${line}</b> (${esc(t.module)}${t.ext ? " " + esc(t.ext) : ""})` : ""}; fill <i>reviewer</i>, <i>status</i> `
      + `(approved / concern / rejected / note) and <i>comment</i>, and save. It shows here after the next `
      + `<code>run_superWADERs --campaign &lt;folder&gt;</code> (with new data, or without: a second).</div>`;
  }
  /* the reviews of a CCD or module and where to add one: el = container; q(r) -> does review r belong here;
     target {module, ext}: what this is; stamp: the data shown now */
  function form(el, q, target, stamp) {
    el.innerHTML = `<div class="swr-list">${listHTML(all().filter(q), stamp)}</div>${howTo(target || {})}`;
  }
  function bar(el) {
    const s = store(), n = all().length;
    el.innerHTML = `<span class="swr-bar-t">Reviews: ${n}${s.made ? ` (as of ${esc(String(s.made).replace("T", " "))})` : ""}</span>`
      + `<span class="swr-hint">written in reviews.csv in the campaign folder</span><button class="swr-help">?</button>`;
    el.querySelector(".swr-help").onclick = () => alert("Reviews are written in a spreadsheet: reviews.csv in the campaign folder (open it with Excel, Numbers or LibreOffice).\n\nEvery CCD has a line, with its verdict and main numbers. Type your name (reviewer), a status (approved / concern / rejected / note) and a comment on the CCD's line and save the file. A second review of the same CCD: copy the line. A review of one tool only: put the tool (images, packet, dc_series, defects, sr) in 'tool'; of a whole module: leave 'ext' empty.\n\nThe reviews show on every page after the next run:\n  run_superWADERs --campaign <folder>\n(with new data, or without input: takes a second). A review keeps the numbers it was written on; when newer data arrive it is marked 'newer data since'.");
  }
  function style() {
    if (document.getElementById("swr-style")) return;
    const s = document.createElement("style"); s.id = "swr-style";
    s.textContent = `.swr-badge{display:inline-block;font-size:11px;font-weight:700;padding:0 5px;border-radius:8px;margin-left:4px;white-space:nowrap}
.swr-pass{color:var(--pass,#2e9d5b)}.swr-warn{color:var(--warn,#d68a00)}.swr-fail{color:var(--fail,#d23f3f)}.swr-na{color:var(--muted,#888)}
.swr-badge.swr-pass{background:var(--passbg,#e5f4ea)}.swr-badge.swr-warn{background:var(--warnbg,#fbf0d9)}.swr-badge.swr-fail{background:var(--failbg,#fbe3e3)}.swr-badge.swr-na{background:var(--nabg,#ececee)}
.swr-item{border-left:3px solid currentColor;padding:4px 8px;margin:6px 0;font-size:13px}.swr-text{color:var(--fg,#111);margin-top:2px}
.swr-where{color:var(--muted,#888);font-size:11px}.swr-none{color:var(--muted,#888);font-size:13px}.swr-stale{color:var(--fail,#d23f3f)}
.swr-how{font-size:12px;color:var(--muted,#888);margin-top:6px;line-height:1.45}.swr-hint{color:var(--muted,#888)}
#swrbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:13px;margin:6px 0}
#swrbar button{font:inherit;font-size:13px;padding:2px 8px;border:1px solid var(--line,#ccc);border-radius:6px;background:var(--panel,var(--bg,#fff));color:var(--fg,#111);cursor:pointer}
@media print{#swrbar button,.swr-how{display:none}}`;
    document.head.appendChild(s);
  }
  function init(c) { cfg = Object.assign(cfg, c); style(); }
  return { init, all, form, bar, badge, statusText, latest, stale, STATUS };
})();
