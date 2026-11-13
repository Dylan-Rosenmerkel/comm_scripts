/* superWADERs reviews: the review panel shared by the overview pages and the module pages (inlined into both).

A review: {id, campaign, created, updated, reviewer, status: approved | concern | rejected | note, comment,
           target: {tool, module, ext, ccd, view, page, tab}, data: the data it refers to (a stamp),
           values: the main numbers shown when it was written, page}.
Where reviews live (no server: the pages are files):
  <campaign>/reviews/*.json    one file per save, {format, campaign, saved, reviews: [...]}; several people can save
                               into the same (shared) folder without overwriting each other
  <campaign>/reviews.js        all of them merged (window.SW_REVIEWS), loaded by every page; written by
                               run_superWADERs / commissioning/reviews.py, or by this page ("Save to campaign folder",
                               Chrome / Edge: File System Access)
  this browser                 reviews not saved yet (localStorage), shown as "not saved"; "Download" writes them to a
                               file for the reviews/ folder (any browser)
Merging: by id, the latest 'updated' wins; a deleted review is kept as a deletion record. */
const SWR = (() => {
  const STATUS = { approved: ["✔", "pass", "approved"], concern: ["▲", "warn", "concern"], rejected: ["✖", "fail", "rejected"],
                   note: ["✎", "na", "note"] };
  const FORMAT = "superWADERs-reviews-1";
  let cfg = { campaign: "", page: "", onChange: () => {}, same: () => true };   // same(r): r's stamp is comparable here
  let mem = [];
  const esc = s => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const pkey = () => "swreviews:pending:" + (cfg.campaign || "?");
  const saved = () => ((window.SW_REVIEWS && Array.isArray(window.SW_REVIEWS.reviews)) ? window.SW_REVIEWS.reviews : []);
  function pending() { try { return JSON.parse(localStorage.getItem(pkey()) || "[]"); } catch (e) { return mem; } }
  function setPending(a) { mem = a; try { localStorage.setItem(pkey(), JSON.stringify(a)); } catch (e) {} }
  function merge(list) {
    const m = new Map();
    for (const r of list) { if (!r || !r.id) continue; const o = m.get(r.id); if (!o || String(r.updated || "") >= String(o.updated || "")) m.set(r.id, r); }
    return [...m.values()].sort((a, b) => String(a.created || "").localeCompare(String(b.created || "")));
  }
  function all() {
    const p = new Set(pending().map(r => r.id));
    return merge(saved().concat(pending())).filter(r => !r.deleted).map(r => Object.assign({ unsaved: p.has(r.id) }, r));
  }
  function name() { try { return localStorage.getItem("swreviews:name") || ""; } catch (e) { return ""; } }
  function setName(n) { try { localStorage.setItem("swreviews:name", n); } catch (e) {} }
  function uid() { return (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : Date.now().toString(36) + Math.random().toString(36).slice(2); }
  function add(target, stamp, status, comment, reviewer, values) {
    const now = new Date().toISOString();
    const r = { id: uid(), campaign: cfg.campaign, created: now, updated: now, reviewer, status, comment, target, data: stamp || null,
                values: values || null, page: cfg.page };
    setPending(pending().concat([r])); cfg.onChange(); return r;
  }
  function remove(id) {
    const p = pending();
    if (p.some(r => r.id === id)) setPending(p.filter(r => r.id !== id));
    else { const r = saved().find(x => x.id === id); if (r) setPending(p.concat([Object.assign({}, r, { deleted: true, updated: new Date().toISOString(), deleted_by: name() })])); }
    cfg.onChange();
  }
  const latest = rs => rs.length ? rs[rs.length - 1] : null;
  const stale = (r, stamp) => !!(stamp && r.data && r.data !== stamp && cfg.same(r));
  function badge(rs, stamp) {
    const r = latest(rs); if (!r) return "";
    const s = STATUS[r.status] || STATUS.note, st = stale(r, stamp);
    return `<span class="swr-badge swr-${s[1]}" title="${rs.length} review(s); latest: ${esc(r.reviewer)}, ${s[2]}${st ? " — the data changed after this review" : ""}${r.unsaved ? " (not saved yet)" : ""}">${s[0]}${rs.length > 1 ? " " + rs.length : ""}${st ? " *" : ""}</span>`;
  }
  function statusText(rs, stamp) { const r = latest(rs); if (!r) return ""; return `${(STATUS[r.status] || STATUS.note)[2]} (${r.reviewer}${stale(r, stamp) ? ", data changed since" : ""})`; }
  function where(t) { return [t.tool && t.tool !== "campaign" ? t.tool : "", t.view, t.page ? "report " + t.page : "", t.tab ? "tab " + t.tab : "", t.ext || (t.module ? "whole module" : "")].filter(Boolean).join(" · "); }
  function listHTML(rs, stamp) {
    if (!rs.length) return `<div class="swr-none">No review yet.</div>`;
    return rs.slice().reverse().map(r => { const s = STATUS[r.status] || STATUS.note;
      return `<div class="swr-item swr-${s[1]}"><div class="swr-head"><b>${s[0]} ${esc(s[2])}</b> · ${esc(r.reviewer || "?")} · ${esc(String(r.updated || r.created || "").slice(0, 16).replace("T", " "))}`
        + `${r.unsaved ? ' · <span class="swr-unsaved">not saved yet</span>' : ""}${stale(r, stamp) ? ' · <span class="swr-stale">data changed since</span>' : ""}`
        + `<button class="swr-x" data-id="${esc(r.id)}" title="delete this review">×</button></div>`
        + `${r.comment ? `<div class="swr-text">${esc(r.comment).replace(/\n/g, "<br>")}</div>` : ""}<div class="swr-where">${esc(where(r.target || {}))}`
        + `${r.values && stale(r, stamp) ? `<br>values then: ${esc(Object.entries(r.values).map(([k, v]) => k + " " + v).join(", "))}` : ""}</div></div>`; }).join("");
  }
  /* the list and the form: el = container; q(r) -> does review r belong here; target, stamp: of a new review;
     choices: optional [[label, target patch]] for what the review is about (e.g. one CCD or the whole module);
     values: the numbers shown now ({label: text}), kept with the review */
  function form(el, q, target, stamp, choices, values) {
    const rs = all().filter(q);
    const sel = choices && choices.length > 1 ? `<select class="swr-on">${choices.map((c, i) => `<option value="${i}">${esc(c[0])}</option>`).join("")}</select>` : "";
    el.innerHTML = `<div class="swr-list">${listHTML(rs, stamp)}</div>
      <div class="swr-form noprint"><div class="swr-row"><input class="swr-name" placeholder="your name" value="${esc(name())}">
      <select class="swr-status"><option value="approved">✔ approved</option><option value="concern">▲ concern</option><option value="rejected">✖ rejected</option><option value="note" selected>✎ note</option></select>${sel}</div>
      <textarea class="swr-comment" placeholder="comment (what you checked, what is wrong, what to do)"></textarea>
      <div class="swr-row"><button class="swr-add">Add review</button><span class="swr-msg"></span></div></div>`;
    el.querySelectorAll(".swr-x").forEach(b => b.onclick = () => { if (confirm("Delete this review?")) remove(b.dataset.id); });
    el.querySelector(".swr-add").onclick = () => {
      const n = el.querySelector(".swr-name").value.trim(), c = el.querySelector(".swr-comment").value.trim(), s = el.querySelector(".swr-status").value;
      if (!n) { el.querySelector(".swr-msg").textContent = "give your name"; return; }
      if (!c && s === "note") { el.querySelector(".swr-msg").textContent = "write a comment"; return; }
      setName(n);
      const ch = sel ? choices[+el.querySelector(".swr-on").value][1] : {};
      add(Object.assign({}, target, ch), stamp, s, c, n, values);
    };
  }
  function fileText(list) { return JSON.stringify({ format: FORMAT, campaign: cfg.campaign, saved: new Date().toISOString(), reviewer: name(), reviews: list }, null, 1); }
  const fname = () => `reviews_${(name() || "anonymous").replace(/[^\w-]+/g, "_")}_${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
  function download() {
    const p = pending(); if (!p.length) return;
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([fileText(p)], { type: "application/json" })); a.download = fname(); a.click();
    if (confirm("Downloaded " + p.length + " review(s).\n\nPut the file into the campaign's reviews/ folder and run\n  run_superWADERs --campaign DIR\n(or: python superWADERs/commissioning/reviews.py DIR) so that every page shows them.\n\nClear them from this browser now? (Cancel keeps them here, shown as 'not saved'.)")) { setPending([]); cfg.onChange(); }
  }
  async function saveToFolder() {
    const p = pending(); if (!p.length) return;
    let dir;
    try { dir = await window.showDirectoryPicker({ id: "swcampaign", mode: "readwrite" }); } catch (e) { return; }
    try { await dir.getFileHandle("campaign.json"); } catch (e) { alert("That is not a campaign folder (no campaign.json). Choose the folder that holds index.html and campaign.json."); return; }
    const rdir = await dir.getDirectoryHandle("reviews", { create: true });
    const w = await (await rdir.getFileHandle(fname(), { create: true })).createWritable(); await w.write(fileText(p)); await w.close();
    const got = [];
    for await (const [n, h] of rdir.entries()) {
      if (h.kind === "file" && n.endsWith(".json")) { try { const j = JSON.parse(await (await h.getFile()).text()); got.push(...(Array.isArray(j) ? j : (j.reviews || []))); } catch (e) {} }
    }
    const merged = merge(got);
    const js = { format: FORMAT, campaign: cfg.campaign, made: new Date().toISOString(), reviews: merged };
    const w2 = await (await dir.getFileHandle("reviews.js", { create: true })).createWritable();
    await w2.write("window.SW_REVIEWS = " + JSON.stringify(js) + ";\n"); await w2.close();
    window.SW_REVIEWS = js; setPending([]); cfg.onChange();
    alert("Saved " + p.length + " review(s) into reviews/ and updated reviews.js: every page of the campaign shows them (reload a page that is already open).");
  }
  function bar(el) {
    const n = pending().length;
    const canFolder = !!window.showDirectoryPicker;
    el.innerHTML = `<span class="swr-bar-t">Reviews: ${all().length} in this campaign${n ? ` · <b class="swr-unsaved">${n} not saved yet</b>` : ""}</span>`
      + (n ? (canFolder ? `<button class="swr-save">Save to campaign folder</button>` : "") + `<button class="swr-dl">Download${canFolder ? "" : " to save"}</button>` : "")
      + `<button class="swr-help">?</button>`;
    const q = s => el.querySelector(s);
    if (q(".swr-save")) q(".swr-save").onclick = saveToFolder;
    if (q(".swr-dl")) q(".swr-dl").onclick = download;
    q(".swr-help").onclick = () => alert("Reviews: give your name, a status (approved / concern / rejected / note) and a comment for a CCD (overview pages) or a tab of a report (module pages).\n\nA new review is kept in this browser until it is saved:\n  Save to campaign folder (Chrome, Edge): choose the campaign folder; the review goes into reviews/ and every page shows it.\n  Download (any browser): put the file into the campaign's reviews/ folder and run  run_superWADERs --campaign DIR  (no input needed).\n\nEach save is its own file, so several people can review the same (shared) folder. A review records the data it refers to: when newer data arrive it is marked 'data changed since'.");
  }
  function style() {
    if (document.getElementById("swr-style")) return;
    const s = document.createElement("style"); s.id = "swr-style";
    s.textContent = `.swr-badge{display:inline-block;font-size:11px;font-weight:700;padding:0 5px;border-radius:8px;margin-left:4px;white-space:nowrap}
.swr-pass{color:var(--pass,#2e9d5b)}.swr-warn{color:var(--warn,#d68a00)}.swr-fail{color:var(--fail,#d23f3f)}.swr-na{color:var(--muted,#888)}
.swr-badge.swr-pass{background:var(--passbg,#e5f4ea)}.swr-badge.swr-warn{background:var(--warnbg,#fbf0d9)}.swr-badge.swr-fail{background:var(--failbg,#fbe3e3)}.swr-badge.swr-na{background:var(--nabg,#ececee)}
.swr-item{border-left:3px solid currentColor;padding:4px 8px;margin:6px 0;font-size:13px}.swr-item .swr-head{color:inherit}.swr-text{color:var(--fg,#111);margin-top:2px}
.swr-where{color:var(--muted,#888);font-size:11px}.swr-none{color:var(--muted,#888);font-size:13px}.swr-unsaved{color:var(--warn,#d68a00)}.swr-stale{color:var(--fail,#d23f3f)}
.swr-x{float:right;border:none;background:none;color:var(--muted,#888);cursor:pointer;font-size:14px;padding:0 2px}
.swr-form{margin-top:8px}.swr-row{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:4px 0}.swr-name{flex:1;min-width:110px}
.swr-comment{width:100%;min-height:60px;font:13px/1.4 inherit;box-sizing:border-box}
.swr-form input,.swr-form select,.swr-form textarea,.swr-form button,#swrbar button{font:inherit;font-size:13px;padding:4px 7px;border:1px solid var(--line,#ccc);border-radius:6px;background:var(--panel,var(--bg,#fff));color:var(--fg,#111)}
.swr-form button,#swrbar button{cursor:pointer}#swrbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:13px;margin:6px 0}
@media print{.swr-form,.swr-x,#swrbar button{display:none}}`;
    document.head.appendChild(s);
  }
  function init(c) { cfg = Object.assign(cfg, c); style(); }
  return { init, all, add, remove, form, bar, badge, statusText, latest, stale, STATUS };
})();
