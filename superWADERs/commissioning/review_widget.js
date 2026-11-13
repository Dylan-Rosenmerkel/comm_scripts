/* superWADERs reviews: the review panel shared by the overview pages and the module pages (inlined into both).

A review: {id, campaign, created, updated, reviewer, status: approved | concern | rejected | note, comment,
           target: {tool, module, ext, ccd, view, page, tab}, data: the data it refers to (a stamp),
           values: the main numbers shown when it was written, page}.
Where reviews live: one file per campaign,
  <campaign>/reviews.js        every review (window.SW_REVIEWS), read by every page; each save merges into it
How a review gets there, as soon as it is added:
  review server                the page was opened through review_server.py (http://localhost:...): POST /api/reviews
  Chrome / Edge, file://       'Save to campaign folder' asks for the folder once (remembered, File System Access);
                               later reviews are merged into its reviews.js as they are added
  otherwise                    kept in this browser (localStorage, "not saved yet", still shown on the page) until the
                               campaign's reviews.js has them; 'Download' gives the updated reviews.js to put in the
                               campaign folder
Merging: by id, the latest 'updated' wins; a deleted review is kept as a deletion record. */
const SWR = (() => {
  const STATUS = { approved: ["✔", "pass", "approved"], concern: ["▲", "warn", "concern"], rejected: ["✖", "fail", "rejected"],
                   note: ["✎", "na", "note"] };
  const FORMAT = "superWADERs-reviews-1";
  let cfg = { campaign: "", page: "", root: "", onChange: () => {}, same: () => true };   // same(r): r's stamp is comparable here
  let mode = location.protocol.startsWith("http") ? "probing" : "file", msg = "", busy = false;   // server | folder | file
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
    setPending(pending().concat([r])); cfg.onChange(); persist(); return r;
  }
  function remove(id) {
    const p = pending();
    if (p.some(r => r.id === id)) setPending(p.filter(r => r.id !== id));
    else { const r = saved().find(x => x.id === id); if (r) setPending(p.concat([Object.assign({}, r, { deleted: true, updated: new Date().toISOString(), deleted_by: name() })])); }
    cfg.onChange(); persist();
  }
  // ---- saving: through the review server, or into the remembered campaign folder (Chrome / Edge)
  const api = () => new URL(cfg.root + "api/reviews", location.href).href;
  async function probe() {
    try {
      const r = await fetch(api(), { cache: "no-store" }); if (!r.ok) throw new Error(r.status);
      const j = await r.json(); if (!j.server) throw new Error("no server");
      window.SW_REVIEWS = j; mode = "server"; prune();
    } catch (e) { mode = "file"; }
    if (mode === "file" && await folderHandle()) mode = "folder";
    cfg.onChange(); persist();
  }
  async function persist() {
    const p = pending(); if (!p.length || busy) return;
    if (mode === "server") {
      busy = true;
      try {
        const r = await fetch(api(), { method: "POST", headers: { "Content-Type": "application/json", "X-SW-Reviews": "1" },
                                       body: JSON.stringify({ reviewer: name(), reviews: p }) });
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.status);
        window.SW_REVIEWS = await r.json(); const ids = new Set(p.map(x => x.id)); setPending(pending().filter(x => !ids.has(x.id)));
        msg = "saved " + new Date().toLocaleTimeString();
      } catch (e) { msg = "not saved: the review server did not answer (" + e.message + "); is it still running?"; }
      busy = false; cfg.onChange();
    } else if (mode === "folder") await saveToFolder(false);
  }
  function idb() { return new Promise((ok, no) => { const q = indexedDB.open("swreviews", 1); q.onupgradeneeded = () => q.result.createObjectStore("dirs"); q.onsuccess = () => ok(q.result); q.onerror = () => no(q.error); }); }
  async function folderHandle(h) {        // get (or, with h, remember) the campaign folder of this campaign
    if (!window.showDirectoryPicker) return null;
    try {
      const db = await idb(), tx = db.transaction("dirs", "readwrite"), st = tx.objectStore("dirs");
      if (h) { st.put(h, cfg.campaign || "?"); return h; }
      return await new Promise(ok => { const q = st.get(cfg.campaign || "?"); q.onsuccess = () => ok(q.result || null); q.onerror = () => ok(null); });
    } catch (e) { return null; }
  }
  const latest = rs => rs.length ? rs[rs.length - 1] : null;
  const stale = (r, stamp) => !!(stamp && r.data && r.data !== stamp && cfg.same(r));
  function badge(rs, stamp) {
    const r = latest(rs); if (!r) return "";
    const s = STATUS[r.status] || STATUS.note, st = stale(r, stamp);
    return `<span class="swr-badge swr-${s[1]}" title="${rs.length} review(s); latest: ${esc(r.reviewer)}, ${s[2]}${st ? " — the data changed after this review" : ""}${r.unsaved ? " (not saved yet)" : ""}">${s[0]}${rs.length > 1 ? " " + rs.length : ""}${st ? " *" : ""}</span>`;
  }
  function statusText(rs, stamp) { const r = latest(rs); if (!r) return ""; return `${(STATUS[r.status] || STATUS.note)[2]} (${r.reviewer}${stale(r, stamp) ? ", data changed since" : ""})`; }
  function where(t) { return [t.tool && t.tool !== "campaign" ? t.tool : "", t.dataset ? "data set " + t.dataset : "", t.view && t.view !== t.dataset ? t.view : "", t.page ? "report " + t.page : "", t.tab ? "tab " + t.tab : "", t.ext || (t.module ? "whole module" : "")].filter(Boolean).join(" · "); }
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
  // the campaign's review file, reviews.js: every review, one per line ("window.SW_REVIEWS = {...};")
  function storeText(reviews) {
    const head = JSON.stringify({ format: FORMAT, campaign: cfg.campaign, made: new Date().toISOString() }).slice(0, -1);
    return "window.SW_REVIEWS = " + head + ', "reviews": [\n' + reviews.map(r => JSON.stringify(r).replace(/<\//g, "<\\/")).join(",\n") + "\n]};\n";
  }
  function parseStore(text) { return JSON.parse(text.trim().replace(/^window\.SW_REVIEWS\s*=\s*/, "").replace(/;\s*$/, "")); }
  // a review stays 'not saved yet' (and on the page) until the campaign's reviews.js has it
  function prune() {
    const s = new Map(saved().map(r => [r.id, String(r.updated || "")]));
    const p = pending(), q = p.filter(r => !(s.has(r.id) && s.get(r.id) >= String(r.updated || "")));
    if (q.length !== p.length) setPending(q);
  }
  function download() {             // the campaign's reviews.js with the new reviews merged in
    const p = pending(); if (!p.length) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([storeText(merge(saved().concat(p)))], { type: "text/javascript" })); a.download = "reviews.js"; a.click();
    msg = "downloaded reviews.js: put it into the campaign folder";
    cfg.onChange();
    alert("Downloaded reviews.js: the campaign's review file with your " + p.length + " new review(s) in it.\n\n"
      + "Put it into the campaign folder (the one with index.html), replacing the reviews.js there, and reload the page: "
      + "the review(s) then show as saved, on every page.\n\n"
      + "If other people review the same campaign at the same time, merge instead of replacing:\n"
      + "  run_superWADERs --campaign DIR --reviews ~/Downloads/reviews.js\n\n"
      + "To save without this step: open the pages with  python3 review_server.py  (in the campaign folder).\n"
      + "Until then the review(s) stay on this page, marked 'not saved yet'.");
  }
  async function saveToFolder(ask) {     // ask: may show the folder picker (a click on 'Save to campaign folder')
    const p = pending(); if (!p.length || busy) return;
    busy = true;
    try {
      let dir = await folderHandle();
      if (dir) {
        let perm = await dir.queryPermission({ mode: "readwrite" });
        if (perm !== "granted") perm = await dir.requestPermission({ mode: "readwrite" }).catch(() => "denied");
        if (perm !== "granted") dir = null;
      }
      if (!dir && ask) {
        dir = await window.showDirectoryPicker({ id: "swcampaign", mode: "readwrite" });
        try { await dir.getFileHandle("campaign.json"); } catch (e) { alert("That is not a campaign folder (no campaign.json). Choose the folder that holds index.html and campaign.json."); dir = null; }
      }
      if (!dir) { busy = false; if (mode === "folder" && !ask) { msg = "click 'Save to campaign folder' to allow saving"; cfg.onChange(); } return; }
      let have = [];                    // merge into the folder's reviews.js as it is now (others may have saved)
      try { have = parseStore(await (await (await dir.getFileHandle("reviews.js")).getFile()).text()).reviews || []; } catch (e) {}
      const all_ = merge(have.concat(saved(), p));
      const w2 = await (await dir.getFileHandle("reviews.js", { create: true })).createWritable();
      await w2.write(storeText(all_)); await w2.close();
      window.SW_REVIEWS = { format: FORMAT, campaign: cfg.campaign, made: new Date().toISOString(), reviews: all_ }; const ids = new Set(p.map(x => x.id)); setPending(pending().filter(x => !ids.has(x.id)));
      await folderHandle(dir); mode = "folder"; msg = "saved into the campaign folder " + new Date().toLocaleTimeString();
    } catch (e) { if (e && e.name !== "AbortError") msg = "not saved (" + (e.message || e) + ")"; }
    busy = false; cfg.onChange();
  }
  function bar(el) {
    const n = pending().length, canFolder = !!window.showDirectoryPicker;
    const where = mode === "server" ? `<span class="swr-ok">saved straight into the campaign folder (review server)</span>`
      : mode === "folder" ? `<span class="swr-ok">saved into the campaign folder as you add them</span>`
      : mode === "probing" ? "" : `<span class="swr-hint">to save reviews automatically, open the pages with <code>python3 review_server.py</code> in the campaign folder${canFolder ? ", or use Save to campaign folder" : ""}</span>`;
    el.innerHTML = `<span class="swr-bar-t">Reviews: ${all().length} in this campaign${n ? ` · <b class="swr-unsaved">${n} not saved yet</b>` : ""}</span>`
      + (n && mode !== "server" ? (canFolder ? `<button class="swr-save">Save to campaign folder</button>` : "") + `<button class="swr-dl">Download</button>` : "")
      + (n && mode === "server" ? `<button class="swr-retry">Save now</button>` : "")
      + ` ${where}${msg ? ` <span class="swr-msgbar">· ${esc(msg)}</span>` : ""}<button class="swr-help">?</button>`;
    const q = s => el.querySelector(s);
    if (q(".swr-save")) q(".swr-save").onclick = () => saveToFolder(true);
    if (q(".swr-dl")) q(".swr-dl").onclick = download;
    if (q(".swr-retry")) q(".swr-retry").onclick = persist;
    q(".swr-help").onclick = () => alert("Reviews: give your name, a status (approved / concern / rejected / note) and a comment for a CCD (overview pages) or a tab of a report (module pages).\n\nAll reviews of the campaign are in one file, reviews.js in the campaign folder; each save adds to it.\n\nSaving as soon as a review is added:\n  - open the pages through the review server: in the campaign folder run\n      python3 review_server.py\n    (or run_superWADERs --campaign DIR --serve) and use the address it prints; any browser;\n  - Chrome / Edge on a double-clicked page: 'Save to campaign folder' asks for the folder once, then saves as you go.\nOtherwise a review stays on the page, 'not saved yet': 'Download' gives the updated reviews.js; put it into the campaign folder (replacing the old one) and reload, or run  run_superWADERs --campaign DIR --reviews ~/Downloads/reviews.js  (merges).\n\nA review records the data it refers to: when newer data arrive it is marked 'data changed since'.");
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
.swr-form button,#swrbar button{cursor:pointer}#swrbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:13px;margin:6px 0}.swr-ok{color:var(--pass,#2e9d5b)}.swr-hint{color:var(--muted,#888)}.swr-msgbar{color:var(--muted,#888)}
@media print{.swr-form,.swr-x,#swrbar button{display:none}}`;
    document.head.appendChild(s);
  }
  function init(c) { cfg = Object.assign(cfg, c); style(); prune(); if (mode === "probing") probe(); else folderHandle().then(h => { if (h) { mode = "folder"; cfg.onChange(); } }); }
  return { init, all, add, remove, form, bar, badge, statusText, latest, stale, STATUS };
})();
