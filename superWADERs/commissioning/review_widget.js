/* superWADERs reviews: the review panel shared by the overview pages and the module pages (inlined into both).

A review: {id, campaign, created, updated, reviewer, status: approved | concern | rejected | note, comment,
           target: {tool, module, ext, ccd, view, page, tab}, data: the data it refers to (a stamp),
           values: the main numbers shown when it was written, page}.
Where reviews live: one file per campaign,
  <campaign>/reviews.js        every review (window.SW_REVIEWS), read by every page; each save merges into it
A review shows on the page as soon as it is added (kept in this browser, localStorage, until it is in reviews.js) and
is written into the campaign folder at once:
  review helper                the campaign's review_server.py, started in the background by run_superWADERs: the page
                               came from it (http), or it runs for this folder (review_server_info.js gives its
                               address and token to pages opened by double-click); any browser
  Chrome / Edge, file://       without the helper: the page writes reviews.js itself (asks for the folder once)
  otherwise                    the review stays on the page and is written when the helper runs (checked every
                               20 s and when the page is shown again)
Merging: by id, the latest 'updated' wins; a deleted review is kept as a deletion record. */
const SWR = (() => {
  const STATUS = { approved: ["✔", "pass", "approved"], concern: ["▲", "warn", "concern"], rejected: ["✖", "fail", "rejected"],
                   note: ["✎", "na", "note"] };
  const FORMAT = "superWADERs-reviews-1";
  let cfg = { campaign: "", page: "", root: "", onChange: () => {}, same: () => true };   // same(r): r's stamp is comparable here
  let mode = (location.protocol.startsWith("http") || window.SW_SERVER) ? "probing" : "file", msg = "", busy = false;   // server | folder | file
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
    setPending(pending().concat([r])); cfg.onChange(); persist(true); return r;
  }
  function remove(id) {
    const p = pending();
    if (p.some(r => r.id === id)) setPending(p.filter(r => r.id !== id));
    else { const r = saved().find(x => x.id === id); if (r) setPending(p.concat([Object.assign({}, r, { deleted: true, updated: new Date().toISOString(), deleted_by: name() })])); }
    cfg.onChange(); persist(true);
  }
  // ---- saving: through the review server, or into the remembered campaign folder (Chrome / Edge)
  // the review server: this page came from it (http), or it runs for this folder (review_server_info.js, written
  // while it runs, gives its address and token to pages opened by double-click)
  const srv = () => (mode !== "file" && location.protocol.startsWith("http")) ? null : window.SW_SERVER || null;
  const api = () => srv() ? srv().url + "api/reviews" : new URL(cfg.root + "api/reviews", location.href).href;
  const hdr = () => Object.assign({ "X-SW-Reviews": "1" }, srv() ? { "X-SW-Token": srv().token } : {});
  async function probe() {
    try {
      const r = await fetch(api(), { cache: "no-store", headers: srv() ? { "X-SW-Token": srv().token } : {} }); if (!r.ok) throw new Error(r.status);
      const j = await r.json(); if (!j.server) throw new Error("no server");
      window.SW_REVIEWS = j; mode = "server"; prune();
    } catch (e) {
      mode = "file";
    }
    if (mode === "file" && await folderHandle()) mode = "folder";
    cfg.onChange(); persist();
  }
  async function persist(click) {          // click: right after Add / delete (the browser lets the page ask for a folder)
    const p = pending(); if (!p.length || busy) return;
    if (mode === "server") {
      busy = true;
      try {
        const r = await fetch(api(), { method: "POST", headers: Object.assign({ "Content-Type": "application/json" }, hdr()),
                                       body: JSON.stringify({ reviewer: name(), reviews: p }) });
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.status);
        window.SW_REVIEWS = await r.json(); const ids = new Set(p.map(x => x.id)); setPending(pending().filter(x => !ids.has(x.id)));
        msg = "";
      } catch (e) { mode = "file"; msg = ""; }
      busy = false; cfg.onChange();
    } else if (mode === "folder" || (mode === "file" && window.showDirectoryPicker && click)) await saveToFolder(click);
  }
  // the helper may start (or others add reviews) while the page is open: read review_server_info.js and reviews.js again
  function reload(src, done) {
    const s = document.createElement("script"); s.src = src + "?t=" + Date.now();
    s.onload = s.onerror = () => { s.remove(); done(); }; document.head.appendChild(s);
  }
  function refresh() {
    if (busy || location.protocol.startsWith("http")) { if (mode === "server") probe(); return; }
    reload(cfg.root + "reviews.js", () => reload(cfg.root + "review_server_info.js", () => {
      prune();
      if (window.SW_SERVER && mode !== "server") probe(); else cfg.onChange();
    }));
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
    return `<span class="swr-badge swr-${s[1]}" title="${rs.length} review(s); latest: ${esc(r.reviewer)}, ${s[2]}${st ? " — the data changed after this review" : ""}${r.unsaved ? " (on this page; not yet in the campaign folder)" : ""}">${s[0]}${rs.length > 1 ? " " + rs.length : ""}${st ? " *" : ""}</span>`;
  }
  function statusText(rs, stamp) { const r = latest(rs); if (!r) return ""; return `${(STATUS[r.status] || STATUS.note)[2]} (${r.reviewer}${stale(r, stamp) ? ", data changed since" : ""})`; }
  function where(t) { return [t.tool && t.tool !== "campaign" ? t.tool : "", t.dataset ? "data set " + t.dataset : "", t.view && t.view !== t.dataset ? t.view : "", t.page ? "report " + t.page : "", t.tab ? "tab " + t.tab : "", t.ext || (t.module ? "whole module" : "")].filter(Boolean).join(" · "); }
  function listHTML(rs, stamp) {
    if (!rs.length) return `<div class="swr-none">No review yet.</div>`;
    return rs.slice().reverse().map(r => { const s = STATUS[r.status] || STATUS.note;
      return `<div class="swr-item swr-${s[1]}"><div class="swr-head"><b>${s[0]} ${esc(s[2])}</b> · ${esc(r.reviewer || "?")} · ${esc(String(r.updated || r.created || "").slice(0, 16).replace("T", " "))}`
        + `${r.unsaved ? ' · <span class="swr-unsaved" title="kept on this page (this browser); written into the campaign folder as soon as the review helper runs">not yet in the campaign folder</span>' : ""}${stale(r, stamp) ? ' · <span class="swr-stale">data changed since</span>' : ""}`
        + `<button class="swr-x" data-id="${esc(r.id)}" title="delete this review">×</button></div>`
        + `${r.comment ? `<div class="swr-text">${esc(r.comment).replace(/\n/g, "<br>")}</div>` : ""}<div class="swr-where">${esc(where(r.target || {}))}`
        + `${r.values && stale(r, stamp) ? `<br>values then: ${esc(typeof r.values === "string" ? r.values : Object.entries(r.values).map(([k, v]) => k + " " + v).join(", "))}` : ""}</div></div>`; }).join("");
  }
  /* the list and the form: el = container; q(r) -> does review r belong here; target, stamp: of a new review;
     choices: optional [[label, target patch]] for what the review is about (e.g. one CCD or the whole module);
     values: the numbers shown now ({label: text}), kept with the review */
  function form(el, q, target, stamp, choices, values) {
    const rs = all().filter(q);
    const sel = choices && choices.length > 1 ? `<select class="swr-on">${choices.map((c, i) => `<option value="${i}">${esc(c[0])}</option>`).join("")}</select>` : "";
    const warn = rs.some(r => r.unsaved) && mode === "file" && !window.showDirectoryPicker
      ? `<div class="swr-how">Kept on this page. It goes into the campaign folder as soon as the review helper runs on this computer: it starts with every <code>run_superWADERs</code> run (or: <code>run_superWADERs --campaign &lt;folder&gt;</code>); no need to reload.</div>` : "";
    el.innerHTML = `<div class="swr-list">${listHTML(rs, stamp)}</div>${warn}
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
      if (!dir) { busy = false; cfg.onChange(); return; }
      let have = [];                    // merge into the folder's reviews.js as it is now (others may have saved)
      try { have = parseStore(await (await (await dir.getFileHandle("reviews.js")).getFile()).text()).reviews || []; } catch (e) {}
      const all_ = merge(have.concat(saved(), p));
      const w2 = await (await dir.getFileHandle("reviews.js", { create: true })).createWritable();
      await w2.write(storeText(all_)); await w2.close();
      window.SW_REVIEWS = { format: FORMAT, campaign: cfg.campaign, made: new Date().toISOString(), reviews: all_ }; const ids = new Set(p.map(x => x.id)); setPending(pending().filter(x => !ids.has(x.id)));
      await folderHandle(dir); mode = "folder"; msg = "";
    } catch (e) { if (e && e.name !== "AbortError") msg = "could not write into the folder (" + (e.message || e) + ")"; }
    busy = false; cfg.onChange();
  }
  function bar(el) {
    const n = pending().length;
    const where = mode === "server" ? `<span class="swr-ok">comments are saved into the campaign folder as you add them</span>`
      : mode === "folder" ? `<span class="swr-ok">comments are saved into the campaign folder as you add them</span>`
      : mode === "probing" ? "" : n ? `<span class="swr-unsaved">${n} comment(s) on this page only, for now: they go into the campaign folder when the review helper runs (it starts with run_superWADERs)</span>`
      : window.showDirectoryPicker ? `<span class="swr-hint">your first comment asks for the campaign folder once; then comments are saved into it as you add them</span>`
      : `<span class="swr-hint">comments are saved into the campaign folder while the review helper runs (it starts with run_superWADERs)</span>`;
    el.innerHTML = `<span class="swr-bar-t">Reviews: ${all().length}</span> ${where}${msg ? ` <span class="swr-msgbar">· ${esc(msg)}</span>` : ""}<button class="swr-help">?</button>`;
    el.querySelector(".swr-help").onclick = () => alert("Reviews: your name, a status (approved / concern / rejected / note) and a comment, for a CCD (overview pages) or a tab of a report (module pages). A review shows on the page at once.\n\nAll reviews of the campaign are in one file, reviews.js in the campaign folder, and every page shows them. A review is written there as soon as you add it:\n  - through the review helper, which run_superWADERs starts in the background on this computer (any browser);\n  - in Chrome / Edge also without it: the first review asks for the campaign folder once.\nOtherwise it stays on this page and is written when the helper runs (run_superWADERs --campaign <folder>).\n\nA review records the data it refers to: when newer data arrive it is marked 'data changed since'.");
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
.swr-form button,#swrbar button{cursor:pointer}#swrbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:13px;margin:6px 0}.swr-ok{color:var(--pass,#2e9d5b)}.swr-how{font-size:12px;color:var(--muted,#888);margin-top:6px;line-height:1.45}.swr-warnbox{flex-basis:100%;background:var(--warnbg,#fbf0d9);color:var(--fg,#111);border:1px solid var(--warn,#d68a00);border-radius:6px;padding:6px 9px}.swr-hint{color:var(--muted,#888)}.swr-msgbar{color:var(--muted,#888)}
@media print{.swr-form,.swr-x,#swrbar button{display:none}}`;
    document.head.appendChild(s);
  }
  function init(c) {
    cfg = Object.assign(cfg, c); style(); prune();
    if (mode === "probing") probe(); else folderHandle().then(h => { if (h) { mode = "folder"; cfg.onChange(); persist(); } });
    setInterval(() => { if (!document.hidden) refresh(); }, 20000);           // others' reviews; the helper starting
    document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
  }
  return { init, all, add, remove, form, bar, badge, statusText, latest, stale, STATUS };
})();
