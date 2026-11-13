#!/usr/bin/env python3
"""superWADERs reviews: who looked at which CCD, what they concluded, and why, in one spreadsheet per campaign.

  <campaign>/reviews.csv   the review sheet. One row per CCD, filled in by run_superWADERs with what the CCD is
                           (module, ext, ccd, acm), its verdict and its main numbers. A reviewer opens it (Excel,
                           Numbers, LibreOffice, any editor), types reviewer, status and comment on the CCD's row and
                           saves it. A second review of the same CCD: copy the row. A review of one tool only: put the
                           tool (images, packet, dc_series, defects, sr) in 'tool'; of a whole module: leave 'ext' empty.
  run_superWADERs --campaign DIR   (with new data, or without INPUT: seconds) then
      - keeps every filled row as it is (the numbers it was written on stay with it), dates it, and marks it
        'newer data: yes' when the CCD's data changed since;
      - adds a row for every new CCD; refreshes the numbers of the rows not filled yet;
      - writes reviews.js (read by every page: the reviews show on the overviews and the module reports) and
        reviews.txt (the latest review of every CCD).

Columns: module, ext, ccd, acm, tool (all | images | packet | dc_series | defects | sr), data_set, verdict, values
(filled in) | reviewer, status (approved | concern | rejected | note), comment (yours) | date, newer_data, stamp
(filled in). The sheet can be saved as CSV with ',' or ';' by any program; it is written back with ','.

Standard library only (run before the analysis environment is looked for).

usage:  python reviews.py DIR      the same as run_superWADERs --campaign DIR without INPUT"""
import csv
import datetime
import hashlib
import json
import os
import sys

FORMAT = "superWADERs-reviews-2"
SHEET = "reviews.csv"
STORE = "reviews.js"
CCDS = os.path.join("records", "ccds.json")        # written by campaign.campaign_index: the CCDs and their numbers
STATUSES = ("approved", "concern", "rejected", "note")
TOOLS = ("all", "images", "packet", "dc_series", "defects", "sr")
COLS = ["module", "ext", "ccd", "acm", "tool", "data_set", "verdict", "values", "reviewer", "status", "comment", "date",
        "newer_data", "stamp"]
YOURS = ("reviewer", "status", "comment")
OLD = ("review_server.py", "review_server_info.js", "open_campaign.command", "open_campaign.bat")   # earlier versions


def _today():
    return datetime.date.today().isoformat()


def campaign_id(root):
    try:
        with open(os.path.join(root, "campaign.json")) as fh:
            return json.load(fh).get("id", "")
    except (OSError, ValueError):
        return ""


def status_of(s):
    """approved / concern / rejected / note from what was typed (first letters, symbols, a few words)."""
    s = str(s or "").strip().lower()
    for k in STATUSES:
        if s.startswith(k[:3]):
            return k
    if s in ("no", "x"):
        return "rejected"
    for k, words in (("approved", ("ok", "good", "pass", "yes", "✔", "✓", "accept")), ("rejected", ("fail", "bad", "✖", "reject")),
                     ("concern", ("warn", "check", "?", "▲", "problem", "issue"))):
        if any(s.startswith(w) for w in words):
            return k
    return "note"


def is_review(r):
    return any(str(r.get(k) or "").strip() for k in YOURS)


def read_sheet(root):
    """The rows of reviews.csv ([] if there is none); ',' or ';' separated, UTF-8 (with or without BOM) or Latin-1."""
    p = os.path.join(root, SHEET)
    if not os.path.exists(p):
        return []
    for enc in ("utf-8-sig", "cp1252"):
        try:
            with open(p, newline="", encoding=enc) as fh:
                text = fh.read()
            break
        except UnicodeDecodeError:
            continue
    first = text.split("\n", 1)[0]
    delim = ";" if first.count(";") > first.count(",") else ","
    rows = []
    for r in csv.DictReader(text.splitlines(), delimiter=delim):
        r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items() if k}
        if any(r.values()):
            rows.append({c: r.get(c, "") for c in COLS})
    return rows


def load_ccds(root):
    try:
        with open(os.path.join(root, CCDS)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return []


def _natural(s):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def update(root, log=print):
    """Bring reviews.csv up to date with the campaign's CCDs (keeping every review), then write reviews.js and
    reviews.txt. Returns the summary lines."""
    ccds = {(c["module"], c["ext"]): c for c in load_ccds(root)}
    rows = read_sheet(root) or legacy(root)
    filled, have = [], set()
    for r in rows:
        if not is_review(r):
            continue
        r["status"] = status_of(r["status"]) if r["status"] or r["comment"] else ""
        r["tool"] = r["tool"] if r["tool"] in TOOLS else "all"
        r["date"] = r["date"] or _today()
        c = ccds.get((r["module"], r["ext"]))
        if c is None and not r["ext"]:                       # a review of the whole module: any CCD of it
            c = next((x for (m, _), x in ccds.items() if m == r["module"]), None)
        now = (c or {}).get("stamp") if r["tool"] == "all" else ((c or {}).get("tools", {}).get(r["tool"]) or {}).get("stamp")
        if c is not None and r["ext"]:
            for k in ("ccd", "acm"):
                r[k] = r[k] or str(c.get(k) or "")
        if now and not r["stamp"]:                           # written on the data it shows now
            r["stamp"] = now
            if r["ext"]:                                     # the numbers of the CCD (not for a whole module)
                r["values"], r["verdict"] = r["values"] or c.get("values", ""), r["verdict"] or c.get("verdict", "")
        r["newer_data"] = "yes" if now and r["stamp"] and r["stamp"] != now else ""
        filled.append(r)
        have.add((r["module"], r["ext"]))
    blank = [dict(module=c["module"], ext=c["ext"], ccd=c.get("ccd") or "", acm=str(c.get("acm") or ""), tool="all",
                  data_set=c.get("dataset") or "", verdict=c.get("verdict", ""), values=c.get("values", ""), reviewer="",
                  status="", comment="", date="", newer_data="", stamp=c.get("stamp", ""))
             for k, c in ccds.items() if k not in have]
    rows = sorted(filled + blank, key=lambda r: (_natural(r["module"]), _natural(r["ext"]), not is_review(r), r["date"]))
    p = os.path.join(root, SHEET)
    try:
        tmp = p + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8-sig") as fh:      # the BOM: Excel reads it as UTF-8
            w = csv.DictWriter(fh, fieldnames=COLS)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, p)
    except OSError as e:                                     # e.g. open in Excel on Windows
        log(f"  {SHEET} could not be written ({e}); close it in the spreadsheet program and run again")
    for f in OLD:                                            # the review server of earlier versions is not used
        if os.path.exists(os.path.join(root, f)):
            os.remove(os.path.join(root, f))
    reviews = [to_review(r) for r in rows if is_review(r)]
    line = {}                                                # the sheet's line of each CCD (header: line 1)
    for i, r in enumerate(rows):
        line.setdefault(f"{r['module']} {r['ext']}", i + 2)
    js = dict(format=FORMAT, campaign=campaign_id(root), made=datetime.datetime.now().isoformat(timespec="minutes"),
              sheet=SHEET, lines=line, reviews=reviews)
    with open(os.path.join(root, STORE), "w") as fh:
        fh.write("window.SW_REVIEWS = " + json.dumps(js, ensure_ascii=False).replace("</", "<\\/") + ";\n")
    lines = summary_lines(reviews)
    with open(os.path.join(root, "reviews.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return lines


def legacy(root):
    """The reviews an earlier version kept in reviews.js (written on the pages), as rows of the sheet."""
    try:
        with open(os.path.join(root, STORE)) as fh:
            j = json.loads(fh.read().strip().split("=", 1)[1].strip().rstrip(";"))
    except (OSError, ValueError, IndexError):
        return []
    if j.get("format") != "superWADERs-reviews-1":
        return []
    by = {}
    for r in j.get("reviews", []):                           # by id, the latest version; deleted ones left out
        if r.get("id") and str(r.get("updated") or "") >= str((by.get(r["id"]) or {}).get("updated") or ""):
            by[r["id"]] = r
    out = []
    for r in by.values():
        if r.get("deleted"):
            continue
        t = r.get("target") or {}
        v = r.get("values")
        out.append(dict(module=t.get("module") or "", ext=t.get("ext") or "", ccd=t.get("ccd") or "",
                        acm=str(t.get("acm") or ""), tool="all" if t.get("tool") in (None, "campaign") else t["tool"],
                        data_set=t.get("dataset") or "", verdict=(v or {}).get("verdict", "") if isinstance(v, dict) else "",
                        values=", ".join(f"{k} {x}" for k, x in v.items() if k != "verdict") if isinstance(v, dict) else "",
                        reviewer=r.get("reviewer") or "", status=r.get("status") or "note",
                        comment=r.get("comment") or "",
                        date=str(r.get("created") or "")[:10], newer_data="", stamp=""))
    return out


def to_review(r):
    """A row of the sheet as the pages show it."""
    key = "|".join(r[k] for k in ("module", "ext", "tool", "reviewer", "comment", "date"))
    return dict(id=hashlib.sha1(key.encode()).hexdigest()[:12], reviewer=r["reviewer"] or "?", status=r["status"] or "note",
                comment=r["comment"], created=r["date"], updated=r["date"], data=r["stamp"] or None,
                values=r["values"] or None, verdict=r["verdict"] or None, newer_data=r["newer_data"] == "yes",
                target=dict(tool="campaign" if r["tool"] == "all" else r["tool"], module=r["module"], ext=r["ext"] or None,
                            ccd=r["ccd"] or None, acm=r["acm"] or None, dataset=r["data_set"] or None))


def summary_lines(rs):
    """Counts per status and the latest review of every CCD (or module) that has one."""
    if not rs:
        return [f"reviews: none yet (write them in {SHEET} in the campaign folder: reviewer, status, comment on a "
                f"CCD's row; then run_superWADERs --campaign DIR)"]
    n = {s: sum(1 for r in rs if r["status"] == s) for s in STATUSES}
    people = sorted({r["reviewer"] for r in rs})
    L = [f"reviews ({SHEET}): {len(rs)} ({', '.join(f'{v} {k}' for k, v in n.items() if v)}) by {', '.join(people)}"]
    latest = {}
    for r in rs:
        t = r["target"]
        latest[(t["module"], t["ext"] or "", t["tool"])] = r
    for (m, e, tool), r in sorted(latest.items(), key=lambda kv: (_natural(kv[0][0]), _natural(kv[0][1]))):
        c = (r["comment"] or "").replace("\n", " ")
        L.append(f"  {m:10s} {e or 'module':7s} {r['status']:9s} {r['reviewer']}, {r['updated']}"
                 f"{' (' + tool + ')' if tool != 'campaign' else ''}{' [newer data since]' if r['newer_data'] else ''}: "
                 f"{c[:90] + ('…' if len(c) > 90 else '')}")
    return L


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if len(a) != 1 or a[0].startswith("-"):
        sys.exit(__doc__)
    root = os.path.abspath(a[0])
    if not os.path.exists(os.path.join(root, "campaign.json")):
        sys.exit(f"reviews: {root} is not a campaign folder (no campaign.json)")
    print("\n".join(update(root)))


if __name__ == "__main__":
    main()
