#!/usr/bin/env python3
"""superWADERs reviews: who looked at which data, what they concluded, and why.

A review is written on a page of a campaign (an overview: one CCD; a module report: one tab, the whole module or one
CCD of it) by anyone who opens it: a name, a status (approved / concern / rejected / note) and a comment. It records
what it is about (target: tool, module, ext, ccd, view, page, tab), the data it refers to (data: a stamp of the
inputs; when newer data arrive the pages mark it 'data changed since') and the main values shown at the time.

Where they live: one file per campaign,
  <campaign>/reviews.js       every review of the campaign ("window.SW_REVIEWS = {format, campaign, made, reviews}"),
                              read by every page. Each save merges into it (the review server, a page's 'Save to
                              campaign folder', run_superWADERs --reviews FILE, reviews.py --add / --import), so
                              several people can review the same folder without losing each other's reviews.
  <campaign>/reviews.csv      the same as a table (one row per review), reviews.txt the latest review per CCD
  <campaign>/reviews/*.json   (earlier versions: one file per save) still read and merged in
Merging: by id, the latest 'updated' wins; a deleted review stays as a deletion record (so a deletion wins over an
older copy of the review in a downloaded file).

Saving from the pages. A page opened by double-click (file://) cannot write files in most browsers, so:
  review server   python3 review_server.py  in the campaign folder (a copy of this file, written by every run), or
                  run_superWADERs --campaign DIR --serve: serves the campaign's pages on http://localhost:<port>/ and
                  every review is saved into reviews.js the moment it is added (any browser)
  Chrome / Edge   on a double-clicked page: 'Save to campaign folder' asks for the folder once; later reviews are
                  then saved there as they are added
  any browser     'Download' gives the updated reviews.js: put it into the campaign folder (replace the old one), or
                  run_superWADERs --campaign DIR --reviews ~/Downloads/reviews.js  (merges: safe when others review too)

Standard library only (run before the analysis environment is looked for).

usage:
  python reviews.py DIR                              rebuild reviews.js / reviews.csv / reviews.txt
  python reviews.py DIR --serve [--port N]           the review server (also: python3 DIR/review_server.py)
  python reviews.py DIR --import FILE [FILE ...]     merge review files (reviews.js downloaded from a page) in
  python reviews.py DIR --add --reviewer NAME --status concern --module M1 [--ext EXT2] [--tool images]
                    --comment "..."                  write one review from the command line
(run_superWADERs --campaign DIR [--reviews FILE ...] does the same; no INPUT needed for that)."""
import argparse
import csv
import datetime
import json
import os
import re
import shutil
import sys
import uuid

FORMAT = "superWADERs-reviews-1"
STATUSES = ("approved", "concern", "rejected", "note")
TARGET_KEYS = ("tool", "module", "ext", "ccd", "acm", "dataset", "view", "page", "tab")


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def campaign_id(root):
    try:
        with open(os.path.join(root, "campaign.json")) as fh:
            return json.load(fh).get("id", "")
    except (OSError, ValueError):
        return ""


STORE = "reviews.js"
JS_PREFIX = "window.SW_REVIEWS = "


def read_file(path):
    """The reviews of one file: reviews.js (window.SW_REVIEWS = {...};), {format, reviews: [...]} or a bare list."""
    with open(path) as fh:
        text = fh.read().strip()
    if text.startswith("window."):
        text = text.split("=", 1)[1].strip().rstrip(";")
    j = json.loads(text)
    rs = j if isinstance(j, list) else j.get("reviews", [])
    if not isinstance(rs, list):
        raise ValueError("no list of reviews")
    return [r for r in rs if isinstance(r, dict) and r.get("id")]


def merge(reviews):
    """By id, the latest 'updated' wins; sorted by creation time. Deletion records are kept."""
    by = {}
    for r in reviews:
        o = by.get(r["id"])
        if o is None or str(r.get("updated") or "") >= str(o.get("updated") or ""):
            by[r["id"]] = r
    return sorted(by.values(), key=lambda r: str(r.get("created") or ""))


def load(root, log=None):
    """Every review of the campaign (deletion records included), merged: reviews.js and the files of reviews/
    (earlier versions)."""
    got = []
    store = os.path.join(root, STORE)
    if os.path.exists(store):
        try:
            got += read_file(store)
        except (OSError, ValueError) as e:
            if log:
                log(f"  {STORE}: not read ({e}); kept as {STORE}.unreadable")
            shutil.copyfile(store, store + ".unreadable")
    d = os.path.join(root, "reviews")
    for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if n.endswith(".json"):
            try:
                got += read_file(os.path.join(d, n))
            except (OSError, ValueError) as e:
                if log:
                    log(f"  reviews/{n}: not read ({e})")
    return merge(got)


def live(reviews):
    return [r for r in reviews if not r.get("deleted")]


def where(t):
    return " · ".join(str(x) for x in (t.get("tool") if t.get("tool") not in (None, "campaign") else None, t.get("module"),
                                       t.get("ext") or "whole module", t.get("ccd"),
                                       f"data set {t['dataset']}" if t.get("dataset") else None,
                                       t.get("view") if t.get("view") != t.get("dataset") else None,
                                       f"report {t['page']}" if t.get("page") else None,
                                       f"tab {t['tab']}" if t.get("tab") else None) if x)


def store_text(cid, reviews):
    """reviews.js: one review per line (readable, small diffs)."""
    head = json.dumps(dict(format=FORMAT, campaign=cid, made=_now()))[:-1]
    body = ",\n".join(json.dumps(r, ensure_ascii=False).replace("</", "<\\/") for r in reviews)
    return f"{JS_PREFIX}{head}, \"reviews\": [\n{body}\n]}};\n"


def write(root, log=print, add=()):
    """Merge add (new or changed reviews) into reviews.js, then reviews.csv and reviews.txt; returns the summary lines
    (also in reviews.txt)."""
    cid = campaign_id(root)
    allr = merge(load(root, log) + list(add))
    other = sorted({r.get("campaign") for r in allr if r.get("campaign") and cid and r.get("campaign") != cid})
    tmp = os.path.join(root, ".reviews.js.tmp")
    with open(tmp, "w") as fh:
        fh.write(store_text(cid, allr))
    os.replace(tmp, os.path.join(root, STORE))
    rs = live(allr)
    with open(os.path.join(root, "reviews.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["created", "updated", "reviewer", "status", "module", "ext", "ccd", "tool", "dataset", "view", "page", "tab",
                    "comment", "data", "values_then", "id"])
        for r in rs:
            t = r.get("target") or {}
            w.writerow([r.get("created"), r.get("updated"), r.get("reviewer"), r.get("status"), t.get("module"),
                        t.get("ext"), t.get("ccd"), t.get("tool"), t.get("dataset"), t.get("view"), t.get("page"), t.get("tab"),
                        r.get("comment"), r.get("data"), json.dumps(r.get("values")) if r.get("values") else "", r.get("id")])
    lines = summary_lines(rs)
    if other:
        lines.append(f"  note: {sum(1 for r in rs if r.get('campaign') in other)} review(s) were written in another campaign "
                     f"({', '.join(other)}); they are shown where their module / CCD matches")
    with open(os.path.join(root, "reviews.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    server = os.path.join(root, "review_server.py")      # the review server travels with the campaign folder
    me = os.path.abspath(__file__)
    if os.path.abspath(server) != me:
        with open(me) as fh:
            code = fh.read()
        if not os.path.exists(server) or open(server).read() != code:
            with open(server, "w") as fh:
                fh.write(code)
    for name, text in LAUNCHERS.items():
        f = os.path.join(root, name)
        if not os.path.exists(f):
            with open(f, "w", newline="") as fh:
                fh.write(text)
            os.chmod(f, 0o755)
    return lines


def summary_lines(rs):
    """Counts per status and the latest review of every CCD (or module) that has one."""
    if not rs:
        return ["reviews: none yet (write them on the pages: a name, a status and a comment per CCD or report tab)"]
    n = {s: sum(1 for r in rs if r.get("status") == s) for s in STATUSES}
    people = sorted({r.get("reviewer") or "?" for r in rs})
    L = [f"reviews: {len(rs)} ({', '.join(f'{v} {k}' for k, v in n.items() if v)}) by {', '.join(people)}  "
         f"(reviews.csv: all of them)"]
    latest = {}
    for r in rs:
        t = r.get("target") or {}
        latest[(str(t.get("module") or "?") + (f"@{t['dataset']}" if t.get("dataset") else ""), str(t.get("ext") or ""))] = r
    for (m, e), r in sorted(latest.items()):
        c = (r.get("comment") or "").replace("\n", " ")
        L.append(f"  {m:10s} {e or 'module':7s} {r.get('status', '?'):9s} {r.get('reviewer', '?')}, "
                 f"{str(r.get('updated') or '')[:10]}: {c[:90] + ('…' if len(c) > 90 else '')}  [{where(r.get('target') or {})}]")
    return L


def save(root, reviews, reviewer=""):
    """Merge these reviews into the campaign's reviews.js (and rebuild reviews.csv / .txt); returns its path."""
    write(root, log=lambda *a: None, add=reviews)
    return os.path.join(root, STORE)


def import_files(root, files, log=print):
    """Merge review files (reviews.js downloaded from a page, or older .json files) into reviews.js. Returns the
    number of reviews that were new or newer than the campaign's."""
    have = {r["id"]: str(r.get("updated") or "") for r in load(root)}
    add = []
    for f in files:
        try:
            rs = read_file(f)
        except (OSError, ValueError) as e:
            log(f"  {f}: not a review file ({e})")
            continue
        new = [r for r in rs if r["id"] not in have or str(r.get("updated") or "") > have[r["id"]]]
        log(f"  {os.path.basename(f)}: {len(rs)} review(s), {len(new)} new or changed")
        add += new
    if add:
        write(root, log=lambda *a: None, add=add)
    return len(add)


def new_review(root, reviewer, status, comment, target, data=None, values=None):
    if status not in STATUSES:
        raise ValueError(f"status must be one of {', '.join(STATUSES)}")
    now = _now()
    return dict(id=str(uuid.uuid4()), campaign=campaign_id(root), created=now, updated=now, reviewer=reviewer,
                status=status, comment=comment, target={k: v for k, v in target.items() if v is not None},
                data=data, values=values, page="command line")


def refresh(root, files=(), log=print):
    """Import files (if any), then rebuild reviews.js / .csv / .txt; returns the summary lines."""
    if files:
        import_files(root, files, log)
    return write(root, log)


# =============================================================================
# the review server: the campaign's pages over http://localhost, reviews saved as they are added
# =============================================================================
API = "/api/reviews"
MAX_POST = 2_000_000
INFO = "review_server_info.js"        # written while the server runs: pages opened by double-click save through it
LAUNCHERS = {                          # double-click: the review server for this folder, the browser opens on it
    "open_campaign.command": ("#!/bin/bash\n# macOS / Linux: double-click (macOS: the first time right-click > Open)\n"
                              "cd \"$(dirname \"$0\")\" && exec python3 review_server.py\n"),
    "open_campaign.bat": ("@echo off\r\nrem Windows: double-click\r\ncd /d \"%~dp0\"\r\n"
                          "py -3 review_server.py 2>nul || python review_server.py\r\npause\r\n"),
}


def downloaded(root, log=print):
    """Merge the review files a page downloaded (~/Downloads/reviews*.js, of this campaign) into reviews.js."""
    import glob
    cid = campaign_id(root)
    files = []
    for f in glob.glob(os.path.join(os.path.expanduser("~"), "Downloads", "reviews*.js")):
        try:
            if any(r.get("campaign") == cid for r in read_file(f)):
                files.append(f)
        except (OSError, ValueError):
            continue
    if files:
        n = import_files(root, files, log=lambda *a: None)
        if n:
            log(f"  {n} review(s) from downloaded file(s) merged in: {', '.join(os.path.basename(f) for f in files)}")


def serve(root, host="127.0.0.1", port=8765, open_browser=True, log=print):
    """Serve the campaign folder (its pages, read only) and the reviews: GET /api/reviews (all of them), POST
    /api/reviews {reviewer, reviews: [...]} (merged into reviews.js, reviews.csv / .txt rebuilt).
    host 127.0.0.1: this computer only; 0.0.0.0: everyone on the network can open the pages and add reviews."""
    import functools
    import http.server
    import secrets
    import threading
    lock = threading.Lock()
    token = secrets.token_hex(16)

    def allowed(h):
        """A request of a campaign page: same origin (the pages served here), or a page opened by double-click
        (origin 'null') that read this server's token from review_server_info.js in the campaign folder (a page of
        any other site cannot read that file: it is not served)."""
        origin = h.headers.get("Origin")
        if origin == "null" or h.headers.get("X-SW-Token"):
            return h.headers.get("X-SW-Token") == token
        return origin is None or origin.split("//")[-1] == h.headers.get("Host")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, fmt, *a):                     # quiet: only the saves are printed
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")   # reviews.js and the pages: always the current ones
            if self.headers.get("Origin") == "null":         # a page opened by double-click (file://)
                self.send_header("Access-Control-Allow-Origin", "null")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, X-SW-Reviews, X-SW-Token")
                self.send_header("Access-Control-Allow-Methods", "GET, POST")
                self.send_header("Access-Control-Allow-Private-Network", "true")
            super().end_headers()

        def do_OPTIONS(self):                               # the browser's check before a file:// page posts
            self.send_response(204 if self.path.split("?")[0] == API else 404)
            self.end_headers()

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split("?")[0] == API:
                if not allowed(self):
                    return self._json(403, dict(error="not from a page of this campaign"))
                return self._json(200, dict(format=FORMAT, campaign=campaign_id(root), server=True, reviews=load(root)))
            if os.path.basename(self.path.split("?")[0]) == INFO:   # the token stays in the folder
                return self._json(404, dict(error="not found"))
            return super().do_GET()

        def do_POST(self):
            if self.path.split("?")[0] != API:
                return self._json(404, dict(error="not found"))
            # only the pages themselves: a custom header and JSON (a page of another site cannot send these here)
            if (self.headers.get("X-SW-Reviews") != "1" or "json" not in (self.headers.get("Content-Type") or "")
                    or not allowed(self)):
                return self._json(403, dict(error="not from a page of this campaign"))
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= MAX_POST:
                return self._json(413, dict(error="too large"))
            try:
                j = json.loads(self.rfile.read(n))
                rs = [r for r in j.get("reviews", []) if isinstance(r, dict) and r.get("id")
                      and (r.get("deleted") or r.get("status") in STATUSES)]
                if not rs:
                    raise ValueError("no review")
            except (ValueError, AttributeError) as e:
                return self._json(400, dict(error=str(e)))
            with lock:
                save(root, rs)
                allr = load(root)
            log(f"  saved {len(rs)} review(s) by {j.get('reviewer') or '?'} into {STORE}")
            return self._json(200, dict(format=FORMAT, campaign=campaign_id(root), server=True, reviews=allr))

    write(root, log=log)
    downloaded(root, log)
    srv = None
    for p in range(port, port + 30):                        # the next free port
        try:
            srv = http.server.ThreadingHTTPServer((host, p), functools.partial(Handler, directory=root))
            break
        except OSError:
            continue
    if srv is None:
        sys.exit(f"reviews: no free port in {port}..{port + 29}")
    url = f"http://{'localhost' if host in ('127.0.0.1', 'localhost') else host}:{srv.server_address[1]}/index.html"
    info = os.path.join(root, INFO)
    with open(info, "w") as fh:
        fh.write("window.SW_SERVER = " + json.dumps(dict(url=f"http://127.0.0.1:{srv.server_address[1]}/", token=token,
                                                         campaign=campaign_id(root), started=_now())) + ";\n")
    log(f"review server for {root}\n  open {url}\n  reviews added on the pages are saved into {STORE} at once"
        f"{' (open to everyone on the network)' if host not in ('127.0.0.1', 'localhost') else ''}; Ctrl-C stops it")
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    import signal

    def stop(*a):
        raise KeyboardInterrupt
    for s in ("SIGTERM", "SIGHUP"):                         # closing the terminal window stops it cleanly too
        if hasattr(signal, s):
            signal.signal(getattr(signal, s), stop)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log("review server stopped")
    finally:
        srv.server_close()
        try:
            os.remove(info)
        except OSError:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))       # as review_server.py, the campaign is its own folder
    as_server = os.path.basename(__file__) == "review_server.py"
    ap.add_argument("campaign", metavar="DIR", nargs="?" if as_server else None, default=here if as_server else None,
                    help="the campaign folder (holds campaign.json)")
    ap.add_argument("--serve", action="store_true", default=as_server, help="run the review server")
    ap.add_argument("--port", type=int, default=8765, help="review server: first port to try (default 8765)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="review server: 127.0.0.1 (this computer, default) or 0.0.0.0 (everyone on the network)")
    ap.add_argument("--no-browser", action="store_true", help="review server: do not open the browser")
    ap.add_argument("--import", dest="files", nargs="+", default=[], metavar="FILE", help="review files to add")
    ap.add_argument("--add", action="store_true", help="write one review (with --reviewer, --status, --comment, target)")
    ap.add_argument("--reviewer")
    ap.add_argument("--status", choices=STATUSES, default="note")
    ap.add_argument("--comment", default="")
    for k in TARGET_KEYS:
        ap.add_argument("--" + k, default=None, help=f"target: {k}")
    a = ap.parse_args(argv)
    root = os.path.abspath(a.campaign)
    if not os.path.exists(os.path.join(root, "campaign.json")):
        sys.exit(f"reviews: {root} is not a campaign folder (no campaign.json)")
    if a.add:
        if not a.reviewer or not a.module:
            sys.exit("reviews: --add needs --reviewer and --module (and --ext for one CCD)")
        if a.status == "note" and not a.comment:
            sys.exit("reviews: a note needs --comment")
        t = {k: getattr(a, k) for k in TARGET_KEYS}
        t["tool"] = t["tool"] or "campaign"
        p = save(root, [new_review(root, a.reviewer, a.status, a.comment, t)], a.reviewer)
        print(f"saved {p}")
    print("\n".join(refresh(root, a.files)))
    if a.serve:
        serve(root, a.host, a.port, not a.no_browser)


if __name__ == "__main__":
    main()
