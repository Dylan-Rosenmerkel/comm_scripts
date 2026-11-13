#!/usr/bin/env python3
"""superWADERs reviews: who looked at which data, what they concluded, and why.

A review is written on a page of a campaign (an overview: one CCD; a module report: one tab, the whole module or one
CCD of it) by anyone who opens it: a name, a status (approved / concern / rejected / note) and a comment. It records
what it is about (target: tool, module, ext, ccd, view, page, tab), the data it refers to (data: a stamp of the
inputs; when newer data arrive the pages mark it 'data changed since') and the main values shown at the time.

Where they live (the pages are plain files, there is no server):
  <campaign>/reviews/*.json   one file per save: {format, campaign, saved, reviews: [...]}. Each save is its own file,
                              so several people can review the same (shared) folder without overwriting each other.
  <campaign>/reviews.js       all of them merged: loaded by every page of the campaign (written here, by
                              run_superWADERs, or by a page's 'Save to campaign folder' button)
  <campaign>/reviews.csv      the same as a table (one row per review), reviews.txt a summary per CCD
Merging: by id, the latest 'updated' wins; a deleted review stays as a deletion record (so a deletion in one file
wins over the review in another).

Standard library only (run before the analysis environment is looked for).

usage:
  python reviews.py DIR                              rebuild reviews.js / reviews.csv / reviews.txt from DIR/reviews/
  python reviews.py DIR --import FILE [FILE ...]     add review files (downloaded from a page) and rebuild
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
TARGET_KEYS = ("tool", "module", "ext", "ccd", "acm", "view", "page", "tab")


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def campaign_id(root):
    try:
        with open(os.path.join(root, "campaign.json")) as fh:
            return json.load(fh).get("id", "")
    except (OSError, ValueError):
        return ""


def read_file(path):
    """The reviews of one file: {format, reviews: [...]} or a bare list."""
    with open(path) as fh:
        j = json.load(fh)
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
    """Every review of the campaign (deletion records included), merged."""
    d = os.path.join(root, "reviews")
    got = []
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
                                       t.get("ext") or "whole module", t.get("ccd"), t.get("view"),
                                       f"report {t['page']}" if t.get("page") else None,
                                       f"tab {t['tab']}" if t.get("tab") else None) if x)


def write(root, log=print):
    """reviews.js, reviews.csv and reviews.txt from reviews/; returns the summary lines (also in reviews.txt)."""
    cid = campaign_id(root)
    allr = load(root, log)
    other = sorted({r.get("campaign") for r in allr if r.get("campaign") and cid and r.get("campaign") != cid})
    js = dict(format=FORMAT, campaign=cid, made=_now(), reviews=allr)
    tmp = os.path.join(root, ".reviews.js.tmp")
    with open(tmp, "w") as fh:
        fh.write("window.SW_REVIEWS = " + json.dumps(js, separators=(",", ":")).replace("</", "<\\/") + ";\n")
    os.replace(tmp, os.path.join(root, "reviews.js"))
    rs = live(allr)
    with open(os.path.join(root, "reviews.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["created", "updated", "reviewer", "status", "module", "ext", "ccd", "tool", "view", "page", "tab",
                    "comment", "data", "values_then", "id"])
        for r in rs:
            t = r.get("target") or {}
            w.writerow([r.get("created"), r.get("updated"), r.get("reviewer"), r.get("status"), t.get("module"),
                        t.get("ext"), t.get("ccd"), t.get("tool"), t.get("view"), t.get("page"), t.get("tab"),
                        r.get("comment"), r.get("data"), json.dumps(r.get("values")) if r.get("values") else "", r.get("id")])
    lines = summary_lines(rs)
    if other:
        lines.append(f"  note: {sum(1 for r in rs if r.get('campaign') in other)} review(s) were written in another campaign "
                     f"({', '.join(other)}); they are shown where their module / CCD matches")
    with open(os.path.join(root, "reviews.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
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
        latest[(str(t.get("module") or "?"), str(t.get("ext") or ""))] = r
    for (m, e), r in sorted(latest.items()):
        c = (r.get("comment") or "").replace("\n", " ")
        L.append(f"  {m:10s} {e or 'module':7s} {r.get('status', '?'):9s} {r.get('reviewer', '?')}, "
                 f"{str(r.get('updated') or '')[:10]}: {c[:90] + ('…' if len(c) > 90 else '')}  [{where(r.get('target') or {})}]")
    return L


def _file_name(reviewer):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    who = re.sub(r"[^\w-]+", "_", reviewer or "anonymous")
    return f"reviews_{who}_{stamp}_{uuid.uuid4().hex[:6]}.json"


def save(root, reviews, reviewer=""):
    """A new file in reviews/ with these reviews; returns its path."""
    d = os.path.join(root, "reviews")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, _file_name(reviewer))
    with open(p, "w") as fh:
        json.dump(dict(format=FORMAT, campaign=campaign_id(root), saved=_now(), reviewer=reviewer, reviews=reviews), fh, indent=1)
    return p


def import_files(root, files, log=print):
    """Copy review files (downloaded from a page) into reviews/; a file already there (same content) is skipped."""
    d = os.path.join(root, "reviews")
    os.makedirs(d, exist_ok=True)
    n = 0
    for f in files:
        try:
            rs = read_file(f)
        except (OSError, ValueError) as e:
            log(f"  {f}: not a review file ({e})")
            continue
        dest = os.path.join(d, os.path.basename(f))
        if os.path.exists(dest):
            with open(dest, "rb") as a, open(f, "rb") as b:
                if a.read() == b.read():
                    log(f"  {os.path.basename(f)}: already in reviews/")
                    continue
            dest = os.path.join(d, _file_name(os.path.splitext(os.path.basename(f))[0]))
        shutil.copyfile(f, dest)
        log(f"  {os.path.basename(f)}: {len(rs)} review(s) -> reviews/{os.path.basename(dest)}")
        n += len(rs)
    return n


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign", metavar="DIR", help="the campaign folder (holds campaign.json)")
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


if __name__ == "__main__":
    main()
