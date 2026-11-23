"""Continuous-readout stitching (run_superWADERs --stitch): consecutive images of one readout joined row by row into
one image per group, so an event read across an image boundary is one cluster and only the ends of the whole readout
lose edge rows.

    groups = group_images(paths, mode="time")      # [[path, ...], ...] consecutive images of one setting
    made = write_stitched(groups, out_dir)          # [(stitched path, members)], the members' row offsets in a JSON

Grouping: images of the same ACM, size, binning (NPBIN x NSBIN), number of skips (NDCM) and exposure. mode "time":
in readout order (DATEINI), joined while the next image starts within max_gap s (default TOL_S) of the end of the
previous one (DATEEND; real L2 runs of consecutive images have ~58 s between images: not continuous, not joined);
mode "order": the list order, every image of a setting joined (trusted continuous; e.g. simulations). Groups of one
image stay as they are. Rows of a vertical overscan (PHYSROW / NPBIN onward) are left out of the stack.
The stitched file: the first member's headers, NROW = all rows, DATEEND of the last member, STITCHN = members, the
name of the first member with 'stitch<N>_' before the ACM id (its ACM and image number stay readable). Its
<name>.stitch.json lists every member with its first row in the stitched image: a cluster at row y belongs to the
member with row0 <= y < row0 + nrows (member_of()).
"""
import json
import math
import os
import re
from datetime import datetime

import numpy as np

TOL_S = 5.0                # s: the next image must start within this of the previous one's end (--stitch-gap)
ACM_RE = re.compile(r"_(\d+_\d{8}_\d{6}_\d+)$")


def _time(v):
    try:
        return datetime.fromisoformat(str(v).replace("Z", "")).timestamp()
    except (TypeError, ValueError):
        return None


def _info(path):
    from astropy.io import fits
    with fits.open(path) as h:
        h0 = h[0].header
        shapes = tuple(tuple(e.data.shape) for e in h[1:] if e.data is not None)
        acm = re.search(r"_(\d{2,4})_\d{8}_\d{6}", os.path.basename(path))
        key = (h0.get("ACMID") or (acm.group(1) if acm else None), shapes, h0.get("NPBIN", 1), h0.get("NSBIN", 1),
               h0.get("NDCM"), h0.get("EXPOSURE"))
        return dict(path=path, key=key, t0=_time(h0.get("DATEINI")), t1=_time(h0.get("DATEEND")))


def group_images(paths, mode="time", max_gap=TOL_S):
    """lists of consecutive images of one readout (see the module doc); single images as lists of one"""
    infos = [_info(p) for p in paths]
    by_key = {}
    for i in infos:
        by_key.setdefault(i["key"], []).append(i)
    groups = []
    for items in by_key.values():
        if mode == "order":
            groups.append([i["path"] for i in items])
            continue
        timed = sorted([i for i in items if i["t0"] is not None and i["t1"] is not None], key=lambda i: i["t0"])
        groups += [[i["path"]] for i in items if i["t0"] is None or i["t1"] is None]       # no times: not joined
        cur = []
        for i in timed:
            if cur:
                prev = cur[-1]
                if abs(i["t0"] - prev["t1"]) > max_gap:
                    groups.append([c["path"] for c in cur])
                    cur = []
            cur.append(i)
        if cur:
            groups.append([c["path"] for c in cur])
    return groups


def stitched_name(first, n):
    stem = os.path.basename(first).split(".")[0]
    m = ACM_RE.search(stem)
    return (stem[:m.start()] + f"_stitch{n}" + stem[m.start():]) if m else f"{stem}_stitch{n}"


def write_stitched(groups, out_dir, log=print):
    """stitched FITS (+ .stitch.json) for every group of >= 2 images in out_dir; returns [(path, members)] for all
    groups (single images: their own path)"""
    from astropy.io import fits
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for g in groups:
        if len(g) < 2:
            out.append((g[0], [dict(file=g[0], row0=0, nrows=None)]))
            continue
        with fits.open(g[0]) as h:
            h0 = h[0].header.copy()
            hx = [e.header.copy() for e in h[1:] if e.data is not None]
        try:
            n_act = int(math.ceil(float(h0["PHYSROW"]) / float(h0.get("NPBIN", 1) or 1)))
        except (KeyError, TypeError, ValueError):
            n_act = None
        stacks, members, row0 = [[] for _ in hx], [], 0
        for p in g:
            with fits.open(p) as h:
                exts = [e.data for e in h[1:] if e.data is not None]
                rows = exts[0].shape[0] if n_act is None else min(exts[0].shape[0], n_act)
                for k, d in enumerate(exts):
                    stacks[k].append(np.asarray(d[:rows], np.float32))
                members.append(dict(file=os.path.abspath(p), row0=row0, nrows=rows, dateini=h[0].header.get("DATEINI"),
                                    dateend=h[0].header.get("DATEEND")))
                row0 += rows
        with fits.open(g[-1]) as h:
            h0["DATEEND"] = h[0].header.get("DATEEND", h0.get("DATEEND"))
        h0["NROW"] = row0
        h0["STITCHN"] = (len(g), "images joined row by row (run_superWADERs --stitch)")
        name = stitched_name(g[0], len(g))
        path = os.path.join(out_dir, name + ".fits")
        hdus = [fits.PrimaryHDU(header=h0)] + [fits.ImageHDU(np.vstack(s), header=hh) for s, hh in zip(stacks, hx)]
        fits.HDUList(hdus).writeto(path, overwrite=True)
        json.dump(dict(stitched=path, members=members), open(os.path.join(out_dir, name + ".stitch.json"), "w"), indent=1)
        log(f"  stitched {len(g)} images -> {name}.fits ({row0} rows)")
        out.append((path, members))
    return out


def member_of(y, members):
    """the member (dict) holding row y of a stitched image, and the row within it"""
    for m in members:
        if m["nrows"] is not None and m["row0"] <= y < m["row0"] + m["nrows"]:
            return m, y - m["row0"]
    return None, None
