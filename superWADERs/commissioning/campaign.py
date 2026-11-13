"""A commissioning campaign: one folder that grows packet by packet (run_superWADERs --campaign DIR).

Data arrive in packets, so every input is processed ONCE and its results are kept; each run adds the new inputs and
rebuilds every report and overview page from everything the campaign holds, without fitting earlier data again.

    DIR/campaign.json           the ledger: every input (absolute path, size + modification time, kind, the tools it
                                was given to), and every run (date, command, new / skipped inputs, summary lines)
    DIR/criteria.json           the criteria the pages are judged with (a copy of --criteria, else of
                                commissioning/criteria_default.json); verdicts are made when the pages are written,
                                so a changed criteria file re-judges everything without new processing
    DIR/records/images/<image>.json.gz   per image: the native summary (fits, gain choice, clustering), the
                                CCD metrics (ccd_metrics.py), crosstalk, the dark-current fit cards, and the parts a
                                tool asked for (packet viewer, serial-register streaks)
    DIR/records/packet/<file>.json.gz    per level-0 trace / PSD trace of a packet
    DIR/defects/<group>/group.json       per defect-map group (images of one ACM and size)
An input is new when its path is not in the ledger, or its size / modification time changed (then it is processed
again). An input given to a further tool (e.g. an image of a packet later also in a --dc-series list) gets only the
part that tool needs.

Pages: every tool writes an overview (DIR/<tool>/index.html: module x CCD grid with verdicts, what needs attention,
distributions against the limits, the full table, the criteria) from overview_template.html, and DIR/index.html
combines them per CCD.
"""
import datetime
import gzip
import posixpath
import html
import json
import math
import os
import re
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CRITERIA = os.path.join(HERE, "criteria_default.json")
TEMPLATE = os.path.join(HERE, "overview_template.html")
REVIEW_JS = os.path.join(HERE, "review_widget.js")       # the review panel shared by the overviews and the module pages
ACTIVE = None                                            # the Campaign being written (pages find reviews.js from it)
VERSION = "campaign-1 (2026-11-12)"


# =============================================================================
# the campaign folder
# =============================================================================
def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _atomic_write(path, text, mode="w"):
    tmp = path + ".tmp"
    with open(tmp, mode) as fh:
        fh.write(text)
    os.replace(tmp, path)


def jsonable(o):
    """JSON-safe copy: numpy scalars / arrays -> Python, non-finite floats -> None, tuples -> lists."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, (datetime.datetime, datetime.date)):
        return o.isoformat()
    return o


class Campaign:
    """The ledger and the records of one campaign folder."""

    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.path = os.path.join(self.root, "campaign.json")
        if os.path.exists(self.path):
            with open(self.path) as fh:
                self.data = json.load(fh)
        else:
            self.data = dict(version=VERSION, created=_now(), inputs={}, runs=[])
        # the campaign's id: reviews carry it, so a review file of another campaign is recognised
        self.data.setdefault("id", f"{os.path.basename(self.root)}@{self.data.get('created', _now())[:19]}")
        global ACTIVE
        ACTIVE = self

    # ---- inputs
    @staticmethod
    def stat_key(path):
        st = os.stat(path)
        return [int(st.st_size), int(st.st_mtime_ns)]

    def entry(self, path):
        return self.data["inputs"].get(os.path.abspath(path))

    def current(self, path):
        """True if the input is in the ledger and unchanged."""
        e = self.entry(path)
        try:
            return bool(e) and e["key"] == self.stat_key(path)
        except OSError:
            return False

    def has(self, path, tool):
        return self.current(path) and tool in self.entry(path).get("tools", [])

    def add(self, path, tool, kind=None, stem=None, dataset=None):
        """dataset: the data set of this input for this tool (--dataset; None keeps the one it had, "" = default)."""
        p = os.path.abspath(path)
        e = self.data["inputs"].get(p)
        if e is None or e.get("key") != self.stat_key(p):
            e = self.data["inputs"][p] = dict(key=self.stat_key(p), tools=[], added=_now())
        if tool not in e["tools"]:
            e["tools"].append(tool)
        if kind:
            e["kind"] = kind
        if stem:
            e["stem"] = stem
        if dataset is not None:
            if dataset:
                e.setdefault("datasets", {})[tool] = dataset
            else:
                e.get("datasets", {}).pop(tool, None)

    def inputs(self, tool):
        return [p for p, e in sorted(self.data["inputs"].items()) if tool in e.get("tools", [])]

    def dataset(self, path, tool):
        return ((self.entry(path) or {}).get("datasets") or {}).get(tool, "")

    def dataset_of_stem(self, tool):
        """{stem: data set} of the tool's image inputs."""
        return {e["stem"]: (e.get("datasets") or {}).get(tool, "") for e in self.data["inputs"].values()
                if tool in e.get("tools", []) and e.get("stem")}

    def dataset_order(self, tool):
        """The tool's data sets, newest first (by when their first input came in)."""
        first = {}
        for e in self.data["inputs"].values():
            if tool in e.get("tools", []):
                ds = (e.get("datasets") or {}).get(tool, "")
                first[ds] = min(first.get(ds, e.get("added", "")), e.get("added", ""))
        return sorted(first, key=lambda d: first[d], reverse=True)

    def stems(self, tool):
        return [e["stem"] for p, e in sorted(self.data["inputs"].items()) if tool in e.get("tools", []) and e.get("stem")]

    # ---- records
    def record_file(self, kind, name):
        d = os.path.join(self.root, "records", kind)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, re.sub(r"[^\w.+-]", "_", name) + ".json.gz")

    def save_record(self, kind, name, obj):
        p = self.record_file(kind, name)
        _atomic_write(p, gzip.compress(json.dumps(jsonable(obj), separators=(",", ":")).encode(), 5), "wb")

    def load_record(self, kind, name):
        p = self.record_file(kind, name)
        if not os.path.exists(p):
            return None
        with gzip.open(p, "rt") as fh:
            return json.load(fh)

    def has_record(self, kind, name):
        return os.path.exists(self.record_file(kind, name))

    # ---- runs
    def add_run(self, argv, tools, new, skipped, lines=()):
        self.data["runs"].append(dict(date=_now(), command=" ".join(argv), tools=sorted(tools), new=len(new),
                                      skipped=len(skipped), new_inputs=[os.path.basename(p) for p in new][:200],
                                      lines=list(lines)))

    def save(self):
        self.data["updated"] = _now()
        _atomic_write(self.path, json.dumps(jsonable(self.data), indent=1))


# =============================================================================
# criteria and verdicts
# =============================================================================
def load_criteria(root, path=None):
    """The campaign's criteria: --criteria FILE (copied into the campaign), else the campaign's own copy, else the
    default file (copied). Returns the criteria dict."""
    camp = os.path.join(root, "criteria.json")
    if path:
        shutil.copyfile(path, camp)
    elif not os.path.exists(camp):
        shutil.copyfile(DEFAULT_CRITERIA, camp)
    with open(camp) as fh:
        return json.load(fh)


def section(crit, tool):
    """The criteria of one tool, with the file's name / version / note (for the page)."""
    s = dict(crit.get(tool) or {})
    s.setdefault("metrics", {})
    s.update(name=crit.get("name"), version=crit.get("version"), note=crit.get("note"))
    return s


def fmt_value(v, c=None):
    c = c or {}
    if isinstance(v, bool) or v is None:
        return str(v)
    d = c.get("digits", 3)
    s = f"{v:.{d}g}" if (v != 0 and (abs(v) < 1e-2 or abs(v) >= 1e4)) else f"{v:.{max(d, 0)}f}".rstrip("0").rstrip(".") or "0"
    return s + (f" {c['unit']}" if c.get("unit") else "")


def judge(values, sec, flags=(), status=None):
    """(verdict, reasons, levels) of one CCD. values: {metric: number}; sec: the tool's criteria section; flags:
    [(flag key, default level, text, group label)] of the non-numeric checks (their level from sec['flag_levels'];
    'note' shows it without changing the verdict, 'off' drops it); status: None = processed, else why there is no data ('na' verdict).
    reasons: [[level, key, text, group label]], fail first; levels: {metric: pass / warn / fail}."""
    reasons, levels = [], {}
    for key, c in (sec.get("metrics") or {}).items():
        v = values.get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            continue
        x = abs(v) if c.get("abs") else v
        below = c.get("direction") == "below"
        lv = "pass"
        for level in ("fail", "warn"):
            lim = c.get(level)
            if lim is not None and (x < lim if below else x > lim):
                lv = level
                reasons.append([level, key, f"{c.get('label', key)} {fmt_value(v, c)} {'<' if below else '>'} {level} "
                                            f"limit {fmt_value(lim, c)}", c.get("label", key)])
                break
        if c.get("warn") is not None or c.get("fail") is not None:
            levels[key] = lv
    fl = sec.get("flag_levels") or {}
    for key, default, text, label in flags:
        lv = fl.get(key, default)
        if lv in ("warn", "fail", "note"):          # note: shown with the CCD, not part of the verdict
            reasons.append([lv, key, text, label])
    reasons.sort(key=lambda r: {"fail": 0, "warn": 1}.get(r[0], 2))
    if status:
        return "na", reasons, levels
    verdict = "fail" if any(r[0] == "fail" for r in reasons) else "warn" if any(r[0] == "warn" for r in reasons) else "pass"
    return verdict, reasons, levels


def counts(rows):
    n = {"pass": 0, "warn": 0, "fail": 0, "na": 0}
    for r in rows:
        n[r.get("verdict", "na")] = n.get(r.get("verdict", "na"), 0) + 1
    return n


# =============================================================================
# the overview page
# =============================================================================
SHORT = {"read_noise_e": "noise", "gain": "gain", "dc_rate_e_per_pix_day": "DC", "hot_columns": "hot cols",
         "noise_single_adu": "1-skip noise", "psd_sigma_cds_adu": "σ CDS", "rate_last": "last rate", "rate_median": "median rate",
         "plateau": "plateau", "images": "images", "bright_columns": "bright cols", "dark_columns": "dark cols",
         "hot_pixels": "hot px", "cold_pixels": "cold px", "streaks_per_1000_reads": "streaks/1k reads",
         "dc_median": "DC", "glow_index": "glow", "crosstalk_max": "xtalk", "best_rate": "rate",
         "dc_rate_median": "DC rate"}


# =============================================================================
# data sets: several entries of the same module for one tool (two packets, two dark-current runs, ...)
# =============================================================================
DS_SEP = "@"                     # a module of a data set is rendered as "<module>@<data set>" (pages: DM01@packet2_...)


def dataset_name(name):
    return re.sub(r"[^\w.+-]+", "-", str(name or "")).strip("-")


def ds_module(module, ds):
    return f"{module}{DS_SEP}{ds}" if ds else module


def split_module(module):
    m, _, ds = str(module).rpartition(DS_SEP) if DS_SEP in str(module) else (module, "", "")
    return m, ds


def split_datasets(views, order):
    """A tool's views with the data sets apart: rows of '<module>@<data set>' go to a view of their data set (newest
    data set first, as order), the module name without the data set; a view of a data set has dataset=<name> ("" for
    the data without --dataset). Without data sets the views are unchanged."""
    if not any(DS_SEP in str(r.get("module", "")) for v in views for r in v["ccds"]):
        return views
    out = []
    for ds in order:
        for v in views:
            rs = []
            for r in v["ccds"]:
                m, d = split_module(r["module"])
                if d == ds:
                    rs.append(dict(r, module=m, dataset=ds))
            if rs:
                name = ds or "without data set"
                out.append(dict(v, name=f"{name} · {v['name']}" if len(views) > 1 else name, ccds=rs, dataset=ds,
                                note=f"data set {ds}" if ds else "inputs given without --dataset"))
    return out


def review_context(path, tool, **extra):
    """What a page needs for its reviews: the tool, the campaign id and the relative path from the page to the
    campaign folder (where reviews.js is)."""
    root = ""
    if ACTIVE is not None:
        rel = os.path.relpath(ACTIVE.root, os.path.dirname(os.path.abspath(path)))
        root = "" if rel == "." else rel.replace(os.sep, "/") + "/"
    return dict(tool=tool, campaign=ACTIVE.data["id"] if ACTIVE is not None else "", root=root, **extra)


def page_scripts(page, root):
    """Fill the review placeholders of a page template: the campaign's reviews.js and the review widget."""
    with open(REVIEW_JS) as fh:
        widget = fh.read()
    return page.replace("__REVIEWS_SRC__", html.escape(root + "reviews.js")).replace("__REVIEW_WIDGET__", widget)


def write_overview(path, data):
    """One overview page (overview_template.html) from data: title, sub, back [[label, href]], criteria (a section),
    fields {key: {label, unit}}, sections [[name, [keys]]], cell [keys], views [{name, note, ccds: [row]}],
    columns [keys], definitions [[term, text]], matrices / matrix_title / matrix_note (optional), tools (campaign
    index), flag_labels, csv_name. row: module, acm, ext, ccd, verdict, status, reasons, levels, values, links,
    history {head, rows}, when, tools (campaign index). Returns the views as written (data sets apart)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data["review"] = review_context(path, (data.get("review") or {}).get("tool", "campaign"))
    if ACTIVE is not None and data["review"]["tool"] != "campaign":      # several data sets: one view each
        data["views"] = split_datasets(data.get("views") or [], ACTIVE.dataset_order(data["review"]["tool"]))
    for k, f in (data.get("fields") or {}).items():
        f.setdefault("short", SHORT.get(k.split(".")[-1], f.get("label", k)))
    body = json.dumps(jsonable(data), separators=(",", ":")).replace("</", "<\\/")
    with open(TEMPLATE) as fh:
        page = fh.read()
    page = page_scripts(page, data["review"]["root"])
    _atomic_write(path, page.replace("__TITLE__", html.escape(data.get("title", "overview"))).replace("__DATA__", body))
    return data["views"]


def sub_line(camp, extra=""):
    runs = camp.data.get("runs", [])
    return (f"campaign {os.path.basename(camp.root)} · {len(camp.data.get('inputs', {}))} input file(s) in "
            f"{len(runs)} run(s) · updated {_now()}" + (f" · {extra}" if extra else ""))


# =============================================================================
# images: fits and CCD quality (one row per CCD and image setting; history over the images)
# =============================================================================
IMAGE_FIELDS = {
    "gain": ("gain", "ADU/e-"), "gain_err": ("gain error", "ADU/e-"), "gain_vs_module_pct": ("gain vs module median", "%"),
    "gain_vs_nominal_pct": ("gain vs nominal", "%"), "gain_measured": ("gain measured on this CCD", ""),
    "read_noise_e": ("read noise", "e-"), "read_noise_gain": ("read noise converted with the gain", ""), "noise_adu": ("pedestal noise", "ADU"), "overscan_noise_adu": ("overscan noise", "ADU"),
    "dc_e_per_pix": ("dark charge (fit)", "e-/pix"), "dc01_e_per_pix": ("dark charge (0e-/1e- counts)", "e-/pix"),
    "dc_rate_e_per_pix_day": ("dark current", "e-/pix/day"), "dc_row_slope_e_per_pix_day": ("dark current from the rows (slope)", "e-/pix/day"),
    "dc_row_intercept_e_per_pix": ("readout charge (row-fit intercept)", "e-/pix"), "chi2ndf": ("dark-current fit chi2/ndf", ""),
    "glow_index": ("glow index", ""), "hot_columns": ("hot columns", ""), "hot_column_list": ("hot column numbers", ""),
    "usable_columns_pct": ("columns not hot", "%"), "first_column_artefact": ("first-column artefact", ""),
    "baseline_row_rms_adu": ("baseline row-to-row rms", "ADU"), "baseline_max_step_adu": ("largest baseline step", "ADU"),
    "baseline_step_nsigma": ("largest baseline step / row-to-row rms", ""),
    "baseline_drift_adu": ("baseline drift over the image", "ADU"), "saturated_pixels": ("saturated pixels", ""),
    "crosstalk_max": ("largest crosstalk into it (>= 5 sigma)", "e-/e-"), "crosstalk_from": ("from", ""),
    "crosstalk_max_any": ("largest crosstalk coefficient (any)", "e-/e-"), "clustering_chain": ("clustering chain", ""),
    "clusters": ("clusters", ""), "image": ("image", ""), "nskips": ("skips", ""), "exposure_s": ("exposure", "s"),
    "readout_s": ("readout", "s"), "images": ("images of this CCD", ""),
}
IMAGE_SECTIONS = [
    ("Gain", ["gain", "gain_err", "gain_vs_module_pct", "gain_vs_nominal_pct", "gain_measured"]),
    ("Noise", ["read_noise_e", "read_noise_gain", "noise_adu", "overscan_noise_adu"]),
    ("Dark current", ["dc_rate_e_per_pix_day", "dc_e_per_pix", "dc01_e_per_pix", "dc_row_slope_e_per_pix_day",
                      "dc_row_intercept_e_per_pix", "chi2ndf", "glow_index"]),
    ("Defects", ["hot_columns", "hot_column_list", "usable_columns_pct", "first_column_artefact", "saturated_pixels"]),
    ("Baseline", ["baseline_step_nsigma", "baseline_max_step_adu", "baseline_row_rms_adu", "baseline_drift_adu"]),
    ("Crosstalk", ["crosstalk_max", "crosstalk_from", "crosstalk_max_any"]),
    ("Image", ["image", "nskips", "exposure_s", "readout_s", "clustering_chain", "clusters", "images"]),
]
IMAGE_DEFS = [
    ("verdict", "Worst of the CCD's checks against the criteria (fail > warn > pass), on the latest image of this "
                "setting; the History table in the panel lists every image of the CCD."),
    ("read noise", "Width of the electron peaks in the dark-current fit [e-] (images whose peaks are resolved); pedestal "
                   "noise [ADU] for every image."),
    ("gain", "Spacing of the electron peaks [ADU/e-]; vs module = against the median of the module's CCDs in the same "
             "image (each relative to its nominal gain). A gain taken from a prior (no comb) is not judged and is "
             "flagged on images with enough skips to resolve the peaks."),
    ("dark current", "Fit lambda / (exposure + half the readout), in e-/pix/day; row fit: dark charge per row (0e- "
                     "fraction) against the row's waiting time: slope = rate, intercept = charge made during the readout."),
    ("glow index", "Dark charge in the last 10 % of the active columns (next to the amplifier) / the CCD median."),
    ("hot columns", "Columns whose dark charge (0e- fraction) is > 5 robust sigma and > 0.05 e-/pix above a running "
                    "median over 31 columns."),
    ("baseline", "Overscan level per row: the largest step between rows in units of the robust rms of the row-to-row "
                 "changes (judged; the same meaning at any number of skips), the step and the rms in ADU, the drift [ADU]."),
    ("crosstalk", "Charge in a CCD at the pixels where another CCD of the module has > 200 e- (where it has no event of "
                  "its own): slope with an intercept, e- per e-; judged only at >= 5 sigma."),
    ("clustering chain", "What the CCD was clustered with (--clusterer auto: A_GMX + Bayesian switch, crowded, "
                         "unmodelled, threshold)."),
]


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def record_time(rec):
    t = (rec.get("meta") or {}).get("DATEINI")
    if t:
        return str(t)
    m = re.search(r"_(\d{8})_(\d{6})", rec.get("stem", ""))
    return f"{m.group(1)}T{m.group(2)}" if m else ""


def setting_of(meta):
    nd = meta.get("NDCM", meta.get("NSKIPS", "?"))
    return f"{nd}-skip {meta.get('NPBIN', 1) or 1}x{meta.get('NSBIN', 1) or 1}"


def readout_seconds(meta):
    try:
        t0 = datetime.datetime.fromisoformat(str(meta["DATEINI"]))
        t1 = datetime.datetime.fromisoformat(str(meta["DATEEND"]))
        return (t1 - t0).total_seconds()
    except (KeyError, ValueError, TypeError):
        return None


def image_values(rec, ext, crit):
    """(values, flags, status) of one CCD of one image record."""
    s = rec["summary"]["ccds"].get(ext, {})
    meta = rec.get("meta") or {}
    if "gain" not in s:
        return {}, [], s.get("skipped") or "not processed"
    ch = (((rec.get("char") or {}).get("ccds") or {}).get(ext) or {})
    g = s["gain"]
    resolved = bool(s.get("resolved"))
    nsk = int(meta.get("NDCM", meta.get("NSKIPS", 1)) or 1)
    v = dict(gain=g, gain_err=s.get("gain_err"), noise_adu=s["noise_e"] * g, nskips=nsk, image=rec["stem"],
             exposure_s=_num(meta.get("EXPOSURE")), readout_s=readout_seconds(meta))
    gsrc = (ch.get("gain") or {}).get("source")
    measured = (gsrc in (None, "comb")) and "overridden" not in (s.get("gain_note") or "") and resolved
    v["gain_measured"] = measured
    expect = nsk >= int(crit.get("expect_resolved_min_skips", 100))
    if resolved or expect:
        # with enough skips the peaks should be resolved: an unresolved CCD is judged on its noise too (in e- with the
        # gain that was used, a prior when the comb could not be measured: 'read noise gain' says which)
        v["read_noise_e"] = s["noise_e"]
        v["read_noise_gain"] = "measured" if measured else "prior (peaks not resolved)" if not resolved else "prior"
    if resolved:
        v.update(dc_e_per_pix=s["dc_e"], dc01_e_per_pix=s.get("lam01"), chi2ndf=s.get("chi2ndf"))
        t_eff = (v["exposure_s"] or 0) + (v["readout_s"] or 0) / 2
        if t_eff > 0:
            v["dc_rate_e_per_pix_day"] = s["dc_e"] / t_eff * 86400
        lr = np.asarray([x if x is not None else np.nan for x in (s.get("lam_rows") or [])], float)
        if lr.size > 10 and v["readout_s"]:
            t_row = (v["exposure_s"] or 0) + np.arange(lr.size) * v["readout_s"] / lr.size
            ok = np.isfinite(lr)
            if ok.sum() > 10:
                sl, ic = np.polyfit(t_row[ok], lr[ok], 1)
                v["dc_row_slope_e_per_pix_day"], v["dc_row_intercept_e_per_pix"] = float(sl * 86400), float(ic)
    v.update({k: x for k, x in (rec.get("metrics") or {}).get(ext, {}).items()})
    nominal = (crit.get("nominal_gain") or {}).get(ext)
    if nominal and measured:
        v["gain_vs_nominal_pct"] = 100 * (g / nominal - 1)
    dec = s.get("decision") or {}
    outs = [x for x in s.values() if isinstance(x, dict) and "n_clusters" in x]
    if outs:
        v["clustering_chain"] = dec.get("chain") or outs[0].get("chain")
        v["clusters"] = outs[0]["n_clusters"]
    fc = (ch.get("first_column") or {}).get("value") or {}
    if fc:
        v["first_column_artefact"] = bool(fc.get("significant"))
    flags = []
    if (ch.get("channel") or {}).get("value") == "quiet":
        flags.append(("channel", "fail", "no charge seen (active region = overscan): a dead channel or an ultra-low "
                                         "dark-charge CCD; check with a light / Fe-55 image", "no charge seen (dead channel?)"))
    if not measured and nsk >= int(crit.get("expect_resolved_min_skips", 100)):
        flags.append(("gain", "warn", f"gain not measured on this CCD ({'peaks not resolved' if not resolved else 'from ' + str(gsrc or 'a prior')})",
                      "gain not measured on the CCD"))
    for t in (s.get("fit_at_limit") or []) if resolved else []:
        flags.append(("fit", "warn", f"dark-current fit: {t}", "fit parameter at a limit"))
    return v, flags, None


def image_rows(records, crit, mods_of, link_prefix=""):
    """Overview views of the images tool: one view per image setting (skips, binning); per CCD the latest image's
    values, verdict and reasons, and the history of all its images."""
    sec = section(crit, "images")
    per = {}                                       # (setting, module, ext) -> [(time, rec, values, flags, status)]
    ccd_name, acm_of = {}, {}
    for rec in records:
        meta = rec.get("meta") or {}
        acm = rec.get("acm")
        exts = sorted(rec["summary"]["ccds"])
        vals = {e: image_values(rec, e, crit) for e in exts}
        # gain vs the module median (measured gains, each relative to its nominal) within this image
        nominal = crit.get("nominal_gain") or {}
        rel = [vals[e][0]["gain"] / nominal.get(e, 1.0) for e in exts if vals[e][0].get("gain_measured")]
        if len(rel) >= 2:
            med = float(np.median(rel))
            for e in exts:
                if vals[e][0].get("gain_measured"):
                    vals[e][0]["gain_vs_module_pct"] = 100 * (vals[e][0]["gain"] / (med * nominal.get(e, 1.0)) - 1)
        xt = rec.get("crosstalk") or {}
        for e in exts:
            cs = [(k, c) for k, c in xt.items() if k.endswith("->" + e)]
            if cs:
                vals[e][0]["crosstalk_max_any"] = max(abs(c[0]) for _, c in cs)
                sig = [(k, c) for k, c in cs if c[1] and abs(c[0]) >= 5 * c[1]]
                if sig:
                    k, c = max(sig, key=lambda kc: abs(kc[1][0]))
                    vals[e][0]["crosstalk_max"], vals[e][0]["crosstalk_from"] = c[0], k.split("->")[0]
            ccd = meta.get(f"CH{int(e[3:]) - 1}") or f"ch{int(e[3:]) - 1}"
            mod = mods_of(acm, int(e[3:]), ccd)
            ccd_name[(mod, e)], acm_of[(mod, e)] = ccd, acm
            per.setdefault((setting_of(meta), mod, e), []).append((record_time(rec), rec, *vals[e]))
    views = {}
    for (setting, mod, e), items in sorted(per.items()):
        items.sort(key=lambda it: (it[0], it[1]["stem"]))
        hist_rows = []
        for t, rec, v, flags, status in items:
            verdict, reasons, _ = judge(v, sec, flags, status)
            hist_rows.append([rec["stem"], t[:19], v.get("gain"), v.get("read_noise_e"), v.get("noise_adu"),
                              v.get("dc_rate_e_per_pix_day"), v.get("hot_columns"), verdict])
        t, rec, v, flags, status = items[-1]
        v = dict(v, images=len(items))
        verdict, reasons, levels = judge(v, sec, flags, status)
        links = [] if status else [["DC fit (PDF)", f"{link_prefix}{rec['stem']}_FitDarkCurrentProcess_DCfit_{e}.pdf"]]
        views.setdefault(setting, []).append(dict(
            module=mod, acm=acm_of[(mod, e)], ext=e, ccd=ccd_name[(mod, e)], verdict=verdict, status=status, reasons=reasons,
            levels=levels, values=v, when=t[:19], links=links, stamp=rec["stem"],
            history=dict(head=["image", "start", "gain", "noise [e-]", "noise [ADU]", "DC [e-/pix/day]", "hot cols", "verdict"],
                         rows=hist_rows) if len(items) > 1 else None))
    # the setting with the most resolved CCDs first
    order = sorted(views, key=lambda k: (-sum(1 for r in views[k] if "read_noise_e" in r["values"]), k))
    out = [dict(name=k, ccds=views[k], note=f"image setting {k}: {len({(r['module']) for r in views[k]})} module(s)")
           for k in order]
    matrices = {}
    for rec in records:
        if rec.get("crosstalk"):
            e0 = sorted(rec["summary"]["ccds"])[0]
            meta = rec.get("meta") or {}
            mod = mods_of(rec.get("acm"), int(e0[3:]), meta.get(f"CH{int(e0[3:]) - 1}"))
            prev = matrices.get(mod)
            if prev is None or record_time(rec) >= prev[0]:
                matrices[mod] = (record_time(rec), rec["crosstalk"])
    return out, {m: x for m, (_, x) in matrices.items()}, sec


def images_overview(camp, records, crit, mods_of, path, back):
    views, matrices, sec = image_rows(records, crit, mods_of, link_prefix="../")
    data = dict(title="Images: fits and CCD quality", sub=sub_line(camp, f"{len(records)} image(s)"), back=back,
                criteria=sec, fields={k: dict(label=a, unit=b) for k, (a, b) in IMAGE_FIELDS.items()},
                sections=IMAGE_SECTIONS, cell=["read_noise_e", "gain", "dc_rate_e_per_pix_day"], views=views,
                review=dict(tool="images"),
                columns=["gain", "gain_vs_module_pct", "read_noise_e", "noise_adu", "dc_rate_e_per_pix_day",
                         "dc_row_intercept_e_per_pix", "chi2ndf", "glow_index", "hot_columns", "baseline_step_nsigma",
                         "crosstalk_max", "crosstalk_from", "saturated_pixels", "clustering_chain", "clusters", "image", "images"],
                definitions=IMAGE_DEFS, matrices=matrices or None, matrix_title="Crosstalk between the CCDs of a module",
                matrix_note="Charge seen in the victim (column) per electron in the source (row), x1e-4, latest image of "
                            "each module. Coloured: >= 5 sigma and above the warn / fail limit; grey: not significant.",
                flag_labels={"channel": "no charge seen", "gain": "gain not measured on the CCD", "fit": "fit parameter at a limit"},
                csv_name="images_overview",
                gridnote="Each cell: the verdict, read noise, gain and dark current of the CCD's latest image of this "
                         "setting (amber / red when outside a limit) and its review status. 'Cells show' maps any "
                         "parameter over all CCDs. Click a cell for the reasons, the reviews, the links and every value.")
    write_overview(path, data)
    return views


# =============================================================================
# the campaign index: every tool per CCD
# =============================================================================
TOOL_LABELS = {"images": "Images", "packet": "Packets", "dc_series": "Dark current", "defects": "Defect maps",
               "sr": "Serial register"}
TOOL_SHORT = {"images": "IM", "packet": "PK", "dc_series": "DC", "defects": "DF", "sr": "SR"}


# the numbers in the campaign cells: (key, label, short, unit, [(tool, value), ...] - the first tool that has it)
INDEX_MAIN = [("noise", "read noise (images, else packet)", "noise", "e-", [("images", "read_noise_e"), ("packet", "read_noise_e")]),
              ("gain", "gain (images, else packet)", "gain", "ADU/e-", [("images", "gain"), ("packet", "gain")]),
              ("dc", "dark current (latest image)", "DC", "e-/pix/day", [("images", "dc_rate_e_per_pix_day")]),
              ("dc_rate", "dark-current rate of the series (settling plateau, else the latest image)", "DC rate",
               "e-/pix/day", [("dc_series", "best_rate")]),
              ("sr_dc", "dark-current rate of the serial-register images (median)", "SR DC", "e-/pix/day",
               [("sr", "dc_rate_median")])]
INDEX_SOURCES = {(t, k) for *_, src in INDEX_MAIN for t, k in src}


def tool_fields(tool):
    """{key: (label, unit)} of a tool's overview values."""
    if tool == "images":
        return IMAGE_FIELDS
    mod = {"packet": "packet_report", "dc_series": "dc_series", "defects": "defects", "sr": "sr_images"}[tool]
    return __import__(mod).FIELDS


def campaign_index(camp, tool_views, path, crit):
    """DIR/index.html: one row per CCD with every tool's verdict (worst over the tool's views), the reasons
    (prefixed with the tool) and links to the tools' overviews. tool_views: {tool: (href, [views])}."""
    per, tools = {}, []
    fields, sections, metrics, cell = {}, [], {}, []
    for tool, (href, views) in tool_views.items():
        T = TOOL_LABELS.get(tool, tool)
        tf, tm = tool_fields(tool), (crit.get(tool) or {}).get("metrics") or {}
        keys = []
        for v in views:
            for r in v["ccds"]:
                keys += [k for k, x in (r.get("values") or {}).items() if k not in keys]
        for k in keys:
            lab, un = tf.get(k, (k.replace("_", " "), ""))
            fields[f"{tool}.{k}"] = dict(label=f"{T}: {lab}", unit=un, plabel=lab,
                                         short=f"{TOOL_SHORT.get(tool, tool[:2])} {SHORT.get(k, lab)}")
            if k in tm and (tool, k) in INDEX_SOURCES:
                metrics[f"{tool}.{k}"] = dict(tm[k], label=f"{T}: {tm[k].get('label', k)}")
                for key, lab_m, *_, src in INDEX_MAIN:
                    if src[0] == (tool, k):
                        metrics[f"main.{key}"] = dict(tm[k], label=lab_m)
        main_view = views[0]["name"] if views else ""
        sections.append([T + (f" ({main_view})" if len(views) > 1 else ""), [f"{tool}.{k}" for k in keys]])
        rows = [r for v in views for r in v["ccds"]]
        n = counts(rows)
        tools.append(dict(label=TOOL_LABELS.get(tool, tool), href=href, n=n,
                          note=f"{len(views)} view(s): " + ", ".join(v["name"] for v in views) if len(views) > 1 else ""))
        for v in views:
            for r in v["ccds"]:
                k = (r["module"], r["ext"])
                c = per.setdefault(k, dict(module=r["module"], ext=r["ext"], acm=r.get("acm"), ccd=r.get("ccd"), tools={},
                                           reasons=[], values={}))
                c["ccd"] = c["ccd"] or r.get("ccd")
                if v is views[0] or not any(k.startswith(tool + ".") for k in c["values"]):   # values: the main view
                    for k, x in (r.get("values") or {}).items():
                        c["values"][f"{tool}.{k}"] = x
                    for k, x in (r.get("levels") or {}).items():
                        c.setdefault("levels", {})[f"{tool}.{k}"] = x
                t = c["tools"].setdefault(TOOL_LABELS.get(tool, tool), dict(verdict="na", reasons=[], short=TOOL_SHORT.get(tool, tool[:2]),
                                                                            href=f"{href}#{r['module']}%20{r['ext']}", reports=[],
                                                                            tool=tool, stamp=r.get("stamp")))
                # the CCD's full reports of this tool (module page, defect map, fit PDF), relative to the campaign folder
                base = posixpath.dirname(href)
                for lab_, h_ in r.get("links") or []:
                    if not re.match(r"^[a-z]+:|^/", h_):
                        h_ = posixpath.normpath(posixpath.join(base, h_))
                    name = {"module page": "full report"}.get(lab_, lab_) + (f" ({v['name']})" if len(views) > 1 else "")
                    if [name, h_] not in t["reports"]:
                        t["reports"].append([name, h_])
                rank = {"fail": 3, "warn": 2, "pass": 1, "na": 0}
                if "dataset" in v:                     # data sets: the newest one with the CCD gives the verdict
                    if t.setdefault("dataset", v["dataset"]) != v["dataset"]:
                        continue
                    t["note"] = f"data set {v['dataset'] or '(none)'}; older: see the tool's overview"
                if rank[r["verdict"]] > rank[t["verdict"]]:
                    t["verdict"] = r["verdict"]
                for rr in r.get("reasons", []):
                    lab = f"{TOOL_LABELS.get(tool, tool)}: {rr[3] if len(rr) > 3 else rr[1]}"
                    txt = f"[{TOOL_LABELS.get(tool, tool)}{(' · ' + v['name']) if len(views) > 1 else ''}] {rr[2]}"
                    c["reasons"].append([rr[0], rr[1], txt, lab])
    rows = []
    for key, lab, short, un, src in INDEX_MAIN:      # the main numbers: one per CCD, from the first tool that has it
        got = False
        for c in per.values():
            for tool, k in src:
                v = c["values"].get(f"{tool}.{k}")
                if v is not None:
                    c["values"][f"main.{key}"] = v
                    if (c.get("levels") or {}).get(f"{tool}.{k}"):
                        c["levels"][f"main.{key}"] = c["levels"][f"{tool}.{k}"]
                    got = True
                    break
        if got:
            fields[f"main.{key}"] = dict(label=lab, unit=un, plabel=lab, short=short)
            cell.append(f"main.{key}")
    if cell:
        sections.insert(0, ["Main numbers", list(cell)])
    for c in per.values():
        vs = [t["verdict"] for t in c["tools"].values()]
        c["verdict"] = "fail" if "fail" in vs else "warn" if "warn" in vs else "pass" if "pass" in vs else "na"
        c["reasons"].sort(key=lambda r: {"fail": 0, "warn": 1}.get(r[0], 2))
        c["links"] = []                              # per tool in the panel: the overview and the full reports
        c["stamp"] = " | ".join(f"{k}: {t.get('stamp')}" for k, t in sorted(c["tools"].items()))
        rows.append(c)
    runs = camp.data.get("runs", [])
    defs = [("verdict", "Worst over the tools that have data for the CCD (fail > warn > pass). The letters in a cell are "
                        "the tools (IM images, PK packets, DC dark current, DF defect maps, SR serial register) in "
                        "their verdict's colour.")] + [
        (f"run {i + 1}: {r['date']}", f"{r['command']} — {r['new']} new input(s), {r['skipped']} already in the campaign")
        for i, r in enumerate(runs[-30:])]
    data = dict(title=f"Commissioning campaign · {os.path.basename(camp.root)}", sub=sub_line(camp), back=[],
                criteria=dict(name=crit.get("name"), version=crit.get("version"), note=crit.get("note"), metrics=metrics),
                tools=tools, views=[dict(name="all tools", ccds=rows)], fields=fields, sections=sections, cell=cell,
                columns=[k for _, ks in sections for k in ks if not isinstance(next((r["values"].get(k) for r in rows
                                                                                  if r["values"].get(k) is not None), 0), list)],
                definitions=defs, csv_name="campaign_overview",
                gridnote="Each cell: the CCD's worst verdict, the tools' letters in their verdict's colour, the review "
                         "status, read noise and gain (images, else packet), dark current (latest image), the dark-current "
                         "rate of the series (settling plateau, else the latest image) and of the serial-register images. 'Cells show' maps any "
                         "value over all CCDs. Click a cell for the reasons, the reviews, every tool's links and values.")
    write_overview(path, data)
    return rows
