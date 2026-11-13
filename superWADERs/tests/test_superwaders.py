"""superWADERs: independence from the rest of comm_scripts, and the command line.

    python -m unittest discover -s superWADERs/tests -v        (from the comm_scripts folder)
"""
import glob
import json
import os
import subprocess
import sys
import tempfile
import unittest

SW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMD = os.path.join(SW, "run_superWADERs")
MODULE_DIRS = ("engine", "clustering", "commissioning")


def modules():
    """every module of superWADERs (engine/, clustering/, commissioning/; not the copied third-party scripts that
    need extra packages: raw_levels' viewers, lsm_defects' Image_4, compare_dcfit_vs_skips' pypdf)"""
    out = []
    for d in MODULE_DIRS:
        for f in sorted(glob.glob(os.path.join(SW, d, "*.py"))):
            name = os.path.basename(f)[:-3]
            if name not in ("compare_dcfit_vs_skips",):
                out.append(name)
    return out


class Independence(unittest.TestCase):
    def test_imports_come_from_superwaders(self):
        """Every superWADERs module, and everything they import from comm_scripts, is loaded from superWADERs/
        (never from src/, clustering_tests/ or wip/), even with those folders first on the path."""
        cs = os.path.dirname(SW)
        code = f"""
import importlib, json, os, sys
for d in ("src", "clustering_tests", "wip/lowdc/bayes", "wip/native_pipeline"):
    sys.path.insert(0, os.path.join({cs!r}, d))                # the copies elsewhere first: must not be used
sys.path.insert(0, os.path.join({SW!r}, "engine"))
import native_pipeline                                           # sets superWADERs' own path
bad, failed = {{}}, {{}}
for m in {modules()!r}:
    try:
        importlib.import_module(m)
    except ImportError as e:
        failed[m] = str(e)
for name, mod in list(sys.modules.items()):
    f = getattr(mod, "__file__", None) or ""
    if f.startswith({cs!r}) and not f.startswith({SW!r}):
        bad[name] = f
print(json.dumps(dict(bad=bad, failed=failed)))
"""
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        res = json.loads(r.stdout.strip().splitlines()[-1])
        missing = {m: e for m, e in res["failed"].items() if "ROOT" not in e}
        self.assertEqual(missing, {}, "modules that cannot be imported")
        self.assertEqual(res["bad"], {}, "modules loaded from outside superWADERs/")

    def test_no_reference_to_run_wader_or_src(self):
        """No superWADERs code starts or imports bin/run_WADERs, or reads files from src/, clustering_tests/, wip/."""
        for f in glob.glob(os.path.join(SW, "**", "*.py"), recursive=True) + [CMD]:
            if "/tests/" in f:
                continue
            with open(f) as fh:
                text = "".join(ln for ln in fh if not ln.lstrip().startswith("#"))
            for pat in ('"run_WADERs"', "'run_WADERs'", '"src")', '"clustering_tests")', '"wip"'):
                self.assertNotIn(pat, text, f"{f}: {pat}")


class CommandLine(unittest.TestCase):
    def run_cmd(self, *args):
        return subprocess.run([sys.executable, CMD, *args], capture_output=True, text=True)

    def test_help(self):
        r = self.run_cmd("-h")
        self.assertEqual(r.returncode, 0)
        self.assertIn("--clusterer {auto,agmx,threshold}", r.stdout)
        r = self.run_cmd("-h", "-v")
        self.assertIn("auto's chains", r.stdout)

    def test_default_clusterer_is_auto(self):
        sys.path.insert(0, SW)
        import runpy
        ns = runpy.run_path(CMD, run_name="not_main")
        self.assertEqual(ns["build_parser"]().parse_args(["x.fits"]).clusterer, "auto")

    def test_bad_requests(self):
        with tempfile.TemporaryDirectory() as d:
            img = os.path.join(d, "avg_Image_5_x_106_20260101_000000_1.fits")
            open(img, "w").close()
            for args, msg in [(["--thresholds", "x"], "--thresholds"),
                              (["--autoconfig-args", "--gain 1.3"], "cannot be given"),
                              (["--composite", "--no-clusters"], "--no-clusters"),
                              (["--packet", "--dc-series"], "--packet"),
                              (["--campaign", os.path.join(d, "newcamp"), "--overwrite"], "--overwrite"),
                              (["--criteria", os.path.join(d, "nope.json")], "--criteria")]:
                r = self.run_cmd(img, "-o", d, *args)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(msg, r.stderr)
            r = self.run_cmd(os.path.join(d, "missing.fits"))
            self.assertIn("not found", r.stderr)


class Campaign(unittest.TestCase):
    """The campaign store and the verdicts (commissioning/campaign.py)."""

    def setUp(self):
        for d in MODULE_DIRS:
            p = os.path.join(SW, d)
            if p not in sys.path:
                sys.path.insert(0, p)
        import campaign
        self.CA = campaign

    def test_judge(self):
        sec = dict(metrics=dict(noise=dict(label="noise", unit="e-", warn=0.2, fail=0.3),
                                gain_dev=dict(label="gain dev", unit="%", warn=3, fail=8, abs=True),
                                show=dict(label="shown", warn=None, fail=None),
                                low=dict(label="low", warn=10, fail=5, direction="below")),
                   flag_levels=dict(dead="fail", info="note", gone="off"))
        J = self.CA.judge
        self.assertEqual(J(dict(noise=0.1, gain_dev=-2, show=99, low=20), sec)[0], "pass")
        v, r, lv = J(dict(noise=0.25, gain_dev=-9, low=7), sec)
        self.assertEqual(v, "fail")
        self.assertEqual([x[1] for x in r], ["gain_dev", "noise", "low"])   # fail first
        self.assertEqual(lv, dict(noise="warn", gain_dev="fail", low="warn"))
        self.assertEqual(J(dict(noise=0.1), sec, [("info", "warn", "a note", "info")])[0], "pass")   # note: no effect
        self.assertEqual(J(dict(noise=0.1), sec, [("gone", "fail", "dropped", "gone")])[1], [])       # off: dropped
        self.assertEqual(J(dict(noise=0.1), sec, [("dead", "warn", "no charge", "dead")])[0], "fail")  # level from file
        self.assertEqual(J({}, sec, status="no CCD signal")[0], "na")

    def test_ledger_and_records(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "a.fits")
            open(f, "w").write("x")
            c = self.CA.Campaign(os.path.join(d, "camp"))
            self.assertFalse(c.current(f))
            c.add(f, "images", kind="image", stem="a")
            c.save_record("images", "a", dict(stem="a", x=[1.5, float("nan")]))
            c.save()
            c2 = self.CA.Campaign(os.path.join(d, "camp"))
            self.assertTrue(c2.has(f, "images"))
            self.assertFalse(c2.has(f, "sr"))
            self.assertEqual(c2.load_record("images", "a"), dict(stem="a", x=[1.5, None]))
            self.assertEqual(c2.stems("images"), ["a"])
            open(f, "w").write("changed")                    # a changed file is new again
            os.utime(f, ns=(1, 1))
            self.assertFalse(c2.current(f))

    def test_criteria_copy(self):
        with tempfile.TemporaryDirectory() as d:
            crit = self.CA.load_criteria(d)
            self.assertIn("images", crit)
            self.assertTrue(os.path.exists(os.path.join(d, "criteria.json")))
            for t in ("images", "packet", "dc_series", "defects", "sr"):
                self.assertIn("metrics", crit[t])

    def test_overview_page(self):
        self.CA.ACTIVE = None
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "o", "index.html")
            row = dict(module="DM01", acm=101, ext="EXT1", ccd="CCD_A", verdict="warn", values=dict(noise=0.25),
                       reasons=[["warn", "noise", "noise 0.25 > 0.2", "noise"]], levels=dict(noise="warn"))
            self.CA.write_overview(path, dict(title="t", criteria=dict(metrics=dict(noise=dict(label="noise", warn=0.2))),
                                              views=[dict(name="v", ccds=[row])]))
            with open(path) as fh:
                page = fh.read()
            self.assertIn('"module":"DM01"', page)
            self.assertNotIn("__DATA__", page)
            self.assertNotIn("__REVIEW", page)
            self.assertIn('<script src="reviews.js">', page)      # no campaign: next to the page

    def test_review_paths(self):
        """Every page finds the campaign's reviews.js from where it is."""
        with tempfile.TemporaryDirectory() as d:
            c = self.CA.Campaign(os.path.join(d, "camp"))
            self.assertTrue(c.data["id"].startswith("camp@"))
            for rel, root in (("index.html", ""), ("images/index.html", "../"), ("defects/G/defect_map.html", "../../")):
                ctx = self.CA.review_context(os.path.join(c.root, rel), "images")
                self.assertEqual((ctx["root"], ctx["campaign"]), (root, c.data["id"]))


    def test_datasets(self):
        """--dataset: rows of <module>@<data set> go to their data set's view, newest first."""
        CA = self.CA
        with tempfile.TemporaryDirectory() as d:
            c = CA.Campaign(os.path.join(d, "camp"))
            for name, ds, added in (("a.fits", "", "2026-01-01"), ("b.fits", "retest", "2026-02-01")):
                f = os.path.join(d, name)
                open(f, "w").write(name)
                c.add(f, "packet", kind="image", stem=name[:-5], dataset=ds)
                c.entry(f)["added"] = added
            self.assertEqual(c.dataset_order("packet"), ["retest", ""])
            self.assertEqual(c.dataset_of_stem("packet"), {"a": "", "b": "retest"})
            c.add(os.path.join(d, "b.fits"), "packet")              # no --dataset: keeps its data set
            self.assertEqual(c.dataset(os.path.join(d, "b.fits"), "packet"), "retest")
            self.assertEqual(CA.split_module(CA.ds_module("DM01", "retest")), ("DM01", "retest"))
            self.assertEqual(CA.dataset_name("cool down #2"), "cool-down-2")
            rows = [dict(module="DM01", ext="EXT1"), dict(module="DM01@retest", ext="EXT1")]
            v = CA.split_datasets([dict(name="packet", ccds=rows)], c.dataset_order("packet"))
            self.assertEqual([(x["name"], x["dataset"], x["ccds"][0]["module"]) for x in v],
                             [("retest", "retest", "DM01"), ("without data set", "", "DM01")])
            plain = [dict(name="packet", ccds=rows[:1])]
            self.assertIs(CA.split_datasets(plain, ["retest", ""]), plain)
            # the campaign index: the newest data set gives the CCD's verdict
            v[0]["ccds"][0].update(verdict="pass", reasons=[])
            v[1]["ccds"][0].update(verdict="fail", reasons=[["fail", "noise", "old noise", "noise"]])
            idx = CA.campaign_index(c, {"packet": ("packet/index.html", v)}, os.path.join(c.root, "index.html"), {})
            self.assertEqual(idx[0]["verdict"], "pass")


class Reviews(unittest.TestCase):
    """commissioning/reviews.py: merge, import, the files the pages read; run_superWADERs --campaign DIR (no INPUT)."""

    def setUp(self):
        p = os.path.join(SW, "commissioning")
        if p not in sys.path:
            sys.path.insert(0, p)
        import reviews
        self.R = reviews

    def camp(self, d):
        root = os.path.join(d, "camp")
        os.makedirs(root)
        json.dump(dict(id="camp@x", inputs={}, runs=[]), open(os.path.join(root, "campaign.json"), "w"))
        return root

    def test_merge(self):
        a = dict(id="1", created="t1", updated="t1", status="note")
        b = dict(id="1", created="t1", updated="t2", status="concern")         # edited later: wins
        c = dict(id="2", created="t0", updated="t0", status="approved")
        d = dict(id="2", created="t0", updated="t3", deleted=True)             # deleted later: stays deleted
        m = self.R.merge([b, a, c, d])
        self.assertEqual([r["id"] for r in m], ["2", "1"])
        self.assertEqual(m[1]["status"], "concern")
        self.assertEqual(self.R.live(m), [b])

    def test_files_and_import(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.camp(d)
            r1 = self.R.new_review(root, "Ana", "concern", "noise high", dict(tool="campaign", module="DM01", ext="EXT2"))
            self.R.save(root, [r1], "Ana")
            dl = os.path.join(d, "reviews_Ben.json")                           # as downloaded from a page
            r2 = dict(r1, id="b", reviewer="Ben", status="approved", comment="fine", created="z", updated="z")
            json.dump(dict(format=self.R.FORMAT, campaign="camp@x", reviews=[r2]), open(dl, "w"))
            lines = self.R.refresh(root, [dl], log=lambda *a: None)
            self.assertIn("2 (1 approved, 1 concern) by Ana, Ben", lines[0])
            self.assertEqual(self.R.import_files(root, [dl], log=lambda *a: None), 0)   # the same file again: nothing new
            self.assertFalse(os.path.exists(os.path.join(root, "reviews")))           # one file: reviews.js
            js = open(os.path.join(root, "reviews.js")).read()
            self.assertTrue(js.startswith("window.SW_REVIEWS = "))
            self.assertEqual(len(self.R.read_file(os.path.join(root, "reviews.js"))), 2)
            # a reviews.js downloaded from a page (the campaign's file + a new review) merges in; a deletion wins
            gone = dict(r1, deleted=True, updated="zz")
            dl2 = os.path.join(d, "reviews.js")
            open(dl2, "w").write(self.R.store_text("camp@x", [gone, dict(r2, id="c", created="zz", updated="zz")]))
            self.assertEqual(self.R.import_files(root, [dl2], log=lambda *a: None), 2)
            self.assertEqual(sorted(r["id"] for r in self.R.live(self.R.load(root))), ["b", "c"])
            with open(os.path.join(root, "reviews.csv")) as fh:
                self.assertEqual(len(fh.read().strip().splitlines()), 3)
            with self.assertRaises(ValueError):
                self.R.new_review(root, "Ana", "maybe", "", {})

    def test_command_line(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.camp(d)
            r = subprocess.run([sys.executable, os.path.join(SW, "commissioning", "reviews.py"), root, "--add",
                                "--reviewer", "Cy", "--status", "rejected", "--module", "DM03", "--ext", "EXT1",
                                "--comment", "dead amplifier"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("DM03", r.stdout)
            r = subprocess.run([sys.executable, CMD, "--campaign", root, "--no-review-helper"], capture_output=True, text=True)   # no INPUT
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("1 rejected", r.stdout)
            self.assertTrue(os.path.exists(os.path.join(root, "review_server.py")))   # travels with the folder
            r = subprocess.run([sys.executable, CMD, "--campaign", os.path.join(d, "nothing")], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("not a campaign folder", r.stderr)


class ReviewServer(unittest.TestCase):
    """review_server.py: pages served, reviews posted by a page saved at once; other sites refused."""

    @staticmethod
    def R_read(root):
        sys.path.insert(0, os.path.join(SW, "commissioning"))
        import reviews
        return reviews.read_file(os.path.join(root, "reviews.js"))

    def test_post(self):
        import socket
        import time
        import urllib.request
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "camp")
            os.makedirs(root)
            json.dump(dict(id="camp@x", inputs={}, runs=[]), open(os.path.join(root, "campaign.json"), "w"))
            open(os.path.join(root, "index.html"), "w").write("<html>ok</html>")
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            s.close()
            pr = subprocess.Popen([sys.executable, os.path.join(SW, "commissioning", "reviews.py"), root, "--serve",
                                   "--port", str(port), "--no-browser"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                url = f"http://127.0.0.1:{port}"
                for _ in range(50):
                    try:
                        urllib.request.urlopen(url + "/index.html")
                        break
                    except OSError:
                        time.sleep(0.1)
                self.assertIn(b"ok", urllib.request.urlopen(url + "/index.html").read())
                rev = dict(id="r1", campaign="camp@x", created="t", updated="t", reviewer="Ana", status="approved",
                           comment="fine", target=dict(tool="campaign", module="DM01", ext="EXT1"))
                body = json.dumps(dict(reviewer="Ana", reviews=[rev])).encode()
                bad = urllib.request.Request(url + "/api/reviews", data=body, headers={"Content-Type": "text/plain"})
                with self.assertRaises(urllib.error.HTTPError):          # not from a page (no header): refused
                    urllib.request.urlopen(bad)
                ok = urllib.request.Request(url + "/api/reviews", data=body,
                                            headers={"Content-Type": "application/json", "X-SW-Reviews": "1"})
                j = json.loads(urllib.request.urlopen(ok).read())
                self.assertEqual([r["id"] for r in j["reviews"]], ["r1"])
                self.assertIn('"r1"', open(os.path.join(root, "reviews.js")).read())
                urllib.request.urlopen(urllib.request.Request(url + "/api/reviews", data=json.dumps(dict(reviews=[dict(
                    rev, id="r2")])).encode(), headers={"Content-Type": "application/json", "X-SW-Reviews": "1"}))
                self.assertEqual(sorted(r["id"] for r in self.R_read(root)), ["r1", "r2"])   # both in the one file
                self.assertFalse(os.path.exists(os.path.join(root, "reviews")))
                j = json.loads(urllib.request.urlopen(url + "/api/reviews").read())
                self.assertTrue(j["server"])
                # a page opened by double-click (origin null): only with the token of review_server_info.js
                info = open(os.path.join(root, "review_server_info.js")).read()
                tok = json.loads(info.split("=", 1)[1].strip().rstrip(";"))["token"]
                hdr = {"Content-Type": "application/json", "X-SW-Reviews": "1", "Origin": "null"}
                body3 = json.dumps(dict(reviews=[dict(rev, id="r3")])).encode()
                with self.assertRaises(urllib.error.HTTPError):
                    urllib.request.urlopen(urllib.request.Request(url + "/api/reviews", data=body3, headers=hdr))
                r = urllib.request.urlopen(urllib.request.Request(url + "/api/reviews", data=body3,
                                                                  headers=dict(hdr, **{"X-SW-Token": tok})))
                self.assertEqual(r.headers.get("Access-Control-Allow-Origin"), "null")
                with self.assertRaises(urllib.error.HTTPError):      # the token file itself is not served
                    urllib.request.urlopen(url + "/review_server_info.js")
                self.assertTrue(os.path.exists(os.path.join(root, "review_server.py")))
            finally:
                pr.terminate()
                pr.communicate(timeout=10)
            self.assertFalse(os.path.exists(os.path.join(root, "review_server_info.js")))   # removed when it stops


class ReviewHelper(unittest.TestCase):
    """run_superWADERs starts the review helper in the background once; a second start finds it running."""

    def test_helper(self):
        import signal
        sys.path.insert(0, os.path.join(SW, "commissioning"))
        import reviews as R
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "camp")
            os.makedirs(root)
            json.dump(dict(id="camp@h", inputs={}, runs=[]), open(os.path.join(root, "campaign.json"), "w"))
            url = R.ensure_helper(root, port=8890, log=lambda *a: None)
            try:
                self.assertTrue(url and url.startswith("http://127.0.0.1:"))
                self.assertEqual(R.ensure_helper(root, port=8890, log=lambda *a: None), url)   # reused
                self.assertEqual(R.helper_url(root), url)
            finally:
                info = open(os.path.join(root, R.INFO)).read()
                os.kill(json.loads(info.split("=", 1)[1].strip().rstrip(";"))["pid"], signal.SIGTERM)
            import time
            for _ in range(30):
                if not os.path.exists(os.path.join(root, R.INFO)):
                    break
                time.sleep(0.1)
            self.assertIsNone(R.helper_url(root))


if __name__ == "__main__":
    unittest.main()
